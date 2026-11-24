"""
scripts/benchmark_inference.py — CPU Latency & Throughput Benchmark.

Compares PyTorch CPU fallback vs. ONNX Runtime CPU inference across:
  - Short requests (single window <= 126 tokens)
  - Long padded requests (multi-window sliding scanner > 128 tokens)
Measures p50, p95, p99 latency (ms), throughput (req/sec), disk footprint, and speedup factor.
"""

import sys
import time
import platform
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from proxy.normalizer import normalize_request
from proxy.scanner import sliding_score

MODEL_DIR = PROJECT_ROOT / "waf-distilbert-final"
ONNX_PATH = MODEL_DIR / "model.onnx"


def run_benchmark():
    print("=" * 70)
    print("  ONNX RUNTIME VS. PYTORCH CPU INFERENCE BENCHMARK")
    print("=" * 70)

    # 1. System Info
    cpu_name = platform.processor() or "Unknown CPU"
    print(f"  OS                     : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python                 : {platform.python_version()}")
    print(f"  PyTorch Version        : {torch.__version__}")
    print(f"  ONNX Runtime Version   : {ort.__version__}")

    safetensors_path = MODEL_DIR / "model.safetensors"
    pt_size_mb = safetensors_path.stat().st_size / (1024 * 1024) if safetensors_path.exists() else 0
    onnx_size_mb = ONNX_PATH.stat().st_size / (1024 * 1024) if ONNX_PATH.exists() else 0
    print(f"  PyTorch Model Size     : {pt_size_mb:.2f} MB")
    print(f"  ONNX Model Size        : {onnx_size_mb:.2f} MB")

    # 2. Load Models
    print("\n[*] Loading tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    pt_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    pt_model.eval()

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    # 3. Prepare Benchmark Payloads
    # Short request (single window)
    short_raw = "username=admin' OR 1=1--"
    short_norm = normalize_request(method="POST", path="/login.php", body=short_raw)

    # Long padded request (multi-window, >250 tokens to force sliding windows)
    benign_padding = "param=test&action=view&category=books&author=john_doe&page=1&limit=50&sort=desc&filter=active&session=xyz987654321&mode=verbose&client=browser&theme=dark " * 6
    long_raw = f"{benign_padding}&injection=' UNION SELECT null, username, password FROM users--"
    long_norm = normalize_request(method="POST", path="/api/search", body=long_raw)

    test_payloads = [short_norm] * 100 + [long_norm] * 100  # 200 total requests

    # 4. Warm-Up Pass
    print("\n[*] Warming up runtimes (50 iterations each)...")
    for _ in range(25):
        sliding_score(short_norm, tokenizer, None, pt_model=pt_model)
        sliding_score(long_norm, tokenizer, None, pt_model=pt_model)
        sliding_score(short_norm, tokenizer, session)
        sliding_score(long_norm, tokenizer, session)

    # 5. Benchmark PyTorch CPU Fallback
    print(f"\n[*] Benchmarking PyTorch CPU Fallback (200 requests)...")
    pt_latencies = []
    t_start_pt = time.perf_counter()
    for p in test_payloads:
        t0 = time.perf_counter()
        sliding_score(p, tokenizer, None, pt_model=pt_model)
        pt_latencies.append((time.perf_counter() - t0) * 1000)
    total_time_pt = time.perf_counter() - t_start_pt

    # 6. Benchmark ONNX Runtime CPU
    print(f"[*] Benchmarking ONNX Runtime CPU (200 requests)...")
    onnx_latencies = []
    t_start_onnx = time.perf_counter()
    for p in test_payloads:
        t0 = time.perf_counter()
        sliding_score(p, tokenizer, session)
        onnx_latencies.append((time.perf_counter() - t0) * 1000)
    total_time_onnx = time.perf_counter() - t_start_onnx

    # 7. Compute Metrics
    pt_p50 = np.percentile(pt_latencies, 50)
    pt_p95 = np.percentile(pt_latencies, 95)
    pt_p99 = np.percentile(pt_latencies, 99)
    pt_mean = np.mean(pt_latencies)
    pt_fps = len(test_payloads) / total_time_pt

    onnx_p50 = np.percentile(onnx_latencies, 50)
    onnx_p95 = np.percentile(onnx_latencies, 95)
    onnx_p99 = np.percentile(onnx_latencies, 99)
    onnx_mean = np.mean(onnx_latencies)
    onnx_fps = len(test_payloads) / total_time_onnx

    speedup_mean = pt_mean / onnx_mean
    speedup_p50 = pt_p50 / onnx_p50
    speedup_p95 = pt_p95 / onnx_p95
    speedup_p99 = pt_p99 / onnx_p99

    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS SUMMARY (200 Requests: 100 Short + 100 Padded)")
    print("=" * 70)
    print(f"  Metric                     PyTorch CPU       ONNX Runtime      Speedup")
    print(f"  -------------------------  ----------------  ----------------  -------")
    print(f"  Mean Latency               {pt_mean:6.2f} ms         {onnx_mean:6.2f} ms         {speedup_mean:4.2f}x")
    print(f"  p50 Latency (Median)       {pt_p50:6.2f} ms         {onnx_p50:6.2f} ms         {speedup_p50:4.2f}x")
    print(f"  p95 Latency                {pt_p95:6.2f} ms         {onnx_p95:6.2f} ms         {speedup_p95:4.2f}x")
    print(f"  p99 Latency                {pt_p99:6.2f} ms         {onnx_p99:6.2f} ms         {speedup_p99:4.2f}x")
    print(f"  Throughput                 {pt_fps:6.1f} req/s      {onnx_fps:6.1f} req/s      {onnx_fps/pt_fps:4.2f}x")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_benchmark()
