"""
tests/test_sliding_window_unit.py — Verification of sliding-window scanning mechanics.
"""

import pytest
from transformers import AutoTokenizer
from proxy.scanner import sliding_score, WINDOW_SIZE, STRIDE
from proxy.model import session, pt_model, tokenizer


class TestSlidingWindowMechanics:

    def test_window_constants(self):
        """Ensure window size and stride match production design specifications."""
        assert WINDOW_SIZE == 128
        assert STRIDE == 64
        # 50% overlap guarantees token coverage
        assert WINDOW_SIZE - STRIDE == 64

    def test_short_request_single_window_bypass(self):
        """Requests <= 126 tokens must use a single scoring pass (windows_checked == 1)."""
        short_text = "[METHOD] GET [PATH] /hello [QUERY]  [BODY] name=alice [UA] Firefox [COOKIE] "
        score, windows, blocked = sliding_score(short_text, tokenizer, session, pt_model=pt_model)
        assert windows == 1
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_payload_past_token_128_detected(self):
        """
        Verify that a malicious injection pushed beyond token 128 (window 1)
        is covered by window 2 (tokens 64-191) and blocked.
        """
        # Create benign padding tokens (~150 tokens)
        padding = "safe_parameter_key=valid_safe_value_abc_123 & " * 15
        tokens = tokenizer.encode(padding, add_special_tokens=False)
        assert len(tokens) > 130, f"Padding should be >130 tokens, got {len(tokens)}"

        # Attack payload appended after padding
        malicious_payload = " ' UNION SELECT username, password FROM users--"
        full_text = padding + malicious_payload

        # Sliding score should check >1 window and successfully block the attack
        score, windows, blocked = sliding_score(full_text, tokenizer, session, pt_model=pt_model)
        assert windows >= 2, f"Expected at least 2 windows, got {windows}"
        assert blocked is True
        assert score >= 0.5

    def test_max_score_aggregation_logic(self):
        """
        Confirms max aggregation preserves high threat signal even when
        multiple preceding windows are completely benign.
        """
        clean_prefix = "info=system_health_status_ok&mode=continuous&version=1.0.0 " * 12
        threat_suffix = " & payload='; DROP TABLE users; --"
        combined = clean_prefix + threat_suffix

        score, windows, blocked = sliding_score(combined, tokenizer, session, pt_model=pt_model)
        assert windows >= 2
        # Max score must dominate over low-scoring windows
        assert score >= 0.5
        assert blocked is True
