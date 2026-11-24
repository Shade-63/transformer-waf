"""
tests/test_normalizer_unit.py — Rigorous unit tests for 5-stage input canonicalization.
"""

import pytest
from proxy.normalizer import normalize, normalize_request


class TestNormalizerEdgeCases:

    def test_double_url_encoding_resolution(self):
        """Resolves double percent-encoding: %2527 -> %27 -> '."""
        raw = "id=%2527%20OR%20%2531%3D%2531"
        cleaned = normalize(raw)
        assert "'" in cleaned
        assert "%25" not in cleaned

    def test_html_entity_decoding_variants(self):
        """Resolves decimal, hex, and named HTML entities."""
        raw = "&quot; OR &#x27;1&#x27;=&#39;1&#39; &lt;script&gt;"
        cleaned = normalize(raw)
        assert '"' in cleaned
        assert "'" in cleaned
        assert "<script>" in cleaned
        assert "&#" not in cleaned

    def test_unicode_nfkc_compatibility_mapping(self):
        """NFKC normalization collapses fullwidth and modifier characters."""
        # Fullwidth UNION SELECT (\uff35\uff2e\uff29\uff2f\uff2e \uff33\uff25\uff2c\uff25\uff23\uff34)
        raw_fullwidth = "\uff35\uff2e\uff29\uff2f\uff2e \uff33\uff25\uff2c\uff25\uff23\uff34"
        cleaned = normalize(raw_fullwidth)
        assert "UNION SELECT" in cleaned

        # Fullwidth apostrophe (\uff07) maps to ASCII apostrophe
        raw_apostrophe = "\uff07 OR 1=1--"
        assert "'" in normalize(raw_apostrophe)

    def test_zero_width_character_stripping(self):
        """Strips zero-width spaces, joiners, non-joiners, and byte order marks."""
        # Injected U+200B (ZWS) and U+200D (ZWJ) and U+FEFF (BOM)
        raw = "U\u200bN\u200cI\u200dO\ufeffN SELECT"
        cleaned = normalize(raw)
        assert cleaned == "UNION SELECT"
        for zws in ["\u200b", "\u200c", "\u200d", "\ufeff"]:
            assert zws not in cleaned

    def test_sql_hex_literal_decoding(self):
        """Decodes \x20 to space, \x27 to apostrophe, \x41 to A."""
        raw = r"SELECT \x75\x73\x65\x72 FROM \x75\x73\x65\x72\x73 WHERE id=\x27\x31\x27"
        cleaned = normalize(raw)
        assert "SELECT user FROM users WHERE id='1'" in cleaned

    def test_normalize_request_structure(self):
        """Verifies structured HTTP field tags."""
        result = normalize_request(
            method="POST",
            path="/api/test",
            query="a=1",
            body="payload=test",
            user_agent="TestAgent",
            cookie="sess=1"
        )
        assert result.startswith("[METHOD] POST")
        assert "[PATH] /api/test" in result
        assert "[QUERY] a=1" in result
        assert "[BODY] payload=test" in result
        assert "[UA] TestAgent" in result
        assert "[COOKIE] sess=1" in result

    def test_normalize_request_truncation(self):
        """Verifies that requests exceeding 1000 characters are safely capped."""
        long_body = "A" * 2000
        result = normalize_request(
            method="POST",
            path="/api/test",
            body=long_body
        )
        assert len(result) == 1000
