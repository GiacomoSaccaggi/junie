"""Tests for secret_shield — credential masking on the prompt input path.

Synthetic data only. No real credentials, no network. Run:
    python3 -m pytest tests/test_secret_shield.py -v --import-mode=importlib --rootdir=tests
"""
from __future__ import annotations

import logging
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_plugin_dir = Path(__file__).resolve().parent.parent / "junie_hermes"
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from secret_shield import (
    REDACTED, RedactResult, RedactionError, SecretRedactor,
    redact_secrets, redact_messages, redact_prompt,
    shield_prompt, shield_messages,
)

R = SecretRedactor()


# ── Provider prefix patterns ─────────────────────────────────────────────────

class TestPrefixPatterns(unittest.TestCase):

    CASES = {
        "google_api":    "AIza" + "a" * 32,
        "google_oauth":  "GOCSPX-" + "a" * 24,
        "google_access": "ya29." + "a" * 24,
        "openai":        "sk-proj-" + "a" * 30,
        "anthropic":     "sk-ant-api03-" + "a" * 30,
        "groq":          "gsk_" + "a" * 24,
        "github":        "ghp_" + "a" * 30,
        "github_fine":   "github_pat_" + "a" * 30,
        "gitlab":        "glpat-" + "a" * 24,
        "aws":           "AKIA" + "A" * 16,
        "junie":         "perm-" + "a" * 24,
        "stripe":        "sk_live_" + "a" * 24,
        "slack":         "xoxb-" + "a" * 24,
        "sendgrid":      "SG." + "a" * 24 + "." + "b" * 24,
        "huggingface":   "hf_" + "a" * 24,
        "npm":           "npm_" + "a" * 24,
        "pypi":          "pypi-" + "a" * 24,
        "vault":         "hvs." + "a" * 24,
    }

    def test_each_prefix(self):
        for name, key in self.CASES.items():
            with self.subTest(name=name):
                r = R.redact(f"my key is {key}")
                self.assertNotIn(key, r.text, f"key for {name} was not redacted")
                self.assertIn(REDACTED, r.text)
                self.assertEqual(r.redacted_count, 1)

    def test_prefix_in_code_context(self):
        r = R.redact('export OPENAI_API_KEY="sk-proj-abc123def456ghi789jkl012mno"')
        self.assertNotIn("abc123def456", r.text)
        self.assertGreaterEqual(r.redacted_count, 1)


# ── Key=value assignments ────────────────────────────────────────────────────

class TestKeyValueAssignment(unittest.TestCase):

    def test_password_equals(self):
        r = R.redact("password=SuperSecret123 done")
        self.assertNotIn("SuperSecret123", r.text)
        self.assertEqual(r.redacted_count, 1)

    def test_token_colon(self):
        r = R.redact("token: my_secret_value")
        self.assertNotIn("my_secret_value", r.text)

    def test_api_key_equals(self):
        r = R.redact("api_key=abcdef123456")
        self.assertNotIn("abcdef123456", r.text)

    def test_quoted_value(self):
        r = R.redact('password="hello world"')
        self.assertNotIn("hello world", r.text)

    def test_single_quoted(self):
        r = R.redact("secret='my_secret'")
        self.assertNotIn("my_secret", r.text)

    def test_client_secret(self):
        r = R.redact("client_secret=abcdef123456")
        self.assertNotIn("abcdef123456", r.text)

    def test_connection_string_key(self):
        r = R.redact("connection_string=Server=host;Password=abc")
        self.assertNotIn("Password=abc", r.text)

    def test_database_url_key(self):
        r = R.redact("database_url=postgres://user:pass@host/db")
        self.assertNotIn("pass@host", r.text)


# ── Natural language ─────────────────────────────────────────────────────────

class TestNaturalLanguage(unittest.TestCase):

    def test_password_is(self):
        r = R.redact("the password is SuperSecret123 and it works")
        self.assertNotIn("SuperSecret123", r.text)
        self.assertEqual(r.redacted_count, 1)

    def test_token_is(self):
        r = R.redact("the token is abc123def456")
        self.assertNotIn("abc123def456", r.text)

    def test_password_est(self):
        r = R.redact("le password est monSecret")
        self.assertNotIn("monSecret", r.text)


# ── HTTP headers and auth schemes ────────────────────────────────────────────

class TestHeaders(unittest.TestCase):

    def test_authorization_header(self):
        r = R.redact("Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig")
        self.assertIn(REDACTED, r.text)

    def test_cookie_header(self):
        r = R.redact("Cookie: session=abc123; path=/")
        self.assertNotIn("abc123", r.text)

    def test_bearer_inline(self):
        r = R.redact("use Bearer my_token_value_here_1234 for auth")
        self.assertNotIn("my_token_value_here_1234", r.text)

    def test_x_api_key_header(self):
        r = R.redact("X-Api-Key: abcdefghij123456")
        self.assertNotIn("abcdefghij123456", r.text)


# ── Connection string URIs ───────────────────────────────────────────────────

class TestConnectionStrings(unittest.TestCase):

    def test_postgres(self):
        r = R.redact("postgres://admin:s3cret@db.example.com:5432/mydb")
        self.assertNotIn("s3cret", r.text)
        self.assertNotIn("admin", r.text)
        self.assertIn("@db.example.com", r.text)

    def test_mysql(self):
        r = R.redact("mysql://root:password123@localhost:3306/db")
        self.assertNotIn("password123", r.text)

    def test_mongodb_srv(self):
        r = R.redact("mongodb+srv://user:pass@cluster.example.com/db")
        self.assertNotIn("pass", r.text)

    def test_redis(self):
        r = R.redact("redis://default:mysecret@redis.host:6379")
        self.assertNotIn("mysecret", r.text)


# ── JWT-shaped tokens ────────────────────────────────────────────────────────

class TestJWT(unittest.TestCase):

    def _make_jwt(self, header: dict, payload: str = "eyJzdWIiOiJ4In0", sig: str = "signature") -> str:
        import base64, json
        h = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        return f"{h}.{payload}.{sig}"

    def test_valid_jwt(self):
        jwt = self._make_jwt({"alg": "HS256", "typ": "JWT"})
        r = R.redact(f"token: {jwt}")
        self.assertNotIn(jwt, r.text)
        self.assertIn(REDACTED, r.text)

    def test_not_a_jwt(self):
        r = R.redact("some.dotted.path is fine")
        self.assertEqual(r.redacted_count, 0)

    def test_jwt_in_env_var(self):
        jwt = self._make_jwt({"alg": "RS256"})
        r = R.redact(f'export TOKEN="{jwt}"')
        self.assertNotIn(jwt, r.text)


# ── JSON sensitive fields ────────────────────────────────────────────────────

class TestJsonFields(unittest.TestCase):

    def test_password_field(self):
        r = R.redact('{"password":"mysecret","port":5432}')
        self.assertNotIn("mysecret", r.text)
        self.assertIn("port", r.text)

    def test_api_key_field(self):
        r = R.redact('{"api_key":"sk-abc123"}')
        self.assertNotIn("sk-abc123", r.text)

    def test_nested_token(self):
        r = R.redact('{"config":{"token":"secret123"}}')
        self.assertNotIn("secret123", r.text)

    def test_non_sensitive_preserved(self):
        text = '{"name":"alice","port":5432}'
        r = R.redact(text)
        self.assertEqual(r.text, text)
        self.assertEqual(r.redacted_count, 0)


# ── Clean passthrough ────────────────────────────────────────────────────────

class TestCleanPassthrough(unittest.TestCase):

    def test_plain_text(self):
        text = "Hello, please review this code for bugs."
        self.assertEqual(R.redact(text).text, text)

    def test_code_snippet(self):
        text = 'fn main() {\n    let x = 42;\n    println!("hello {}", x);\n}'
        self.assertEqual(R.redact(text).text, text)

    def test_url_without_creds(self):
        text = "visit https://example.com/api/v1/users"
        self.assertEqual(R.redact(text).text, text)

    def test_port_number(self):
        r = R.redact("port=5432 host=localhost")
        self.assertEqual(r.redacted_count, 0)

    def test_empty(self):
        self.assertEqual(R.redact("").text, "")


# ── Multiple secrets ─────────────────────────────────────────────────────────

class TestMultipleSecrets(unittest.TestCase):

    def test_two_secrets(self):
        r = R.redact("password=first token=second port=123")
        self.assertNotIn("first", r.text)
        self.assertNotIn("second", r.text)
        self.assertGreaterEqual(r.redacted_count, 2)

    def test_prefix_and_assignment(self):
        r = R.redact("key=AIza" + "a" * 32 + " and password=hunter2 done")
        self.assertNotIn("AIza", r.text)
        self.assertNotIn("hunter2", r.text)
        self.assertGreaterEqual(r.redacted_count, 2)


# ── Idempotency ──────────────────────────────────────────────────────────────

class TestIdempotency(unittest.TestCase):
    """Redacting already-redacted output must produce no further changes."""

    EXAMPLES = [
        "password=SuperSecret123 done",
        "the password is carbonara69 ok",
        'Authorization: Bearer my_token_value_here_1234',
        "postgres://admin:s3cret@db.example.com/mydb",
        '{"password":"mysecret","port":5432}',
        "my key is ghp_" + "a" * 30,
    ]

    def test_double_redact(self):
        for text in self.EXAMPLES:
            with self.subTest(text=text[:40]):
                first = R.redact(text)
                second = R.redact(first.text)
                self.assertEqual(first.text, second.text, "output changed on second pass")
                self.assertEqual(second.redacted_count, 0, "second pass found new secrets")


# ── Overlapping detectors ────────────────────────────────────────────────────

class TestOverlap(unittest.TestCase):
    """A value matched by multiple detectors must still be fully masked."""

    def test_prefix_inside_assignment(self):
        key = "sk-proj-" + "a" * 30
        r = R.redact(f"api_key={key}")
        self.assertNotIn(key, r.text)
        self.assertNotIn("sk-proj", r.text)

    def test_bearer_with_prefix_token(self):
        r = R.redact("Authorization: Bearer ghp_" + "a" * 30)
        self.assertNotIn("ghp_", r.text)

    def test_uri_with_prefix_password(self):
        key = "gsk_" + "a" * 24
        r = R.redact(f"postgres://user:{key}@host/db")
        self.assertNotIn(key, r.text)


# ── Error behavior (fail-closed) ─────────────────────────────────────────────

class TestErrorBehavior(unittest.TestCase):

    def test_shield_prompt_raises_on_internal_error(self):
        with patch.dict(os.environ, {"HERMES_JUNIE_ACP_SECRET_SHIELD": "1"}):
            with patch.object(SecretRedactor, "redact", side_effect=RuntimeError("boom")):
                with self.assertRaises(RedactionError) as ctx:
                    shield_prompt("password=secret")
                self.assertNotIn("secret", str(ctx.exception))

    def test_shield_messages_raises_on_internal_error(self):
        with patch.dict(os.environ, {"HERMES_JUNIE_ACP_SECRET_SHIELD": "1"}):
            with patch.object(SecretRedactor, "redact_messages", side_effect=RuntimeError("boom")):
                with self.assertRaises(RedactionError):
                    shield_messages([{"role": "user", "content": "password=secret"}])

    def test_shield_prompt_passes_redaction_error_through(self):
        with patch.dict(os.environ, {"HERMES_JUNIE_ACP_SECRET_SHIELD": "1"}):
            with patch.object(SecretRedactor, "redact", side_effect=RedactionError("test")):
                with self.assertRaises(RedactionError):
                    shield_prompt("password=secret")

    def test_disabled_does_not_raise(self):
        with patch.dict(os.environ, {"HERMES_JUNIE_ACP_SECRET_SHIELD": "0"}):
            with patch.object(SecretRedactor, "redact", side_effect=RuntimeError("boom")):
                text, count = shield_prompt("password=secret")
                self.assertIn("secret", text)
                self.assertEqual(count, 0)


# ── Toggle behavior ──────────────────────────────────────────────────────────

class TestToggle(unittest.TestCase):

    def test_disabled_by_default_no_env(self):
        with patch.dict(os.environ, {}, clear=True):
            text, count = shield_prompt("password=hunter2")
            self.assertEqual(count, 0)
            self.assertIn("hunter2", text)

    def test_enabled_via_env_1(self):
        with patch.dict(os.environ, {"HERMES_JUNIE_ACP_SECRET_SHIELD": "1"}):
            text, count = shield_prompt("password=hunter2 done")
            self.assertGreaterEqual(count, 1)
            self.assertNotIn("hunter2", text)

    def test_enabled_via_env_true(self):
        with patch.dict(os.environ, {"HERMES_JUNIE_ACP_SECRET_SHIELD": "true"}):
            text, count = shield_prompt("password=hunter2 done")
            self.assertGreaterEqual(count, 1)

    def test_disabled_via_env_0(self):
        with patch.dict(os.environ, {"HERMES_JUNIE_ACP_SECRET_SHIELD": "0"}):
            text, count = shield_prompt("password=hunter2")
            self.assertEqual(count, 0)
            self.assertIn("hunter2", text)

    def test_disabled_via_env_false(self):
        with patch.dict(os.environ, {"HERMES_JUNIE_ACP_SECRET_SHIELD": "false"}):
            text, count = shield_prompt("password=hunter2")
            self.assertEqual(count, 0)

    def test_env_var_skips_config_yaml(self):
        """When env var is set, _resolve_enabled returns without importing hermes_cli."""
        with patch.dict(os.environ, {"HERMES_JUNIE_ACP_SECRET_SHIELD": "0"}):
            from secret_shield import _resolve_enabled
            # Patch at module level — if hermes_cli.config were imported, it would fail.
            with patch.dict(sys.modules, {"hermes_cli": None, "hermes_cli.config": None}):
                result = _resolve_enabled()
                self.assertFalse(result)

    def test_messages_disabled_returns_same_object(self):
        messages = [{"role": "user", "content": "password=hunter2"}]
        with patch.dict(os.environ, {"HERMES_JUNIE_ACP_SECRET_SHIELD": "0"}):
            result, count = shield_messages(messages)
        self.assertEqual(count, 0)
        self.assertIs(result, messages)

    def test_messages_enabled(self):
        messages = [{"role": "user", "content": "password=hunter2 done"}]
        with patch.dict(os.environ, {"HERMES_JUNIE_ACP_SECRET_SHIELD": "1"}):
            result, count = shield_messages(messages)
        self.assertGreaterEqual(count, 1)
        self.assertNotIn("hunter2", repr(result))


# ── Logging safety ───────────────────────────────────────────────────────────

class TestLogging(unittest.TestCase):

    def test_no_secrets_in_log_output(self):
        with self.assertLogs("secret_shield", level="WARNING") as logs:
            redact_prompt("password=never_log_this_value done")
        combined = "\n".join(logs.output)
        self.assertNotIn("never_log_this_value", combined)
        self.assertIn("Redacted", combined)

    def test_no_secrets_in_messages_log(self):
        with self.assertLogs("secret_shield", level="WARNING") as logs:
            R.redact_messages([{"role": "user", "content": "token=secret_log_test done"}])
        self.assertNotIn("secret_log_test", "\n".join(logs.output))


# ── Malformed and edge-case input ────────────────────────────────────────────

class TestMalformed(unittest.TestCase):

    def test_very_long_input(self):
        text = "password=short " + "x" * 100_000
        r = R.redact(text)
        self.assertNotIn("short", r.text)

    def test_empty_password_value(self):
        r = R.redact("password=  ")
        self.assertEqual(r.redacted_count, 0)

    def test_special_chars_in_value(self):
        r = R.redact("password=p@ss!w0rd#123")
        self.assertNotIn("p@ss!w0rd#123", r.text)

    def test_newlines_in_input(self):
        r = R.redact("password=first\ntoken=second\nport=123")
        self.assertNotIn("first", r.text)
        self.assertNotIn("second", r.text)

    def test_unicode_content(self):
        r = R.redact("password=пароль_секрет done")
        self.assertNotIn("пароль_секрет", r.text)

    def test_only_redacted_marker(self):
        r = R.redact(f"password={REDACTED}")
        self.assertEqual(r.redacted_count, 0)


# ── Message wrapper ──────────────────────────────────────────────────────────

class TestMessages(unittest.TestCase):

    def test_masks_content(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "my key is ghp_" + "a" * 30},
        ]
        result, count = R.redact_messages(messages)
        self.assertEqual(count, 1)
        self.assertIn(REDACTED, result[1]["content"])
        self.assertIn("ghp_", messages[1]["content"])  # original not mutated

    def test_non_string_preserved(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result, count = R.redact_messages(messages)
        self.assertEqual(count, 0)

    def test_clean_passthrough(self):
        messages = [{"role": "user", "content": "fix the bug"}]
        result, count = R.redact_messages(messages)
        self.assertEqual(count, 0)
        self.assertEqual(result[0]["content"], "fix the bug")


# ── Type safety ──────────────────────────────────────────────────────────────

class TestTypes(unittest.TestCase):

    def test_none_raises(self):
        with self.assertRaises(TypeError):
            redact_secrets(None)

    def test_int_raises(self):
        with self.assertRaises(TypeError):
            redact_secrets(123)

    def test_prompt_wrapper_returns(self):
        text, count = redact_prompt("password=short done")
        self.assertIsInstance(text, str)
        self.assertEqual(count, 1)

    def test_empty_messages(self):
        self.assertEqual(redact_messages([]), ([], 0))

    def test_bad_messages_type(self):
        with self.assertRaises(TypeError):
            redact_messages("not a list")

    def test_bad_message_item(self):
        with self.assertRaises(TypeError):
            redact_messages(["not a dict"])


if __name__ == "__main__":
    unittest.main()
