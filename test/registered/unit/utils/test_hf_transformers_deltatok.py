"""End-to-end verification that --tokenizer-backend=deltatok swaps the backend
of the loaded tokenizer with deltatok's _TokenizerShim, and that doing so does
not change a single token id versus the default HuggingFace backend.

The id-equality test is the one that matters: deltatok is a tokenizer-of-record
replacement, so "faster" is only acceptable if it is also byte-identical.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
)

TOKENIZER_MODEL = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


try:
    import deltatok  # noqa: F401

    HAS_DELTATOK = True
except ImportError:
    HAS_DELTATOK = False


PROMPTS = [
    "Hello, world!",
    "Question: Natalia sold clips to 48 of her friends in April, and then she "
    "sold half as many clips in May. How many clips did Natalia sell "
    "altogether in April and May?\nAnswer:",
    "你好，世界！这是一个测试。",
    "def solve(n):\n    return sum(i**2 for i in range(n))\n",
    "café naïve résumé — it's don't we're",
    "   \n\t  ",
]


@unittest.skipUnless(HAS_DELTATOK, "deltatok package not installed")
class TestDeltatokBackend(CustomTestCase):
    def test_shim_is_applied(self):
        # `_TokenizerShim` is deltatok's private compat shim. SGLang's
        # integration relies on `tokenizer._tokenizer` being an instance of
        # this class to confirm deltatok is wired up. If deltatok renames or
        # restructures it, update both this assertion and any code in SGLang
        # that depends on the same private name.
        from deltatok._compat import _TokenizerShim

        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(TOKENIZER_MODEL, tokenizer_backend="deltatok")
        backend = getattr(tokenizer, "_tokenizer", None)
        self.assertIsInstance(
            backend,
            _TokenizerShim,
            f"Expected tokenizer._tokenizer to be _TokenizerShim, "
            f"got {type(backend).__name__}",
        )

    def test_encode_decode_roundtrip(self):
        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(TOKENIZER_MODEL, tokenizer_backend="deltatok")
        text = "Hello, world!"
        ids = tokenizer.encode(text, add_special_tokens=False)
        self.assertGreater(len(ids), 0)
        self.assertEqual(tokenizer.decode(ids, skip_special_tokens=True), text)

    def test_ids_match_huggingface_backend(self):
        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        baseline = get_tokenizer(TOKENIZER_MODEL, tokenizer_backend="huggingface")
        patched = get_tokenizer(TOKENIZER_MODEL, tokenizer_backend="deltatok")

        for prompt in PROMPTS:
            with self.subTest(prompt=prompt[:32]):
                hf_ids = baseline.encode(prompt)
                self.assertEqual(patched.encode(prompt), hf_ids)
                self.assertEqual(patched.decode(hf_ids), baseline.decode(hf_ids))

        # batch_decode is what the detokenizer manager actually calls.
        batch = [baseline.encode(p) for p in PROMPTS]
        self.assertEqual(patched.batch_decode(batch), baseline.batch_decode(batch))
        self.assertEqual(patched.vocab_size, baseline.vocab_size)


if __name__ == "__main__":
    unittest.main()
