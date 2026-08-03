"""End-to-end verification that --tokenizer-backend=deltatok swaps the backend
of the loaded tokenizer with deltatok's _TokenizerShim, and that doing so does
not change a single token id versus the default HuggingFace backend.

The id-equality test is the one that matters: deltatok is a tokenizer-of-record
replacement, so "faster" is only acceptable if it is also byte-identical.
"""

import json
import os
import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
)

TOKENIZER_MODEL = os.environ.get(
    "DELTATOK_TEST_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
)

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
        # patch_transformers is process-global. Earlier tests may have patched
        # this interpreter, so both backends need fresh child processes.
        code = """
import json, sys
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer
model, backend = sys.argv[1:]
tok = get_tokenizer(model, tokenizer_backend=backend)
shim_module = type(getattr(tok, "_tokenizer", None)).__module__
assert shim_module.startswith("deltatok") == (backend == "deltatok"), shim_module
if backend == "huggingface":
    assert "deltatok" not in sys.modules
prompts = json.load(sys.stdin)
for special in (False, True):
    entries = [t.content for t in tok.added_tokens_decoder.values()
               if t.special == special][:8]
    prompts.extend("hello " + t + " world" for t in entries)
ids = [tok.encode(p) for p in prompts]
json.dump(dict(ids=ids, decoded=[tok.decode(i) for i in ids],
               batch_decoded=tok.batch_decode(ids),
               skipped=tok.batch_decode(ids, skip_special_tokens=True),
               vocab_size=tok.vocab_size), sys.stdout)
"""
        results = []
        for backend in ("huggingface", "deltatok"):
            result = subprocess.run(
                [sys.executable, "-c", code, TOKENIZER_MODEL, backend],
                input=json.dumps(PROMPTS),
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            results.append(json.loads(result.stdout))
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
