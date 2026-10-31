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
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
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


class TestDeltatokDispatch(unittest.TestCase):
    """Reject routes that silently retain HF after explicitly choosing DeltaTok."""

    def setUp(self):
        from sglang.srt.utils.hf_transformers import tokenizer

        self.loader = tokenizer
        self.state = patch.object(tokenizer, "_patched_backends", set())
        self.state.start()
        self.addCleanup(self.state.stop)

    def test_environment_cannot_select_a_foreign_engine(self):
        with (
            patch.dict(os.environ, {"DELTATOK_BACKEND": "hf"}),
            patch.object(self.loader.importlib, "import_module") as load,
        ):
            self.loader._ensure_backend_patched("deltatok")
            load.return_value.patch_transformers.assert_called_once_with(backend="off")

    def test_backend_switch_requires_fresh_process(self):
        self.loader._patched_backends.add("deltatok")
        for backend in ("huggingface", "fastokens"):
            with (
                self.subTest(backend=backend),
                self.assertRaisesRegex(RuntimeError, "fresh process"),
            ):
                self.loader._ensure_backend_patched(backend)

    def test_external_patches_cannot_override_selected_backend(self):
        for mode in ("auto", "hf", "fastokens"):
            with (
                self.subTest(mode=mode),
                patch.dict(
                    sys.modules,
                    {"deltatok": SimpleNamespace(_patched=True, _backend_mode=mode)},
                ),
                self.assertRaisesRegex(RuntimeError, "fresh process"),
            ):
                self.loader._ensure_backend_patched("deltatok")
        with (
            patch.dict(
                sys.modules,
                {"deltatok": SimpleNamespace(_patched=True, _backend_mode="off")},
            ),
            self.assertRaisesRegex(RuntimeError, "fresh process"),
        ):
            self.loader._ensure_backend_patched("huggingface")

    def test_revision_is_preserved_for_loading_and_post_load_fixes(self):
        for kwargs, expected in (
            ({"tokenizer_revision": "pinned"}, "pinned"),
            ({"revision": "explicit"}, "explicit"),
            ({"tokenizer_revision": "pinned", "revision": "explicit"}, "explicit"),
        ):
            tokenizer = SimpleNamespace()
            with (
                self.subTest(kwargs=kwargs),
                patch.object(self.loader, "_ensure_backend_patched"),
                patch.object(self.loader, "check_gguf_file", return_value=False),
                patch.object(
                    self.loader, "_resolve_tokenizer_name", return_value="model"
                ),
                patch.object(
                    self.loader, "is_bare_tekken_checkpoint", return_value=False
                ),
                patch.object(
                    self.loader,
                    "_auto_tokenizer_from_pretrained",
                    return_value=tokenizer,
                ) as load,
                patch.object(
                    self.loader, "_apply_post_load_fixes", return_value=tokenizer
                ) as fix,
            ):
                self.loader.get_tokenizer("model", **kwargs)
                self.assertEqual(load.call_args.kwargs["revision"], expected)
                self.assertNotIn("tokenizer_revision", load.call_args.kwargs)
                fix.assert_called_once_with(tokenizer, "model", expected)

    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown tokenizer backend"):
            self.loader._ensure_backend_patched("unknown")

    def test_missing_dependency_is_actionable(self):
        with patch.object(
            self.loader.importlib, "import_module", side_effect=ImportError
        ):
            with self.assertRaisesRegex(ImportError, "deltatok package is required"):
                self.loader._ensure_backend_patched("deltatok")

    def test_standalone_json_cannot_bypass_backend(self):
        with patch.object(self.loader, "_ensure_backend_patched"):
            with self.assertRaisesRegex(ValueError, "standalone tiktoken JSON"):
                self.loader.get_tokenizer(
                    "tokenizer.json", tokenizer_backend="deltatok"
                )

    def test_native_gguf_cannot_bypass_backend(self):
        with (
            patch.object(self.loader, "_ensure_backend_patched"),
            patch.object(self.loader, "check_gguf_file", return_value=True),
            patch.object(self.loader, "gguf_sidecar_dir", return_value=None),
            patch.object(self.loader, "has_native_gguf_support", return_value=True),
        ):
            with self.assertRaisesRegex(ValueError, "native GGUF"):
                self.loader.get_tokenizer("model.gguf", tokenizer_backend="deltatok")

    def test_bare_tekken_cannot_bypass_backend(self):
        with (
            patch.object(self.loader, "_ensure_backend_patched"),
            patch.object(self.loader, "check_gguf_file", return_value=False),
            patch.object(self.loader, "_resolve_tokenizer_name", return_value="model"),
            patch.object(self.loader, "is_bare_tekken_checkpoint", return_value=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "deltatok failed") as error:
                self.loader.get_tokenizer("model", tokenizer_backend="deltatok")
            self.assertIn("bare tekken", str(error.exception.__cause__))

    @unittest.skipUnless(HAS_DELTATOK, "deltatok package not installed")
    def test_custom_tokenizer_cannot_retain_foreign_engine(self):
        with self.assertRaisesRegex(ValueError, "native deltatok engine"):
            self.loader._validate_tokenizer_backend(SimpleNamespace(), "deltatok")


@unittest.skipUnless(HAS_DELTATOK, "deltatok package not installed")
class TestDeltatokBackend(unittest.TestCase):
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
                env={**os.environ, "DELTATOK_BACKEND": "hf"},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            results.append(json.loads(result.stdout))
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
