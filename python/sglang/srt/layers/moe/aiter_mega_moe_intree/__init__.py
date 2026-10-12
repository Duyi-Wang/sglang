"""In-tree MegaMoE stage1 skew fix for aiter MegaMoEV2 (DeepSeek-V4 prefill).

ROCm/aiter#6259 (``882ea79dd7``) fixed a race on MegaMoEV2 stage1's compact
ready-order path by making every compact work item run ``tid0 wait(tile_ready)
+ barrier + fence_system_acquire``.  The ready-order scheduler block already
does a barrier + system acquire right after popping the ready queue, so that
path now pays a second round per work item.  Under skewed routing (a hot expert
holds >= 25% of the tiles, which selects ready-order) DSv4-Pro EP8 a8w4 prefill
is up to ~6% slower (1024 tok/rank: +5.4% at 1x skew, +6.1% at 2x skew).

The fix (Duyi-Wang/aiter ``a5d49321ac``, +10/-8 in ``mega_moe_stage1.py``)
moves the ready-order ``_wait_tile_payload`` into the scheduler block, before
its existing barrier + system acquire, and keeps the per-work-item wait for the
sequential path only.  The ordering guarantee of #6259 is unchanged and the
output is bit-identical.

``install_mega_moe_stage1_fix()`` runs before the first ``MegaMoEV2`` is built
(``MegaMoEV2._build_fused_stage1`` imports stage1's entry points at construction
time).  It replaces the five public entry points of aiter's
``aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage1`` module with the ones from
the in-tree copy ``.mega_moe_stage1``; nothing else in aiter is touched, and
MegaMoEV2 instances that already exist keep what they captured.

Upstream-first: the copy is only installed while aiter ships the exact unfixed
file it was made from (sha256 of ``mega_moe_stage1.py``, identical in ROCm/aiter
``b0c4209d4d`` .. ``02575f1c3a``).  If aiter already has the fix (or never had the
regression) nothing is patched; if aiter changed the file some other way, nothing
is patched and a warning asks to refresh or drop the copy.  Upgrading the aiter
pin therefore switches over automatically and never breaks startup.

FlyDSL caches: the copy's kernels are named ``megamoe_stage1_..._sglrof`` and
their cache key includes the copy's own source plus a fingerprint of the aiter
MegaMoE sources it compiles against, so they never share a JIT or AOT cache
entry with aiter's kernels.  aiter's AOT cache only holds aiter's own (unfixed)
kernels, so the fixed stage1 bundle is JIT-compiled once per FlyDSL cache dir.

Switch: ``SGLANG_AITER_INTREE_MEGA_STAGE1_FIX`` (default 1).  Set it to 0 to run
the pinned aiter stage1 as is.

TODO(aiter#6259): remove this package once the ready-order fix for the
ROCm/aiter#6259 regression is in the aiter pin (no upstream PR yet).
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_AITER_STAGE1_MODULE = "aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage1"

# TODO(aiter#6259): remove once the ROCm/aiter#6259 regression fix is in the aiter pin
# sha256 of aiter's mega_moe_stage1.py that the in-tree copy was made from: the
# post-#6259 file, unchanged from ROCm/aiter b0c4209d4d through 02575f1c3a.
_KNOWN_UNFIXED_STAGE1_SHA256 = {
    "7392723fd11bdfd7d7d1e9ce037197117bef3fb3c1c6dd8e272a66bf64393a6d": "02575f1c3a",
}
# Present only in a stage1 that already waits for the tile in the ready-order
# scheduler block (this fix, upstreamed verbatim).
_FIX_MARKER = "_wait_tile_payload(scheduled_first)"
# The #6259 per-work-item wait that the fix narrows to the sequential path.
_REGRESSION_MARKER = "Both the sequential and the ready-order compact paths"

_PATCHED_NAMES = (
    "compile_mega_moe_stage1",
    "compile_mega_moe_stage1_bundle",
    "run_mega_moe_stage1",
    "run_mega_moe_stage1_bundle",
    "preload_mega_moe_stage1_bundle",
)

# None: not decided yet; otherwise the final status string (see below).
_STATUS: Optional[str] = None
# aiter's own stage1 entry points, kept when the in-tree ones are installed.
_ORIGINAL_ENTRY_POINTS: dict = {}


def _enabled() -> bool:
    from sglang.srt.environ import envs

    return envs.SGLANG_AITER_INTREE_MEGA_STAGE1_FIX.get()


def _classify(source: bytes) -> str:
    """Return 'fixed', 'unfixed-known', 'unfixed-unknown' or 'unaffected'."""
    text = source.decode("utf-8", errors="replace")
    if _FIX_MARKER in text:
        return "fixed"
    if hashlib.sha256(source).hexdigest() in _KNOWN_UNFIXED_STAGE1_SHA256:
        return "unfixed-known"
    if _REGRESSION_MARKER in text:
        return "unfixed-unknown"
    return "unaffected"


def install_mega_moe_stage1_fix() -> str:
    """Install the in-tree stage1 fix over aiter's module when it applies.

    Idempotent.  Returns the status: ``installed``, ``disabled``,
    ``aiter-fixed``, ``aiter-unaffected``, ``aiter-changed`` or ``unavailable``.
    """
    global _STATUS
    if _STATUS is not None:
        return _STATUS
    if not _enabled():
        _STATUS = "disabled"
        logger.info(
            "SGLANG_AITER_INTREE_MEGA_STAGE1_FIX=0: MegaMoE stage1 uses aiter as is."
        )
        return _STATUS

    try:
        aiter_stage1 = importlib.import_module(_AITER_STAGE1_MODULE)
        with open(aiter_stage1.__file__, "rb") as f:
            source = f.read()
    except (ImportError, OSError, TypeError) as exc:
        _STATUS = "unavailable"
        logger.warning(
            "MegaMoE stage1 in-tree fix not applied: cannot read %s (%s).",
            _AITER_STAGE1_MODULE,
            exc,
        )
        return _STATUS

    kind = _classify(source)
    if kind == "fixed":
        _STATUS = "aiter-fixed"
        logger.info(
            "aiter MegaMoE stage1 already folds the ready-order tile wait; "
            "the in-tree fix is not needed (TODO(aiter#6259): drop it)."
        )
    elif kind == "unaffected":
        _STATUS = "aiter-unaffected"
        logger.info(
            "aiter MegaMoE stage1 predates the ROCm/aiter#6259 ready-order wait; "
            "the in-tree fix is not needed."
        )
    elif kind == "unfixed-unknown":
        _STATUS = "aiter-changed"
        logger.warning(
            "aiter %s still has the ROCm/aiter#6259 ready-order double wait, but "
            "the file differs from the version the in-tree fix was made from "
            "(sha256 %s); not patching, MegaMoE prefill may be ~5%% slower under "
            "skewed routing. Refresh or drop "
            "sglang/srt/layers/moe/aiter_mega_moe_intree.",
            os.path.basename(aiter_stage1.__file__),
            hashlib.sha256(source).hexdigest()[:16],
        )
    else:
        try:
            from sglang.srt.layers.moe.aiter_mega_moe_intree import (
                mega_moe_stage1 as intree_stage1,
            )
        except Exception as exc:  # never break startup over a perf fix
            _STATUS = "unavailable"
            logger.warning(
                "MegaMoE stage1 in-tree fix not applied: importing it failed (%r).",
                exc,
            )
            return _STATUS
        for name in _PATCHED_NAMES:
            _ORIGINAL_ENTRY_POINTS[name] = getattr(aiter_stage1, name)
            setattr(aiter_stage1, name, getattr(intree_stage1, name))
        _STATUS = "installed"
        logger.info(
            "MegaMoE stage1: using the in-tree ready-order fix for the "
            "ROCm/aiter#6259 skew regression (aiter %s); "
            "SGLANG_AITER_INTREE_MEGA_STAGE1_FIX=0 to disable.",
            _KNOWN_UNFIXED_STAGE1_SHA256[hashlib.sha256(source).hexdigest()],
        )
    return _STATUS
