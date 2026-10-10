"""aiter MegaMoE on a PD decode node: SWA cadence override, pad-row masking and
the decode capacity check (CPU only; the MegaMoEV2 op is faked)."""

import sys
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.layers.moe.mega_moe_flydsl as mega_flydsl
import sglang.srt.managers.schedule_batch as schedule_batch
from sglang.srt.arg_groups.mega_moe_hook import validate_aiter_mega_moe_decode_budget
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_MOD = "sglang.srt.layers.moe.mega_moe_flydsl"


@pytest.mark.parametrize(
    "cadence, mega, hip, expected",
    [
        ("auto", "0", True, False),
        ("auto", "1", True, True),
        ("auto", "1", False, False),
        ("token", "1", True, False),
        ("TOKEN", "1", True, False),
        ("forward", "0", True, True),
        ("forward", "0", False, True),
    ],
)
def test_swa_cadence(monkeypatch, cadence, mega, hip, expected):
    monkeypatch.setattr(schedule_batch, "_is_hip", hip)
    monkeypatch.setenv("SGLANG_SWA_EVICTION_CADENCE", cadence)
    monkeypatch.setenv("SGLANG_AMD_USE_FLYDSL_MEGA_MOE", mega)
    assert schedule_batch._swa_forward_cadence() is expected


def test_swa_cadence_rejects_unknown(monkeypatch):
    monkeypatch.setenv("SGLANG_SWA_EVICTION_CADENCE", "sometimes")
    with pytest.raises(ValueError):
        schedule_batch._swa_forward_cadence()


def _fake_op(supports_mask):
    return SimpleNamespace(op=SimpleNamespace(supports_combine_mask=supports_mask))


@pytest.mark.parametrize(
    "env, supports, expected",
    [("0", True, False), ("1", True, True), ("1", False, False)],
)
def test_mask_pad_rows_gate(monkeypatch, env, supports, expected):
    monkeypatch.setenv("SGLANG_AITER_MEGA_MASK_PAD_ROWS", env)
    monkeypatch.setattr(f"{_MOD}._import_aiter_megamoe", lambda: _fake_op(supports))
    assert mega_flydsl._mask_pad_rows() is expected


class _FakeMega:
    def __init__(self):
        self.calls = []

    def forward(self, x, wts, ids, *, slice_output=True, **kwargs):
        self.calls.append((ids.clone(), wts.clone(), kwargs))
        return torch.zeros((8, x.shape[1]), dtype=x.dtype)


def _fake_moe(experts=16, topk=2, hidden=8):
    def topk_fn(hidden_states, router_logits, **kwargs):
        n = hidden_states.shape[0]
        ids = torch.arange(n * topk, dtype=torch.int64).view(n, topk) % experts
        return SimpleNamespace(topk_ids=ids, topk_weights=torch.full((n, topk), 0.5))

    return SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=hidden,
            num_experts_per_tok=topk,
            moe_intermediate_size=16,
            swiglu_limit=0.0,
        ),
        num_fused_shared_experts=0,
        experts=SimpleNamespace(
            num_experts=experts, should_fuse_routed_scaling_factor_in_topk=True
        ),
        gate=lambda h, forward_batch=None: torch.zeros((h.shape[0], experts)),
        topk=topk_fn,
        is_hash=False,
        layer_id=0,
        routed_scaling_factor=1.0,
    )


def _patch_run(monkeypatch, mega, *, mask_env, supports=True):
    monkeypatch.setenv("SGLANG_AITER_MEGA_MASK_PAD_ROWS", mask_env)
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR", "256")
    monkeypatch.setattr(f"{_MOD}._import_aiter_megamoe", lambda: _fake_op(supports))
    monkeypatch.setattr(f"{_MOD}._get_or_build_mega_moe", lambda *a, **k: mega)
    monkeypatch.setattr(f"{_MOD}._swap_layer_weights", lambda *a: None)
    monkeypatch.setattr(
        f"{_MOD}.ExpertLocationDispatchInfo",
        SimpleNamespace(init_new=lambda **k: None),
    )

    def fill(x, n_valid, value):  # CPU stand-in for the Triton kernel
        x[int(n_valid.item()) :] = value

    monkeypatch.setattr(f"{_MOD}._fill_padded_rows", fill)


@pytest.mark.parametrize("mask_env, pad_id", [("1", -1), ("0", 16)])
def test_padded_rows_routed_to_mask(monkeypatch, mask_env, pad_id):
    mega = _FakeMega()
    _patch_run(monkeypatch, mega, mask_env=mask_env)
    moe = _fake_moe()
    hidden = torch.randn((6, 8))
    batch = SimpleNamespace(num_token_non_padded=torch.tensor([4]))
    out = mega_flydsl._run_mega_routed(moe, hidden, batch, None)
    assert out.shape == (6, 8)
    ids, _, kwargs = mega.calls[0]
    assert ids.dtype == torch.int32
    assert (ids[4:] == pad_id).all()
    assert (ids[:4] >= 0).all() and (ids[:4] < 16).all()
    assert kwargs == ({"mask_invalid_slots": True} if mask_env == "1" else {})


def test_idle_rank_dummy_row_masked(monkeypatch):
    mega = _FakeMega()
    _patch_run(monkeypatch, mega, mask_env="1")
    monkeypatch.setenv("SGLANG_AITER_MEGA_RANK_SYNC", "0")
    out = mega_flydsl._run_mega_routed(_fake_moe(), torch.zeros((0, 8)), None, None)
    assert out.shape == (0, 8)
    ids, wts, kwargs = mega.calls[0]
    assert (ids == -1).all() and (wts == 0).all()
    assert kwargs == {"mask_invalid_slots": True}


def _decode_cfg(mode="decode", max_bs=48, draft=4):
    return SimpleNamespace(
        disaggregation_mode=mode,
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=max_bs)),
        speculative_algorithm="DSPARK",
        speculative_num_draft_tokens=draft,
    )


def test_decode_budget(monkeypatch):
    monkeypatch.setenv("SGLANG_AMD_USE_FLYDSL_MEGA_MOE", "1")
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR", "256")
    validate_aiter_mega_moe_decode_budget(_decode_cfg(), "m")  # 192 <= 256
    with pytest.raises(ValueError, match="SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR"):
        validate_aiter_mega_moe_decode_budget(_decode_cfg(max_bs=96), "m")
    # Prefill / agg roles keep their prefill-sized capacity unchecked here.
    validate_aiter_mega_moe_decode_budget(_decode_cfg(mode="prefill", max_bs=96), "m")
    validate_aiter_mega_moe_decode_budget(_decode_cfg(mode="null", max_bs=96), "m")
    monkeypatch.setenv("SGLANG_AMD_USE_FLYDSL_MEGA_MOE", "0")
    validate_aiter_mega_moe_decode_budget(_decode_cfg(max_bs=96), "m")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
