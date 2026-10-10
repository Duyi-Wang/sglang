import sys
from types import SimpleNamespace

import pytest

import sglang.srt.layers.moe.mega_moe_flydsl as mega_flydsl
import sglang.srt.layers.moe.utils as moe_utils
import sglang.srt.utils.common as common
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_MOD = "sglang.srt.layers.moe.mega_moe_flydsl"


@pytest.fixture
def decode_1024(monkeypatch):
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR", "8192")
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR", "1024")
    monkeypatch.setattr(f"{_MOD}.is_dsa_enable_prefill_cp", lambda: False)
    monkeypatch.setattr(f"{_MOD}.get_is_capture_mode", lambda: False)
    monkeypatch.setattr(f"{_MOD}.get_dp_raw_global_max_tokens", lambda: None)


def _global_tokens(monkeypatch, tokens):
    monkeypatch.setattr(f"{_MOD}.get_dp_global_num_tokens", lambda: tokens)


def test_disabled_keeps_main_instance(monkeypatch):
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR", "8192")
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR", "0")
    _global_tokens(monkeypatch, [7, 7])
    assert mega_flydsl._select_mtpr() == 8192


def test_pick_follows_dp_global_max(monkeypatch, decode_1024):
    # The decode instance only when the largest rank fits: a rank with few
    # tokens must follow a peer that is prefilling.
    _global_tokens(monkeypatch, [896, 0, 7, 112])
    assert mega_flydsl._select_mtpr() == 1024
    _global_tokens(monkeypatch, [112, 8192, 7, 112])
    assert mega_flydsl._select_mtpr() == 8192
    _global_tokens(monkeypatch, [0] * 8)  # every rank idle
    assert mega_flydsl._select_mtpr() == 1024


def test_no_dp_counts_falls_back_to_main(monkeypatch, decode_1024):
    _global_tokens(monkeypatch, None)
    assert mega_flydsl._select_mtpr() == 8192


def test_prefill_cp_falls_back_to_main(monkeypatch, decode_1024):
    monkeypatch.setattr(f"{_MOD}.is_dsa_enable_prefill_cp", lambda: True)
    _global_tokens(monkeypatch, [7, 7])
    assert mega_flydsl._select_mtpr() == 8192


@pytest.mark.parametrize("bad", ["-512", "768", "8192", "16384"])
def test_decode_mtpr_validation(monkeypatch, bad):
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR", "8192")
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR", bad)
    with pytest.raises(ValueError):
        mega_flydsl._decode_mtpr()


def test_both_instances_built_in_fixed_order(monkeypatch, decode_1024):
    built = []

    def fake_build(layer, *, mtpr, **kwargs):
        built.append(mtpr)
        return f"mega{mtpr}"

    monkeypatch.setattr(f"{_MOD}._get_or_build_mega_moe", fake_build)
    # Whichever capacity the first forward selects, the main one is built first,
    # so every rank reaches the construction barriers in the same order.
    assert mega_flydsl._get_mega_moe_for_forward(None, selected_mtpr=1024) == (
        "mega1024"
    )
    assert built == [8192, 1024]
    built.clear()
    assert mega_flydsl._get_mega_moe_for_forward(None, selected_mtpr=8192) == (
        "mega8192"
    )
    assert built == [8192, 1024]


def _dep8_megamoe(monkeypatch, *, hip=True):
    parallel = SimpleNamespace(
        enable_dp_attention=True,
        attn_dp_enabled=True,
        dp_size=8,
        num_dp_ranks=8,
        tp_size=8,
        moe_dense_tp_size=1,
        enable_dp_lm_head=True,
    )
    monkeypatch.setattr(common, "get_parallel", lambda: parallel)
    monkeypatch.setattr(
        common,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(elastic_ep_backend=None)),
    )
    monkeypatch.setattr(common, "is_hip", lambda: hip)
    backend = SimpleNamespace(
        is_none=lambda: False,
        is_flashinfer=lambda: False,
        is_mori=lambda: False,
        is_megamoe=lambda: True,
    )
    monkeypatch.setattr(moe_utils, "get_moe_a2a_backend", lambda: backend)
    monkeypatch.setenv("SGLANG_AMD_USE_FLYDSL_MEGA_MOE", "1")


def test_decode_instance_turns_on_dp_sync_bookkeeping(monkeypatch):
    _dep8_megamoe(monkeypatch)
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR", "0")
    assert not common.require_mlp_tp_gather()
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR", "1024")
    assert common.require_mlp_tp_gather()


def test_dp_sync_bookkeeping_untouched_off_hip(monkeypatch):
    _dep8_megamoe(monkeypatch, hip=False)
    monkeypatch.setenv("SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR", "1024")
    assert not common.require_mlp_tp_gather()


def test_idle_rank_of_global_prefill_follows_the_prefilling_peer(
    monkeypatch, decode_1024
):
    # Regression: an idle rank of a global prefill rewrites its own published
    # global_num_tokens to [1] * dp. It must still pick the instance its
    # prefilling peers pick, from the all-gathered raw max.
    monkeypatch.setattr(f"{_MOD}.get_dp_raw_global_max_tokens", lambda: 3000)
    _global_tokens(monkeypatch, [1] * 8)  # idle rank's view
    assert mega_flydsl._select_mtpr() == 8192
    _global_tokens(monkeypatch, [3000, 0, 0, 0, 0, 0, 0, 0])  # prefilling rank's view
    assert mega_flydsl._select_mtpr() == 8192
    monkeypatch.setattr(f"{_MOD}.get_dp_raw_global_max_tokens", lambda: 1)
    assert mega_flydsl._select_mtpr() == 1024


def test_capture_uses_captured_shape(monkeypatch, decode_1024):
    monkeypatch.setattr(f"{_MOD}.get_is_capture_mode", lambda: True)
    monkeypatch.setattr(f"{_MOD}.get_dp_raw_global_max_tokens", lambda: 3000)  # stale
    _global_tokens(monkeypatch, [896] * 8)
    assert mega_flydsl._select_mtpr() == 1024


def test_raw_max_resets_with_every_dp_buffer_update():
    from sglang.srt.layers import dp_attention as dpa

    dpa.set_dp_raw_global_max_tokens(3000)
    assert dpa.get_dp_raw_global_max_tokens() == 3000
    dpa.set_dp_buffer_len(8, 1, True, [1] * 8)  # e.g. a graph capture step
    assert dpa.get_dp_raw_global_max_tokens() is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
