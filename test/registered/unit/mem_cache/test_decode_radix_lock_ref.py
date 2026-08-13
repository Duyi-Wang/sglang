"""
Unit tests for lock_ref correctness in decode disagg radix cache scenarios.

Verifies that inc_lock_ref / dec_lock_ref are balanced across the four
transfer scenarios identified in PR #19746:

1. Incremental transfer & success (prefix match > 0)
   inc_lock_ref(pop_preallocated) -> dec+inc(cache_unfinished_req) -> dec(cache_finished_req)

2. Full transfer & success (prefix match == 0, full KV transferred)
   inc_lock_ref(get_new_prebuilt_batch) -> dec+inc(cache_unfinished_req) -> dec(cache_finished_req)

3. Incremental transfer & failure (prefix match > 0, transfer fails)
   inc_lock_ref(pop_preallocated) -> dec(cache_finished_req via release_kv_cache is_insert=False)

4. Full transfer & failure (prefix match == 0, transfer fails)
   no inc_lock_ref -> dec(root_node) is no-op since root lock_ref starts at 1

Usage:
    python -m pytest test/registered/unit/mem_cache/test_decode_radix_lock_ref.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import ast
import inspect
import textwrap
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    alloc_for_decode_prealloc,
)
from sglang.srt.disaggregation.decode_hicache_mixin import DecodePrefixMatch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    IncLockRefResult,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.unified_cache.components.tree_component import ComponentType
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils.common import Range


def _make_cache_with_pools(page_size=1):
    """Create a RadixCache with mock pools sufficient for cache_unfinished/finished_req."""
    mock_allocator = MagicMock()
    mock_allocator.device = torch.device("cpu")

    # req_to_token pool: stores kv indices per request slot
    max_seq_len = 64
    max_batch = 4
    req_to_token = torch.zeros(max_batch, max_seq_len, dtype=torch.int64)

    mock_pool = MagicMock()
    mock_pool.req_to_token = req_to_token
    mock_pool.write = lambda idx_tuple, values: req_to_token.__setitem__(
        idx_tuple, values
    )

    cache = RadixCache.create_simulated(
        mock_allocator=mock_allocator, page_size=page_size
    )
    cache.req_to_token_pool = mock_pool
    return cache, req_to_token


class MockReq:
    """Minimal mock Req with fields needed by cache_unfinished/finished_req."""

    def __init__(self, fill_ids, req_pool_idx=0, cache_protected_len=0, last_node=None):
        self.full_untruncated_fill_ids = array("q", fill_ids)
        self.extend_range = Range(0, len(self.full_untruncated_fill_ids))
        self.origin_input_ids = array(
            "q", fill_ids[:-1] if len(fill_ids) > 1 else fill_ids
        )
        self.output_ids = array("q", [fill_ids[-1]] if len(fill_ids) > 1 else [])
        self.req_pool_idx = req_pool_idx
        self.cache_protected_len = cache_protected_len
        self.last_node = last_node
        self.extra_key = None
        self.cache_salt = None
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.priority = 0
        self.kv_committed_len = len(fill_ids)
        self.kv = SimpleNamespace(kv_allocated_len=len(fill_ids))

    def get_fill_ids(self):
        return self.full_untruncated_fill_ids[: self.extend_range.end]


def _make_req(fill_ids, req_pool_idx=0, cache_protected_len=0, last_node=None):
    return MockReq(fill_ids, req_pool_idx, cache_protected_len, last_node)


class _Admitted(Exception):
    """Raised from a `_pre_alloc` stub to mark the admission point."""


def _real_req(token_ids, *, max_new_tokens=16, rid="req-1") -> Req:
    """A real ``Req``, so prefix-match bookkeeping fields behave like production.

    ``MagicMock`` cannot stand in wherever the code under test reads a match
    field back (``_compute_max_prefix_len``, ``len(prefix_indices)``): an
    auto-created attribute is truthy and un-arithmetic, which either hides the
    assertion or turns it into a TypeError.
    """
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", token_ids),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=max_new_tokens),
    )
    req.pd_rebootstrap_in_progress = False
    return req


def _make_prealloc_queue(
    *,
    decode_req,
    tree_cache,
    budgets,
    page_size=1,
    allocator=None,
    uses_swa_tail_prealloc=False,
    sliding_window_size=None,
    swa_budgets=(10**9, 10**9),
):
    """A `DecodePreallocQueue` with only `pop_preallocated`'s collaborators faked.

    Everything on the admission path itself -- `_match_prefix_and_lock`,
    `_release_prefix_lock`, `_admission_fits`, `_takes_swa_tail_path` -- stays
    real; `budgets` is the `_allocatable_token_budgets` side_effect list, i.e.
    the pool readings the loop is driven with.
    """
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.pp_size = 1
    queue.queue = [decode_req]
    queue.pending_reqs = []
    queue.retracted_queue = []
    queue.num_reserved_decode_tokens = 0
    queue.tree_cache = tree_cache
    queue._resolve_pending_reqs = MagicMock()
    queue._update_handshake_waiters = MagicMock()
    queue.transfer_queue = MagicMock(queue=[], enable_staging=False)
    queue.req_to_token_pool = MagicMock()
    queue.req_to_token_pool.available_size.return_value = 1
    queue.req_to_metadata_buffer_idx_allocator = MagicMock()
    queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1
    queue.token_to_kv_pool = MagicMock()
    if allocator is None:
        allocator = MagicMock(page_size=page_size, device=torch.device("cpu"))
    queue.token_to_kv_pool_allocator = allocator
    queue.token_to_kv_pool_allocator.page_size = page_size
    queue._allocatable_token_budgets = MagicMock(side_effect=budgets)
    if uses_swa_tail_prealloc:
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_aware_allocatable_token_budgets = MagicMock(return_value=swa_budgets)

    scheduler = MagicMock()
    scheduler.running_batch = MagicMock(reqs=[])
    scheduler.server_args = MagicMock()
    scheduler.server_args.disaggregation_decode_enable_radix_cache = True
    scheduler.server_args.disable_radix_cache = False
    scheduler.enable_hisparse = False
    scheduler.enable_decode_hicache = False
    scheduler.enable_priority_scheduling = False
    scheduler.sliding_window_size = sliding_window_size
    scheduler.waiting_queue = []
    scheduler.last_batch = None
    scheduler.output_streamer = MagicMock()
    queue.scheduler = scheduler
    return queue


def _decode_request(req):
    decode_req = MagicMock()
    decode_req.req = req
    decode_req.waiting_for_input = True
    # Non-rebootstrap: exercise the normal decode radix-cache path (a truthy
    # MagicMock would disable it via the `not decode_req.is_rebootstrap` gate).
    decode_req.is_rebootstrap = False
    return decode_req


class TestDecodeLockRefScenarios(unittest.TestCase):
    """Test lock_ref balance across decode transfer scenarios."""

    def _populate_prefix(self, cache, prefix_ids, prefix_values):
        """Insert a prefix into the tree so future requests can match it."""
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", prefix_ids)),
                value=torch.tensor(prefix_values, dtype=torch.int64),
            )
        )

    def test_match_prefix_preserves_complete_lock_ownership(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.tree_cache = MagicMock()
        queue._build_decode_prefix_match = MagicMock(return_value=MagicMock())

        req = MagicMock()
        req.origin_input_ids = [1, 2, 3, 4]
        node = object()
        match_result = MagicMock(last_device_node=node)
        lock_result = IncLockRefResult(
            swa_uuid_for_lock=17,
            skip_lock_node_ids={ComponentType.SWA: {41, 42}},
        )
        queue.tree_cache.inc_lock_ref.return_value = lock_result

        with patch(
            "sglang.srt.disaggregation.decode.match_prefix_for_req",
            return_value=match_result,
        ):
            queue._match_prefix_and_lock(req)

        queue.tree_cache.inc_lock_ref.assert_called_once_with(node)
        # The acquire's skip set must be recorded, else the matching release
        # would drop a lock this request never took.
        self.assertEqual(req.swa_uuid_for_lock, 17)
        self.assertEqual(req.skip_lock_node_ids, {ComponentType.SWA: {41, 42}})

    def test_hicache_restore_commit_replays_both_ownership_tokens(self):
        """The HiCache restore handoff releases the ADMISSION lock and hands the
        RESTORE lock to the request. Each release must replay the token of the
        acquire it matches: releasing the admission lock with an empty token
        walks past the SWA window boundary to the root and drops locks another
        request holds; carrying the admission token onto the restored node makes
        the request's own later release do the same."""
        from sglang.srt.disaggregation.decode_hicache_mixin import (
            DecodeHiCacheTransferMixin,
        )

        queue = DecodeHiCacheTransferMixin.__new__(DecodeHiCacheTransferMixin)
        queue.tree_cache = MagicMock()

        admission_node, restored_node = object(), object()
        req = MagicMock()
        req.swa_uuid_for_lock = 17
        req.skip_lock_node_ids = {ComponentType.SWA: {41}}

        decode_req = MagicMock()
        decode_req.req = req
        decode_req.hicache_restored_node = restored_node
        decode_req.hicache_restored_lock = DecLockRefParams(
            swa_uuid_for_lock=99, skip_lock_node_ids={ComponentType.SWA: {7}}
        )
        decode_req.hicache_restored_kv_indices = torch.arange(2, dtype=torch.int64)
        decode_req.prefix_match = MagicMock(
            needs_local_restore=True,
            last_device_node=admission_node,
            l1_prefix_len=0,
            decode_prefix_len=2,
            prefix_indices=torch.empty(0, dtype=torch.int64),
        )

        queue._commit_hicache_local_restore_to_req(decode_req)

        queue.tree_cache.dec_lock_ref.assert_called_once_with(
            admission_node,
            DecLockRefParams(
                swa_uuid_for_lock=17, skip_lock_node_ids={ComponentType.SWA: {41}}
            ),
        )
        self.assertIs(req.last_node, restored_node)
        self.assertEqual(req.swa_uuid_for_lock, 99)
        self.assertEqual(req.skip_lock_node_ids, {ComponentType.SWA: {7}})

    @staticmethod
    def _swa_queue(*, page_size: int = 64, window: int = 128, fill_len: int = 1024):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.token_to_kv_pool_allocator = MagicMock(page_size=page_size)
        queue.num_reserved_decode_tokens = 0
        queue.scheduler = MagicMock()
        queue.scheduler.sliding_window_size = window
        queue.scheduler.server_args.disable_radix_cache = False
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._pre_alloc_fill_len = MagicMock(return_value=fill_len)
        return queue

    def test_swa_admission_charge_matches_the_allocator_path(self):
        """The admission charge must equal what `_pre_alloc` then allocates.

        `_takes_swa_tail_path` is the single predicate both consult. Whenever the
        sliding window fits inside the delta this call allocates, the SWA debit
        is one window -- on a decode-radix hit exactly as on a miss. Charging the
        delta there would over-reserve; charging the window when the allocator
        instead falls back to `alloc_extend` would under-reserve by
        (delta - window) per request and trip the `kv_loc is not None` assert.
        """
        fill_len = 1024
        queue = self._swa_queue(fill_len=fill_len)
        req = MagicMock()
        window = queue._swa_tail_len(fill_len)
        self.assertEqual(window, 128)

        # Miss: tail-only, one window.
        _, swa_on_miss = queue._prealloc_kv_lens(req, prefix_len=0)
        self.assertEqual(swa_on_miss, window)

        # Hit with room for the window inside the delta: still one window.
        _, swa_on_hit = queue._prealloc_kv_lens(req, prefix_len=128)
        self.assertEqual(swa_on_hit, window)
        self.assertEqual(swa_on_hit, swa_on_miss)

        # Hit so long that the window would reach back into the reused prefix:
        # the tail path is not expressible, so the charge is the whole delta.
        deep_prefix = fill_len - 64  # delta 64 < window 128
        _, swa_deep = queue._prealloc_kv_lens(req, prefix_len=deep_prefix)
        self.assertFalse(
            queue._takes_swa_tail_path(fill_len=fill_len, total_prefix_len=deep_prefix)
        )
        self.assertEqual(
            swa_deep,
            queue._required_alloc_tokens(fill_len=fill_len, prefix_len=deep_prefix),
        )

    def test_swa_charge_follows_total_prefix_not_l1_prefix(self):
        """The delta is measured from `total_prefix_len` (L1+L2+L3), which is
        what `_pre_alloc` extends from. With decode HiCache on they diverge, and
        charging against the L1 prefix alone would disagree with the allocator.
        """
        fill_len = 1024
        queue = self._swa_queue(fill_len=fill_len)
        req = MagicMock()
        # L1 hit is small, but prefill was promised nearly the whole prompt, so
        # the delta is tiny and the tail path is unavailable.
        _, swa = queue._prealloc_kv_lens(
            req, prefix_len=64, total_prefix_len=fill_len - 64
        )
        self.assertEqual(
            swa, queue._required_alloc_tokens(fill_len=fill_len, prefix_len=64)
        )
        # Same L1 prefix, no HiCache gap: the tail path applies.
        _, swa_no_gap = queue._prealloc_kv_lens(req, prefix_len=64)
        self.assertEqual(swa_no_gap, queue._swa_tail_len(fill_len))

    def test_swa_charge_is_the_page_the_allocator_debits_not_the_raw_tail(self):
        """The charge has to be what `alloc_extend_swa_tail` actually takes.

        The allocator reserves whole pages --
        `num_swa_pages = ceil(swa_tail_len / page_size)` -- while
        `_swa_tail_len` subtracts a page-aligned window start from `fill_len`,
        so it lands on a page boundary only when `fill_len` does. `fill_len` is
        `_pre_alloc_fill_len`, a live token count, so it generally does not:
        charging the raw tail under-reserves by up to a page on nearly every
        admission, and an SWA charge below the SWA debit drains the pool faster
        than admission believes until `_pre_alloc` hits its
        `kv_loc is not None` assert.

        DeepSeek-V4-Pro's geometry makes this unavoidable rather than a corner
        case: window 128 is smaller than page 256, so the tail is a window plus
        a partial page and can never be aligned unless `fill_len` is.
        """
        page_size, window = 256, 128
        # 1000 = 3 * 256 + 232 -- the tail stops 24 tokens short of a page.
        fill_len = 1000
        queue = self._swa_queue(page_size=page_size, window=window, fill_len=fill_len)
        req = MagicMock()

        raw_tail = queue._swa_tail_len(fill_len)
        self.assertEqual(raw_tail, 232)
        self.assertNotEqual(raw_tail % page_size, 0)

        _, swa = queue._prealloc_kv_lens(req, prefix_len=0)
        self.assertEqual(swa, page_size)
        self.assertGreater(swa, raw_tail)

    def test_a_hit_is_never_charged_more_than_a_miss(self):
        """The property the change buys, swept over every prefix length.

        Before, a hit paid the whole delta while a miss paid one window, so a
        hit could be rejected where the same request would have been admitted as
        a miss -- and the retry would re-derive the same hit, head-of-line
        blocking the decode queue. Now:
          * window fits in the delta -> hit charge == miss charge (one window)
          * window does not fit      -> the delta is SMALLER than the window,
                                        so the charge is smaller still
        Either way the hit is never the more expensive branch, in either pool.

        Swept over both geometries because rounding the tail up to a page could
        invert this: the production layout (window < page) is the one where the
        rounding is not a no-op, so an over-correction there would make a hit
        cost more than the miss it replaced and put the head-of-line block back.
        """
        for page_size, window in ((64, 128), (256, 128)):
            with self.subTest(page_size=page_size, window=window):
                fill_len = 1000
                queue = self._swa_queue(
                    page_size=page_size, window=window, fill_len=fill_len
                )
                req = MagicMock()
                req.sampling_params.max_new_tokens = 0

                _, swa_on_miss = queue._prealloc_kv_lens(req, prefix_len=0)
                full_on_miss = queue._required_alloc_tokens(
                    fill_len=fill_len, prefix_len=0
                )

                for prefix_len in range(0, fill_len, 8):
                    _, swa_on_hit = queue._prealloc_kv_lens(req, prefix_len=prefix_len)
                    full_on_hit = queue._required_alloc_tokens(
                        fill_len=fill_len, prefix_len=prefix_len
                    )
                    self.assertLessEqual(swa_on_hit, swa_on_miss, f"{prefix_len=} SWA")
                    self.assertLessEqual(
                        full_on_hit, full_on_miss, f"{prefix_len=} FULL"
                    )

    def test_ordinary_hit_admitted_where_it_used_to_be_rejected(self):
        """An SWA budget that fits one window but not the delta: before the
        change this hit was rejected and fell back to a miss; now it is admitted
        with its prefix intact, which is the whole point of the feature."""
        fill_len = 1024
        queue = self._swa_queue(fill_len=fill_len)
        req = MagicMock()
        req.sampling_params.max_new_tokens = 0

        budget = dict(
            origin_input_len=fill_len,
            full_allocatable_tokens=10**9,
            swa_allocatable_tokens=256,  # one window fits, an 896-token delta does not
            retractable_tokens=0,
            retractable_swa_tokens=0,
            uses_swa_tail_prealloc=True,
        )
        fits_hit, _ = queue._admission_fits(req, prefix_len=128, **budget)
        self.assertTrue(fits_hit)

    def test_tail_path_allocates_the_delta_when_hicache_owns_part_of_the_prefix(self):
        """L1 miss with an L2/L3 hit: `prefix_len == 0` but `total_prefix_len > 0`.

        The old gate (`uses_swa_tail = ... and prefix_len == 0`) took the tail
        branch here and asked for the WHOLE sequence -- `prefix_lens=[0]`,
        `extend_num_tokens=fill_len` -- while `_pre_alloc` writes the result at
        offset `total_prefix_len`. That over-allocates by `total_prefix_len`
        slots and writes `[fill_len, total_prefix_len + fill_len)`, past the end
        of the sequence.

        The combination is only reachable once the decode radix cache is allowed
        on a hybrid-SWA pool: `_uses_swa_tail_prealloc()` is true exactly for
        those pools, and `enable_decode_hicache` requires
        `disaggregation_decode_enable_radix_cache`.
        """
        fill_len, total_prefix_len = 1024, 256
        delta_len = fill_len - total_prefix_len

        allocator = MagicMock(page_size=64, device=torch.device("cpu"))
        allocator.alloc_extend_swa_tail.return_value = torch.arange(
            delta_len, dtype=torch.int64
        )
        del allocator.c4_attn_allocator  # not the NPU allocator

        req = MagicMock()
        req.kv = None
        alloc_for_decode_prealloc(
            allocator,
            req=req,
            fill_len=fill_len,
            delta_len=delta_len,
            prefix_len=0,
            total_prefix_len=total_prefix_len,
            prefix_indices=None,
            uses_swa_tail=True,
            swa_tail_len=128,
        )

        allocator.alloc_extend_swa_tail.assert_called_once()
        kwargs = allocator.alloc_extend_swa_tail.call_args.kwargs
        self.assertEqual(
            kwargs["extend_num_tokens"],
            delta_len,
            "asked for the whole sequence; the caller writes at total_prefix_len",
        )
        self.assertEqual(int(kwargs["prefix_lens_cpu"][0]), total_prefix_len)
        self.assertEqual(int(kwargs["seq_lens_cpu"][0]), fill_len)

    def test_allocator_site_asks_the_same_predicate_as_the_charge(self):
        """The charge and the allocation must not be able to disagree.

        `_prealloc_kv_lens` bills the SWA pool for one window whenever
        `_takes_swa_tail_path` says so. If `_pre_alloc` decided the allocator
        branch by any other rule, a request could be billed for a window and
        then allocate the whole delta -- the SWA pool drains silently until
        `_pre_alloc`'s own `kv_loc is not None` assert fires mid-serving, which
        is not a failure any charge-side test can see.

        Pinned at the source level because reaching the branch for real needs a
        GPU-sized two-pool allocator.
        """
        source = inspect.getsource(DecodePreallocQueue._pre_alloc)
        tree = ast.parse(textwrap.dedent(source))

        assigns = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "uses_swa_tail"
                for t in node.targets
            )
        ]
        self.assertEqual(len(assigns), 1, "expected one uses_swa_tail decision")
        (decision,) = assigns
        self.assertIsInstance(
            decision.value, ast.Call, f"not a call: {ast.unparse(decision.value)}"
        )
        self.assertEqual(decision.value.func.attr, "_takes_swa_tail_path")
        self.assertEqual(
            sorted(kw.arg for kw in decision.value.keywords),
            ["fill_len", "total_prefix_len"],
        )

        # ...and the tail call must extend from that same prefix, not from zero.
        alloc_tree = ast.parse(
            textwrap.dedent(inspect.getsource(alloc_for_decode_prealloc))
        )
        calls = [
            node
            for node in ast.walk(alloc_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "alloc_extend_swa_tail"
        ]
        self.assertEqual(len(calls), 1)
        kwargs = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
        self.assertIn("total_prefix_len", kwargs["prefix_lens"])
        self.assertIn("total_prefix_len", kwargs["prefix_lens_cpu"])
        self.assertEqual(kwargs["extend_num_tokens"], "delta_len")

    def test_miss_fallback_still_wired_even_though_swa_can_no_longer_trigger_it(self):
        """The fallback at `pop_preallocated` is now belt-and-braces for the SWA
        budget, but it must stay wired: it is the only thing standing between a
        future non-monotonic charge and a permanently blocked decode queue."""
        source = inspect.getsource(DecodePreallocQueue.pop_preallocated)
        self.assertIn("_release_prefix_lock", source)
        self.assertIn("prefix_len=0", source)

    def test_incremental_transfer_success(self):
        """Scenario 1: prefix match > 0, transfer succeeds.

        Flow: inc_lock_ref(pop_preallocated)
              -> dec_lock_ref + inc_lock_ref(cache_unfinished_req)
              -> dec_lock_ref(cache_finished_req)
        """
        cache, req_to_token = _make_cache_with_pools()

        # Pre-populate a prefix [1,2,3] in the tree
        prefix = [1, 2, 3]
        prefix_vals = [10, 20, 30]
        self._populate_prefix(cache, prefix, prefix_vals)

        # Match prefix (simulates _match_prefix_and_lock in pop_preallocated)
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", prefix))))
        matched_node = result.last_device_node
        prefix_len = len(result.device_indices)
        self.assertEqual(prefix_len, 3)

        # Step 1: inc_lock_ref (pop_preallocated locks the matched node)
        cache.inc_lock_ref(matched_node)
        self.assertGreater(matched_node.lock_ref, 0)

        # Simulate _pre_alloc: write prefix + new tokens to req_to_token
        full_ids = [1, 2, 3, 4, 5]  # prefix + 2 new tokens
        full_vals = [10, 20, 30, 40, 50]
        req_to_token[0, : len(full_vals)] = torch.tensor(full_vals, dtype=torch.int64)

        req = _make_req(
            fill_ids=full_ids,
            req_pool_idx=0,
            cache_protected_len=prefix_len,
            last_node=matched_node,
        )

        # Step 2: cache_unfinished_req (dec old lock, inc new lock)
        cache.cache_unfinished_req(req)

        # Step 3: cache_finished_req with is_insert=True (dec lock)
        cache.cache_finished_req(req, kv_len_to_handle=req.kv_committed_len)

        # Verify: all non-root nodes should have lock_ref == 0
        # (root always has lock_ref == 1)
        self.assertEqual(cache.root_node.lock_ref, 1)
        self.assertEqual(cache.protected_size(), 0)
        # The evictable size should equal total inserted tokens
        self.assertEqual(cache.evictable_size(), len(full_ids))

    def test_full_transfer_success(self):
        """Scenario 2: no prefix match, full KV transferred, succeeds.

        Flow: inc_lock_ref(root, via init_next_round_input/get_new_prebuilt_batch)
              -> dec_lock_ref + inc_lock_ref(cache_unfinished_req)
              -> dec_lock_ref(cache_finished_req)
        """
        cache, req_to_token = _make_cache_with_pools()

        # No prefix in tree -- match returns root
        full_ids = [10, 20, 30]
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", full_ids)))
        )
        matched_node = result.last_device_node
        self.assertEqual(len(result.device_indices), 0)  # no match
        # matched_node is root

        root_lock_before = cache.root_node.lock_ref
        # Step 1: inc_lock_ref on root (simulates get_new_prebuilt_batch)
        # Note: inc/dec_lock_ref skip the root node (while node != root_node),
        # so this is a no-op. Root always keeps lock_ref=1.
        cache.inc_lock_ref(matched_node)
        self.assertEqual(cache.root_node.lock_ref, root_lock_before)  # no-op on root

        # Write full KV to pool
        full_vals = [100, 200, 300]
        req_to_token[0, : len(full_vals)] = torch.tensor(full_vals, dtype=torch.int64)

        req = _make_req(
            fill_ids=full_ids,
            req_pool_idx=0,
            cache_protected_len=0,
            last_node=matched_node,
        )

        # Step 2: cache_unfinished_req (dec root=no-op, inc new leaf)
        cache.cache_unfinished_req(req)

        # Step 3: cache_finished_req (dec leaf)
        cache.cache_finished_req(req, kv_len_to_handle=req.kv_committed_len)

        # Root lock unchanged, all nodes unlocked
        self.assertEqual(cache.root_node.lock_ref, root_lock_before)
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(cache.evictable_size(), len(full_ids))

    def test_incremental_transfer_failure(self):
        """Scenario 3: prefix match > 0, transfer fails.

        Flow: inc_lock_ref(pop_preallocated)
              -> dec_lock_ref(cache_finished_req via release_kv_cache is_insert=False)
        """
        cache, req_to_token = _make_cache_with_pools()

        # Pre-populate prefix
        prefix = [1, 2, 3]
        prefix_vals = [10, 20, 30]
        self._populate_prefix(cache, prefix, prefix_vals)

        # Match and lock
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", prefix))))
        matched_node = result.last_device_node
        prefix_len = len(result.device_indices)

        cache.inc_lock_ref(matched_node)
        # Prefix tokens should now be protected (locked)
        self.assertGreater(cache.protected_size(), 0)

        # Simulate _pre_alloc with additional tokens
        full_ids = [1, 2, 3, 4, 5]
        full_vals = [10, 20, 30, 40, 50]
        req_to_token[0, : len(full_vals)] = torch.tensor(full_vals, dtype=torch.int64)

        req = _make_req(
            fill_ids=full_ids,
            req_pool_idx=0,
            cache_protected_len=prefix_len,
            last_node=matched_node,
        )

        # Transfer fails -> cache_finished_req with is_insert=False
        # This frees delta tokens and dec_lock_ref on last_node
        cache.cache_finished_req(
            req, is_insert=False, kv_len_to_handle=req.kv_committed_len
        )

        # The prefix node should be unlocked (back to evictable)
        self.assertEqual(cache.root_node.lock_ref, 1)
        self.assertEqual(cache.protected_size(), 0)
        # Prefix tokens should still be in tree and evictable
        self.assertEqual(cache.evictable_size(), len(prefix))

    def test_full_transfer_failure(self):
        """Scenario 4: no prefix match, transfer fails.

        Flow: _match_prefix_and_lock sets last_node=root and calls
              inc_lock_ref(root) which is a no-op. On failure,
              cache_finished_req calls dec_lock_ref(root) which is also
              a no-op. Net: balanced.
        """
        cache, req_to_token = _make_cache_with_pools()

        root_lock_before = cache.root_node.lock_ref

        # No prefix in tree -- match returns root (simulates _match_prefix_and_lock)
        full_ids = [10, 20, 30]
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", full_ids)))
        )
        matched_node = result.last_device_node
        self.assertIs(matched_node, cache.root_node)

        # inc_lock_ref(root) is a no-op
        cache.inc_lock_ref(matched_node)
        self.assertEqual(cache.root_node.lock_ref, root_lock_before)

        full_vals = [100, 200, 300]
        req_to_token[0, : len(full_vals)] = torch.tensor(full_vals, dtype=torch.int64)

        # last_node = root (as set by _match_prefix_and_lock)
        req = _make_req(
            fill_ids=full_ids,
            req_pool_idx=0,
            cache_protected_len=0,
            last_node=matched_node,
        )

        # Transfer fails -> cache_finished_req with is_insert=False
        # dec_lock_ref(root) is a no-op
        cache.cache_finished_req(
            req, is_insert=False, kv_len_to_handle=req.kv_committed_len
        )

        # Root lock unchanged, nothing protected or evictable
        self.assertEqual(cache.root_node.lock_ref, root_lock_before)
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(cache.evictable_size(), 0)

    def test_pop_preallocated_rechecks_budget_after_lock(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1

        req = MagicMock()
        req.rid = "req-1"
        req.origin_input_ids = list(range(8))
        req.output_ids = [99]
        # Bound separately: the release clears `req.last_node`, so the expected
        # dec_lock_ref argument has to be captured before the call.
        matched_node = object()
        req.last_node = matched_node
        req.finished_reason = None
        req.cache_protected_len = 0
        req.sampling_params.max_new_tokens = 16
        # Ownership token recorded by the acquire in `_match_prefix_and_lock`
        # (mocked out below); the budget-recheck release must replay it.
        req.swa_uuid_for_lock = 17
        req.skip_lock_node_ids = {ComponentType.SWA: {41}}

        decode_req = MagicMock()
        decode_req.req = req
        decode_req.waiting_for_input = True
        # Non-rebootstrap request: exercise the normal decode radix-cache path
        # (a truthy MagicMock would disable use_decode_radix_cache via the
        # `not decode_req.is_rebootstrap` gate in pop_preallocated).
        decode_req.is_rebootstrap = False

        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._match_prefix_and_lock = MagicMock(
            return_value=DecodePrefixMatch(
                prefix_indices=torch.arange(4, dtype=torch.int64),
                l2_host_hit_length=0,
                l3_storage_hit_length=0,
                last_device_node=req.last_node,
            )
        )
        queue._pre_alloc = MagicMock(
            side_effect=AssertionError("_pre_alloc should not run")
        )
        queue.transfer_queue = MagicMock(queue=[], enable_staging=False)
        queue.tree_cache = MagicMock()
        queue.tree_cache.dec_lock_ref = MagicMock()
        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 1
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1
        queue.token_to_kv_pool = MagicMock()
        queue.token_to_kv_pool_allocator = MagicMock()
        queue.token_to_kv_pool_allocator.page_size = 4

        running_batch = MagicMock()
        running_batch.reqs = []
        server_args = MagicMock()
        server_args.disaggregation_decode_enable_radix_cache = True
        scheduler = MagicMock()
        scheduler.running_batch = running_batch
        scheduler.server_args = server_args
        scheduler.enable_hisparse = False
        scheduler.waiting_queue = []
        scheduler.last_batch = None
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        # Initial budget says the request fits; post-lock budget says it does
        # not. The third read is the post-release retry, which still rejects
        # here -- this case is about the recheck happening, not about recovery.
        queue._allocatable_token_budgets = MagicMock(side_effect=[8, 3, 3])

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [])
        queue._pre_alloc.assert_not_called()
        queue.tree_cache.dec_lock_ref.assert_called_once_with(
            matched_node,
            DecLockRefParams(
                swa_uuid_for_lock=17, skip_lock_node_ids={ComponentType.SWA: {41}}
            ),
        )
        self.assertEqual(queue._allocatable_token_budgets.call_count, 3)

    def test_miss_retry_is_judged_against_the_budget_the_release_restored(self):
        """The retry after dropping the prefix must re-read the pool.

        `_release_prefix_lock` hands the matched pages back to the evictable
        pool, and `_allocatable_token_budgets` counts evictable pages as
        available. Reusing the budget captured while the prefix was still
        locked therefore judges the miss against a pool that no longer exists,
        and rejects a request that now fits -- which is exactly the head-of-line
        block the retry was added to break, except now the queue stalls on the
        retry instead of on the hit.

        Budgets: the post-lock 3 rejects the hit, and the post-release read is
        64 because the release returned the pages -- enough for the miss, which
        needs `origin_input_len + max_new_tokens` = 24. Reaching `_pre_alloc` is
        the observable: with the stale 3 the loop breaks instead.
        """

        class _Admitted(Exception):
            """Raised from the `_pre_alloc` stub to mark the admission point."""

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1

        req = MagicMock()
        req.rid = "req-1"
        req.origin_input_ids = list(range(8))
        req.output_ids = [99]
        req.last_node = object()
        req.finished_reason = None
        req.cache_protected_len = 0
        req.sampling_params.max_new_tokens = 16
        req.swa_uuid_for_lock = 17
        req.skip_lock_node_ids = {ComponentType.SWA: {41}}

        decode_req = MagicMock()
        decode_req.req = req
        decode_req.waiting_for_input = True
        decode_req.is_rebootstrap = False

        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._match_prefix_and_lock = MagicMock(
            return_value=DecodePrefixMatch(
                prefix_indices=torch.arange(4, dtype=torch.int64),
                l2_host_hit_length=0,
                l3_storage_hit_length=0,
                last_device_node=req.last_node,
            )
        )
        queue._pre_alloc = MagicMock(side_effect=_Admitted)
        queue.transfer_queue = MagicMock(queue=[], enable_staging=False)
        queue.tree_cache = MagicMock()
        queue.tree_cache.dec_lock_ref = MagicMock()
        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 1
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1
        queue.token_to_kv_pool = MagicMock()
        queue.token_to_kv_pool_allocator = MagicMock()
        queue.token_to_kv_pool_allocator.page_size = 4

        running_batch = MagicMock()
        running_batch.reqs = []
        server_args = MagicMock()
        server_args.disaggregation_decode_enable_radix_cache = True
        scheduler = MagicMock()
        scheduler.running_batch = running_batch
        scheduler.server_args = server_args
        scheduler.enable_hisparse = False
        scheduler.enable_decode_hicache = False
        scheduler.waiting_queue = []
        scheduler.last_batch = None
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        queue._allocatable_token_budgets = MagicMock(side_effect=[8, 3, 64])

        with self.assertRaises(_Admitted):
            queue.pop_preallocated()

        # The prefix was dropped before the retry, and the retry read the pool
        # again rather than reusing the locked-pool number.
        queue.tree_cache.dec_lock_ref.assert_called_once()
        self.assertEqual(queue._allocatable_token_budgets.call_count, 3)
        self.assertEqual(queue._pre_alloc.call_args.args[2], 0)

    def test_miss_retry_admits_without_keeping_the_match_it_released(self):
        """The load-bearing case for clearing the match in `_release_prefix_lock`.

        A request matches a non-empty prefix, fails the FULL admission check
        against the post-lock budget, is retried as a miss, and IS admitted.
        The lock it took at match time was handed back, so the request must also
        stop advertising that match: `req.last_node` is the pipeline's record of
        "still holding a lock on this node", and `cache_finished_req` releases
        it a second time at completion (`UnifiedRadixCache._dec_req_lock`). Leaving
        the released node there spends that release on a lock the request no
        longer owns: `FullComponent.release_component_lock` asserts
        `lock_ref > 0` walking node->root, so it either kills the scheduler or
        silently consumes an ancestor lock a concurrent request holds -- whose
        FULL pages then become evictable while it is still reading them.

        The request must therefore end up in the state a MISS produces, which is
        `last_node` at the ROOT: a missed `match_prefix` returns the root, and
        lock/release of the root are no-ops that cancel. `None` is a different
        state -- it re-opens the batch-assembly re-match, which does not lock --
        and only relocates the unbalanced release; see
        `test_swa_decode_radix_cache.py::...::test_released_prefix_survives_the_first_cache_write`.

        Drives the real `_match_prefix_and_lock` / `_release_prefix_lock`
        against a real `RadixCache`, so the lock accounting is the tree's own.
        """
        cache, _ = _make_cache_with_pools()
        prefix = [1, 2, 3, 4]
        self._populate_prefix(cache, prefix, [10, 20, 30, 40])
        prefix_node = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", prefix)))
        ).last_device_node

        calls = {"inc": 0, "dec": 0}
        real_inc, real_dec = cache.inc_lock_ref, cache.dec_lock_ref

        def counting_inc(node, **kwargs):
            calls["inc"] += 1
            return real_inc(node, **kwargs)

        def counting_dec(node, params=None, **kwargs):
            calls["dec"] += 1
            return real_dec(node, params, **kwargs)

        cache.inc_lock_ref = counting_inc
        cache.dec_lock_ref = counting_dec

        req = _real_req(prefix + [5, 6, 7, 8])
        # Budgets: 8 before the loop, 3 once the match locked the prefix pages
        # (rejects the hit: 8 - 4 + 16 new tokens), 64 after the release handed
        # them back (admits the miss: 8 + 16).
        queue = _make_prealloc_queue(
            decode_req=_decode_request(req), tree_cache=cache, budgets=[8, 3, 64]
        )
        queue._pre_alloc = MagicMock(side_effect=_Admitted)

        with self.assertRaises(_Admitted):
            queue.pop_preallocated()

        # Admitted as a miss...
        self.assertEqual(queue._pre_alloc.call_args.args[2], 0, "prefix_len")
        self.assertEqual(queue._pre_alloc.call_args.args[3], 0, "total_prefix_len")
        # ...with the acquire fully undone: one lock taken, one given back, and
        # nothing left claiming the request still holds it.
        self.assertEqual(calls["inc"], 1)
        self.assertEqual(calls["dec"], calls["inc"])
        self.assertIs(req.last_node, cache.root_node_handle(req.extra_key))
        self.assertEqual(len(req.prefix_indices), 0)
        self.assertEqual(req.num_matched_prefix_tokens, 0)
        self.assertEqual(req.cache_protected_len, 0)
        # The tree agrees: the prefix is evictable again, which is what made the
        # miss fit in the first place.
        self.assertEqual(prefix_node.lock_ref, 0)
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(cache.evictable_size(), len(prefix))

    def test_release_prefix_lock_restores_the_never_matched_state(self):
        """`_release_prefix_lock` must undo every field `_match_prefix_and_lock`
        wrote, not just the lock.

        The field set is not this function's to choose: `match_prefix_for_req`
        populates `prefix_indices` / `last_node` / `num_matched_prefix_tokens` /
        `cache_protected_len`, and `Req.reset_for_retract` is the other place
        that has to put a request back to "never matched". Cross-checked against
        it field by field so the two cannot drift apart -- a new match field
        added to one and not the other is the failure mode.

        Two fields are checked separately rather than against
        `reset_for_retract`:

        * `last_node` -- its correct value depends on which pipeline picks the
          request up next, and the decode-prealloc pipeline is not the retract
          pipeline. `reset_for_retract`'s `None` is right for a request that
          re-enters through `resume_retracted_reqs`; a request released here is
          admitted immediately and must look like a MISS, i.e. hold the ROOT, so
          batch assembly does not re-match it into a node it never locked. See
          `test_swa_decode_radix_cache.py::...::test_released_prefix_survives_the_first_cache_write`.
        * `swa_prefix_lock_released` -- `reset_for_retract` clears it because
          retraction unwinds a *decode-time* SWA release, which cannot have
          happened yet for a request still in the prealloc queue.
        """
        matched_fields = (
            "num_matched_prefix_tokens",
            "cache_protected_len",
            "swa_uuid_for_lock",
            "skip_lock_node_ids",
        )

        def matched_req(rid):
            req = _real_req([1, 2, 3, 4, 5, 6, 7, 8], rid=rid)
            req.prefix_indices = torch.arange(4, dtype=torch.int64)
            req.last_node = node
            req.num_matched_prefix_tokens = 4
            req.cache_protected_len = 4
            req.swa_uuid_for_lock = 17
            req.skip_lock_node_ids = {ComponentType.SWA: {41}}
            return req

        node = object()
        released, retracted = matched_req("released"), matched_req("retracted")

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.tree_cache = MagicMock()
        queue._release_prefix_lock(released)

        # The release replays exactly the token the acquire recorded.
        queue.tree_cache.dec_lock_ref.assert_called_once_with(
            node,
            DecLockRefParams(
                swa_uuid_for_lock=17, skip_lock_node_ids={ComponentType.SWA: {41}}
            ),
        )

        # Back to the miss state: the root handle, scoped by the request's own
        # namespace (a per-extra_key tree has its own root).
        self.assertIs(
            released.last_node, queue.tree_cache.root_node_handle.return_value
        )
        queue.tree_cache.root_node_handle.assert_called_once_with(released.extra_key)

        retracted.kv = None  # reset_for_retract asserts the KV is already freed
        retracted.reset_for_retract()
        for field in matched_fields:
            self.assertEqual(
                getattr(released, field),
                getattr(retracted, field),
                f"{field} disagrees with Req.reset_for_retract",
            )
        self.assertEqual(len(released.prefix_indices), 0)
        self.assertEqual(len(retracted.prefix_indices), 0)

    def test_prefix_reaching_into_the_swa_window_is_admitted_as_a_miss(self):
        """A hit the SWA tail allocator cannot express must become a miss.

        Production geometry (page 256, window 128) with a prefix long enough
        that the window starts *inside* it: `_swa_tail_len(1000) == 232 > 1000 -
        896`. `alloc_extend_swa_tail` maps the window onto the last slots of the
        delta it allocates, so it cannot cover in-window positions that belong
        to the reused prefix; the `alloc_extend` fallback allocates SWA pages for
        the delta only, leaving those positions with no SWA KV of their own --
        their `full_to_swa_index_mapping` entries read the padding slot 0.
        `pop_preallocated` therefore drops the prefix, and the admitted request
        goes back through the tail path it can express.
        """
        fill_len, hit_len = 1000, 896
        allocator = MagicMock(page_size=256, device=torch.device("cpu"))
        allocator.available_size.return_value = 10**9
        allocator.alloc_extend_swa_tail.return_value = torch.arange(
            fill_len, dtype=torch.int64
        )
        del allocator.c4_attn_allocator  # not the NPU allocator

        node = object()
        req = _real_req(list(range(fill_len)))
        req.last_node = node  # set by the match that `match_prefix_for_req` runs
        tree_cache = MagicMock()
        tree_cache.inc_lock_ref.return_value = IncLockRefResult()

        queue = _make_prealloc_queue(
            decode_req=_decode_request(req),
            tree_cache=tree_cache,
            budgets=[10**9] * 4,
            page_size=256,
            allocator=allocator,
            uses_swa_tail_prealloc=True,
            sliding_window_size=128,
        )
        # Precondition: this is the geometry the tail path cannot express.
        self.assertEqual(queue._swa_tail_len(fill_len), 232)
        self.assertFalse(
            queue._takes_swa_tail_path(fill_len=fill_len, total_prefix_len=hit_len)
        )

        captured = {}
        real_pre_alloc = DecodePreallocQueue._pre_alloc

        def spy_pre_alloc(req_, prefix_indices, prefix_len, total_prefix_len):
            captured.update(prefix_len=prefix_len, total_prefix_len=total_prefix_len)
            real_pre_alloc(queue, req_, prefix_indices, prefix_len, total_prefix_len)
            raise _Admitted

        queue._pre_alloc = spy_pre_alloc

        match_result = SimpleNamespace(
            device_indices=torch.arange(hit_len, dtype=torch.int64),
            host_hit_length=0,
            last_device_node=node,
            last_host_node=None,
        )
        with patch(
            "sglang.srt.disaggregation.decode.match_prefix_for_req",
            return_value=match_result,
        ), self.assertRaises(_Admitted):
            queue.pop_preallocated()

        # Admitted as a miss, with the prefix lock handed back.
        self.assertEqual(captured, {"prefix_len": 0, "total_prefix_len": 0})
        self.assertEqual(tree_cache.inc_lock_ref.call_count, 1)
        self.assertEqual(
            tree_cache.dec_lock_ref.call_count, tree_cache.inc_lock_ref.call_count
        )
        # Left in the MISS state (root handle), not `None`: see
        # `test_release_prefix_lock_restores_the_never_matched_state`.
        self.assertIs(req.last_node, tree_cache.root_node_handle.return_value)
        tree_cache.root_node_handle.assert_called_once_with(req.extra_key)

        # ...and the allocation went back through the tail-only entry point.
        allocator.alloc_extend.assert_not_called()
        allocator.alloc_extend_swa_tail.assert_called_once()
        kwargs = allocator.alloc_extend_swa_tail.call_args.kwargs
        self.assertEqual(int(kwargs["prefix_lens_cpu"][0]), 0)
        self.assertEqual(int(kwargs["seq_lens_cpu"][0]), fill_len)
        self.assertEqual(kwargs["extend_num_tokens"], fill_len)
        self.assertEqual(kwargs["swa_tail_len"], 232)

    def test_alloc_extend_fallback_rejects_a_prefix_on_a_swa_tail_allocator(self):
        """The backstop for the branch above, at the allocator boundary.

        `alloc_extend` splits the SWA debit over the delta only, so on a hybrid
        pool it silently produces a sequence whose in-window prefix positions
        have no SWA KV. Nothing downstream can detect that -- SWA attention just
        reads slot 0 -- so the state is refused where it is constructible rather
        than where it is observable. Plain (non-SWA) allocators legitimately take
        this branch with a prefix and must stay unaffected.
        """
        fill_len, prefix_len, delta_len = 1000, 896, 104

        def call(allocator):
            return alloc_for_decode_prealloc(
                allocator,
                req=SimpleNamespace(kv=None, rid="req-1"),
                fill_len=fill_len,
                delta_len=delta_len,
                prefix_len=prefix_len,
                total_prefix_len=prefix_len,
                prefix_indices=torch.arange(prefix_len, dtype=torch.int64),
                uses_swa_tail=False,
                swa_tail_len=232,
            )

        swa_allocator = MagicMock(page_size=256, device=torch.device("cpu"))
        del swa_allocator.c4_attn_allocator
        with self.assertRaises(AssertionError) as ctx:
            call(swa_allocator)
        self.assertIn("total_prefix_len=896", str(ctx.exception))
        swa_allocator.alloc_extend.assert_not_called()

        plain_allocator = MagicMock(page_size=256, device=torch.device("cpu"))
        del plain_allocator.c4_attn_allocator
        del plain_allocator.alloc_extend_swa_tail
        plain_allocator.alloc_extend.return_value = torch.arange(
            delta_len, dtype=torch.int64
        )
        call(plain_allocator)
        plain_allocator.alloc_extend.assert_called_once()

    def test_repeated_incremental_no_leak(self):
        """Multiple incremental transfers shouldn't leak lock_refs."""
        cache, req_to_token = _make_cache_with_pools()

        prefix = [1, 2, 3]
        prefix_vals = [10, 20, 30]
        self._populate_prefix(cache, prefix, prefix_vals)

        for iteration in range(5):
            result = cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", prefix)))
            )
            matched_node = result.last_device_node
            prefix_len = len(result.device_indices)

            cache.inc_lock_ref(matched_node)

            suffix_token = 40 + iteration
            full_ids = prefix + [suffix_token]
            full_vals = prefix_vals + [100 + iteration]
            req_to_token[0, : len(full_vals)] = torch.tensor(
                full_vals, dtype=torch.int64
            )

            req = _make_req(
                fill_ids=full_ids,
                req_pool_idx=0,
                cache_protected_len=prefix_len,
                last_node=matched_node,
            )

            cache.cache_unfinished_req(req)
            cache.cache_finished_req(req, kv_len_to_handle=req.kv_committed_len)

        # After all iterations, root lock should be 1, no protected nodes
        self.assertEqual(cache.root_node.lock_ref, 1)
        self.assertEqual(cache.protected_size(), 0)


if __name__ == "__main__":
    unittest.main()
