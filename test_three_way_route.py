"""
Minimal unit tests for the three-way OT stable-anchor / boundary-info
dual-channel query closed loop.

Plain-assertion script (no pytest dependency, matches this repo's style).
Run with: python test_three_way_route.py
"""
import numpy as np
import torch

from ncd_ot import compute_density_score, QAP2OTSolver
from ncd_train_agcd import (
    make_route_buffers,
    route_accumulate_batch,
    route_finalize_scores,
    parse_anchor_ratio_schedule,
    anchor_ratio_for_round,
)
from ncd_strategies import TransportRouteSampling


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        raise AssertionError(name)


class _FakeTeacher(torch.nn.Module):
    def forward(self, x):
        return x, None


class _FakeActiveDatasetForOT:
    def __init__(self, n):
        self.labeled_mask = np.zeros(n, dtype=bool)


def test_independent_query_active_value():
    """The active-query value Q_i must be computed independently of the OT's
    query/reject columns. The current novel-only protocol must not circularly
    suppress currently known-predicted samples; a genuinely mixed pool must
    compute novelty from one full known-vs-novel softmax."""
    num_known, num_novel = 3, 2
    K = num_known + num_novel
    B = 8
    solver = QAP2OTSolver(num_known=num_known, num_novel=num_novel, n_iter=20,
                           query_active_lambda=0.5)

    torch.manual_seed(0)
    logits2B = torch.randn(2 * B, K) * 0.5
    # sample 0 (and its paired view 8): confidently a KNOWN class.
    logits2B[0] = torch.tensor([5., 0., 0., 0., 0.])
    logits2B[8] = torch.tensor([5., 0., 0., 0., 0.])
    # sample 1 (and its paired view 9): confidently a NOVEL class.
    logits2B[1] = torch.tensor([0., 0., 0., 5., 0.])
    logits2B[9] = torch.tensor([0., 0., 0., 5., 0.])

    feats2B = torch.randn(2 * B, 16)
    all_targets = [0] * 20

    _, _, routing = solver.get_targets(
        student_logits=logits2B, teacher_model=_FakeTeacher(), u_inputs=logits2B,
        active_dataset=_FakeActiveDatasetForOT(20), all_targets=all_targets, device='cpu',
        round_idx=1, student_features=feats2B, return_reliability=True, return_routing=True,
    )
    check("novel-only pool uses novelty=1 for every candidate",
          bool((routing['novelty'] == 1.0).all()))
    check("query_active_value is returned with the right shape", routing['query_active_value'].shape == (16,))
    check(
        "query_active_value is finite and non-negative (all factors are in [0,1])",
        bool(torch.isfinite(routing['query_active_value']).all())
        and bool((routing['query_active_value'] >= 0).all()),
    )

    mixed_solver = QAP2OTSolver(
        num_known=num_known, num_novel=num_novel, n_iter=20,
        query_active_lambda=0.5, novel_only_unlabeled=False,
    )
    _, _, mixed_routing = mixed_solver.get_targets(
        student_logits=logits2B, teacher_model=_FakeTeacher(), u_inputs=logits2B,
        active_dataset=_FakeActiveDatasetForOT(20), all_targets=all_targets, device='cpu',
        round_idx=1, student_features=feats2B, return_reliability=True,
        return_routing=True,
    )
    check(
        "mixed-pool novelty uses full novel-head mass",
        mixed_routing['novelty'][1].item() > mixed_routing['novelty'][0].item(),
    )
    # lambda=0 -> query_active_value should not depend on expert disagreement D_i
    # (only on novelty/density/aug_consistency/prediction-uncertainty).
    solver_lam0 = QAP2OTSolver(num_known=num_known, num_novel=num_novel, n_iter=20,
                                query_active_lambda=0.0)
    _, _, routing_lam0 = solver_lam0.get_targets(
        student_logits=logits2B, teacher_model=_FakeTeacher(), u_inputs=logits2B,
        active_dataset=_FakeActiveDatasetForOT(20), all_targets=all_targets, device='cpu',
        round_idx=1, student_features=feats2B, return_reliability=True, return_routing=True,
    )
    # With no expert_logits, D_i (expert disagreement) is 0 everywhere, so at
    # lambda=1 query_active_value collapses to 0 (pure D_i term), and at
    # lambda=0 it should generally differ (driven by prediction uncertainty).
    solver_lam1 = QAP2OTSolver(num_known=num_known, num_novel=num_novel, n_iter=20,
                                query_active_lambda=1.0)
    _, _, routing_lam1 = solver_lam1.get_targets(
        student_logits=logits2B, teacher_model=_FakeTeacher(), u_inputs=logits2B,
        active_dataset=_FakeActiveDatasetForOT(20), all_targets=all_targets, device='cpu',
        round_idx=1, student_features=feats2B, return_reliability=True, return_routing=True,
    )
    check(
        "lambda=1 with no expert_logits (D_i=0 everywhere) zeroes out query_active_value",
        bool((routing_lam1['query_active_value'] == 0).all()),
    )
    check(
        "lambda=0 uses prediction uncertainty instead, generally nonzero",
        routing_lam0['query_active_value'].sum().item() > 0,
    )


def test_two_way_transport_routing_identity():
    """Two-way OT must expose one real-class route and one deferred route.
    TransportRouteSampling relies on P_i + H_i = 1, not on a query column."""
    num_known, num_novel = 3, 2
    B, K = 8, 5
    torch.manual_seed(7)
    logits2B = torch.randn(2 * B, K)
    feats2B = torch.randn(2 * B, 12)
    solver = QAP2OTSolver(
        num_known=num_known,
        num_novel=num_novel,
        rho_novel=0.75,
        n_iter=100,
        use_three_way_transport=False,
        novel_only_unlabeled=True,
    )
    _, _, routing = solver.get_targets(
        student_logits=logits2B,
        teacher_model=_FakeTeacher(),
        u_inputs=logits2B,
        active_dataset=_FakeActiveDatasetForOT(20),
        all_targets=[0] * 20,
        device='cpu',
        round_idx=1,
        student_features=feats2B,
        return_reliability=True,
        return_routing=True,
    )
    check("two-way transport has no query-column probability",
          bool((routing['query_prob'] == 0.0).all()))
    check("two-way pseudo_prob + defer_prob satisfies the row marginal",
          bool(torch.allclose(
              routing['pseudo_prob'] + routing['defer_prob'],
              torch.ones_like(routing['pseudo_prob']), atol=1e-5, rtol=1e-5)))
    check("legacy reject_prob aliases defer_prob in two-way mode",
          bool(torch.allclose(routing['reject_prob'], routing['defer_prob'], atol=2e-6)))
    check("two-way mean deferred probability follows 1-rho_novel",
          abs(routing['defer_prob'].mean().item() - 0.25) < 5e-3)


def test_paired_view_density_masking():
    anchors = torch.tensor([
        [0.0, 0.0],
        [10.0, 0.0],
        [0.0, 10.0],
        [10.0, 10.0],
    ])
    feats = torch.cat([anchors, anchors], dim=0)  # [8, 2] = [view1(4), view2(4)]

    density_unmasked = compute_density_score(feats, k=1, paired_views=False)
    density_masked = compute_density_score(feats, k=1, paired_views=True)

    check(
        "unmasked paired density is spuriously huge (self-view picked as NN)",
        density_unmasked.min().item() > 1e5,
    )
    check(
        "masked paired density stays orders of magnitude below the unmasked spike",
        density_masked.max().item() < 100.0,
    )
    check(
        "masking strictly reduces density for every sample",
        bool((density_masked < density_unmasked).all()),
    )

    odd_feats = torch.randn(7, 2)
    out = compute_density_score(odd_feats, k=1, paired_views=True)
    check("odd batch size with paired_views=True doesn't crash", out.shape[0] == 7)


def test_route_buffer_accumulation():
    pool_size = 10
    buffers = make_route_buffers(pool_size)

    # Batch 1: samples [2, 5, 7], bs=3 -> each signal is [2*bs]=[6],
    # stacked as [view1(3), view2(3)].
    u_idxs_1 = torch.tensor([2, 5, 7])
    p_1 = torch.tensor([0.90, 0.80, 0.70, 0.70, 0.80, 0.90])
    q_1 = torch.tensor([0.10, 0.20, 0.30, 0.30, 0.20, 0.10])
    r_1 = torch.tensor([0.01, 0.02, 0.03, 0.03, 0.02, 0.01])
    qval_1 = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    rval_1 = torch.tensor([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
    maxy_1 = torch.tensor([0.95, 0.85, 0.75, 0.75, 0.85, 0.95])
    nov_1 = torch.tensor([0.6, 0.7, 0.8, 0.8, 0.7, 0.6])
    qav_1 = torch.tensor([0.1, 0.3, 0.5, 0.9, 0.2, 0.4])
    route_accumulate_batch(buffers, u_idxs_1, p_1, q_1, r_1, qval_1, rval_1, maxy_1, nov_1, qav_1)

    expected_q = {2: 0.5 * (0.10 + 0.30), 5: 0.5 * (0.20 + 0.20), 7: 0.5 * (0.30 + 0.10)}
    for idx, val in expected_q.items():
        check(
            f"query_sum[{idx}] == view-averaged query_prob after 1 batch",
            abs(buffers['query_sum'][idx].item() - val) < 1e-6,
        )
    expected_qav = {2: 0.5 * (0.1 + 0.9), 5: 0.5 * (0.3 + 0.2), 7: 0.5 * (0.5 + 0.4)}
    for idx, val in expected_qav.items():
        check(
            f"query_active_value_sum[{idx}] == view-averaged query_active_value after 1 batch",
            abs(buffers['query_active_value_sum'][idx].item() - val) < 1e-6,
        )
    check("count[2]==1 after first batch", buffers['count'][2].item() == 1.0)
    check("untouched index 0 has count 0", buffers['count'][0].item() == 0.0)

    # Batch 2: index 2 seen again -> sums should accumulate, not overwrite.
    u_idxs_2 = torch.tensor([2, 9])
    p_2 = torch.tensor([0.5, 0.5, 0.5, 0.5])
    q_2 = torch.tensor([0.50, 0.50, 0.90, 0.90])
    r_2 = torch.tensor([0.05, 0.05, 0.09, 0.09])
    qval_2 = torch.tensor([5.0, 5.0, 9.0, 9.0])
    rval_2 = torch.tensor([0.5, 0.5, 0.9, 0.9])
    maxy_2 = torch.tensor([0.6, 0.6, 0.6, 0.6])
    nov_2 = torch.tensor([0.9, 0.9, 0.2, 0.2])
    qav_2 = torch.tensor([0.8, 0.8, 0.1, 0.1])
    route_accumulate_batch(buffers, u_idxs_2, p_2, q_2, r_2, qval_2, rval_2, maxy_2, nov_2, qav_2)

    check("count[2]==2 after being seen twice", buffers['count'][2].item() == 2.0)
    # batch2 view1[:bs]=[0.50(idx2),0.50(idx9)], view2[bs:]=[0.90(idx2),0.90(idx9)]
    expected_sum_idx2 = 0.5 * (0.10 + 0.30) + 0.5 * (0.50 + 0.90)
    check(
        "query_sum[2] accumulates across batches (not overwritten)",
        abs(buffers['query_sum'][2].item() - expected_sum_idx2) < 1e-6,
    )
    check("index 9 (only in batch 2) has count 1", buffers['count'][9].item() == 1.0)
    check("index 1 (never seen) still has count 0", buffers['count'][1].item() == 0.0)

    # ── round >= 1: anchor_score = P_i * max_k(y_ik);
    # H_i = 1 - P_i; info_score = H_i * query_active_value ──
    anchor_score, info_score, debug = route_finalize_scores(
        buffers, round_idx=1, route_round0_fallback='query_value',
        unlabeled_count=pool_size,
    )
    p_avg_idx2 = buffers['pseudo_sum'][2] / 2.0
    maxy_avg_idx2 = buffers['max_soft_sum'][2] / 2.0
    qav_avg_idx2 = buffers['query_active_value_sum'][2] / 2.0
    expected_h_idx2 = 1.0 - p_avg_idx2
    expected_info_idx2 = expected_h_idx2 * qav_avg_idx2
    expected_anchor_idx2 = p_avg_idx2 * maxy_avg_idx2
    check(
        "round>=1 H_i == 1 - pseudo_prob (mode-independent row marginal)",
        abs(debug['h_score'][2].item() - expected_h_idx2.item()) < 1e-6,
    )
    check(
        "round>=1 info_score == H_i * query_active_value",
        abs(info_score[2].item() - expected_info_idx2.item()) < 1e-6,
    )
    check(
        "anchor_score == pseudo_prob * max_soft_target (no redundant (1-R) term)",
        abs(anchor_score[2].item() - expected_anchor_idx2.item()) < 1e-6,
    )
    check("debug.fallback_mode == 'h_gated' for round>=1", debug['fallback_mode'] == 'h_gated')

    # Uncovered indices must fall back to the covered-mean, for both channels.
    covered_mask = buffers['count'] > 0
    anchor_covered_mean = anchor_score[covered_mask].mean().item()
    info_covered_mean = info_score[covered_mask].mean().item()
    for idx in [0, 1, 3, 4, 6, 8]:
        check(
            f"uncovered index {idx} anchor_score falls back to covered-mean, not 0",
            abs(anchor_score[idx].item() - anchor_covered_mean) < 1e-6,
        )
        check(
            f"uncovered index {idx} info_score falls back to covered-mean, not 0",
            abs(info_score[idx].item() - info_covered_mean) < 1e-6,
        )
    check(
        "debug reports the correct uncovered count",
        debug['uncovered_count'] == int((~covered_mask).sum().item()),
    )

    # ── round 0 with the default 'query_value' fallback: H_i is near-zero
    # (rho_novel=1.0 forced at round 0) and uninformative, so info_score
    # should be the raw query_active_value alone, ungated ──
    _, info_score_r0, debug_r0 = route_finalize_scores(
        buffers, round_idx=0, route_round0_fallback='query_value',
        unlabeled_count=pool_size,
    )
    qav_avg_idx5 = buffers['query_active_value_sum'][5] / buffers['count'][5]
    check(
        "round0 query_value fallback uses raw query_active_value, not H-gated",
        abs(info_score_r0[5].item() - qav_avg_idx5.item()) < 1e-6,
    )
    check("debug.fallback_mode == 'query_value' for round0 default", debug_r0['fallback_mode'] == 'query_value')

    # round 0 with 'query_pass': buffers already hold the query_pass routing's
    # real (non-degenerate) H_i (accumulate_batch doesn't know the
    # difference), so info_score should behave like the round>=1 formula.
    _, info_score_pass, debug_pass = route_finalize_scores(
        buffers, round_idx=0, route_round0_fallback='query_pass',
        unlabeled_count=pool_size,
    )
    check(
        "round0 query_pass fallback uses the H-gated formula",
        abs(info_score_pass[2].item() - expected_info_idx2.item()) < 1e-6,
    )
    check("debug.fallback_mode == 'query_pass'", debug_pass['fallback_mode'] == 'query_pass')

    # Empty buffers must not crash and must report 0 coverage.
    empty_buffers = make_route_buffers(pool_size)
    empty_anchor, empty_info, empty_debug = route_finalize_scores(
        empty_buffers, round_idx=1, route_round0_fallback='query_value',
        unlabeled_count=pool_size,
    )
    check("empty buffers -> zero coverage", empty_debug['coverage_ratio'] == 0.0)
    check("empty buffers -> all-zero anchor/info score", bool((empty_anchor == 0.0).all()) and bool((empty_info == 0.0).all()))


def test_anchor_ratio_schedule():
    default = parse_anchor_ratio_schedule(None)
    check("default schedule matches spec (0.70..0.15)", default == (0.70, 0.50, 0.30, 0.20, 0.15))
    check("empty string falls back to default", parse_anchor_ratio_schedule("") == default)

    custom = parse_anchor_ratio_schedule("0.9,0.1")
    check("custom schedule parses to floats", custom == (0.9, 0.1))

    check("round 0 uses schedule[0]", anchor_ratio_for_round(custom, 0) == 0.9)
    check("round 1 uses schedule[1]", anchor_ratio_for_round(custom, 1) == 0.1)
    check("round beyond schedule length clamps to last value", anchor_ratio_for_round(custom, 5) == 0.1)
    check("negative round clamps to first value", anchor_ratio_for_round(custom, -3) == 0.9)

    try:
        parse_anchor_ratio_schedule("0.5,1.5")
        check("out-of-range ratio raises ValueError", False)
    except ValueError:
        check("out-of-range ratio raises ValueError", True)


class _FakeDataset:
    def __init__(self, labeled_mask):
        self.labeled_mask = labeled_mask


class _FakeArgs:
    num_known = 3
    num_workers = 0
    batch_size = 4


def test_transport_route_sampling_dual_channel():
    pool_size = 8  # 2 predicted novel clusters (class 3 and class 4), 4 each
    labeled_mask = np.zeros(pool_size, dtype=bool)
    strategy = TransportRouteSampling(_FakeDataset(labeled_mask), net=None, args=_FakeArgs(), device='cpu')

    preds_target = torch.tensor([3, 3, 3, 3, 4, 4, 4, 4])
    num_classes = 5  # num_known=3 + num_novel=2
    fake_logits = torch.zeros(pool_size, num_classes)
    for i, c in enumerate(preds_target):
        fake_logits[i, c] = 10.0
    strategy.get_logits = lambda idxs: fake_logits[idxs]

    anchor_score = torch.zeros(pool_size)
    info_score = torch.zeros(pool_size)
    # cluster [0,1,2,3] (class 3): 0/1 are clear anchors, 2/3 are clear info picks.
    anchor_score[0], anchor_score[1], anchor_score[2], anchor_score[3] = 5.0, 4.0, 0.1, 0.1
    info_score[0], info_score[1], info_score[2], info_score[3] = 0.1, 0.1, 5.0, 4.0
    # cluster [4,5,6,7] (class 4): same pattern.
    anchor_score[4], anchor_score[5], anchor_score[6], anchor_score[7] = 5.0, 4.0, 0.1, 0.1
    info_score[4], info_score[5], info_score[6], info_score[7] = 0.1, 0.1, 5.0, 4.0

    # n=4 across 2 clusters -> num_per_class=2, ratio=0.5 -> 1 anchor + 1 info each.
    result = strategy.query(4, current_round=0, anchor_score=anchor_score,
                             info_score=info_score, anchor_ratio_schedule=(0.5,))
    result_set = set(int(i) for i in result)
    check(
        "ratio=0.5 picks top-1-anchor + top-1-info per cluster",
        result_set == {0, 2, 4, 6},
    )

    # ratio=1.0 -> all anchor, no info picks.
    result_anchor_only = strategy.query(4, current_round=0, anchor_score=anchor_score,
                                         info_score=info_score, anchor_ratio_schedule=(1.0,))
    check(
        "ratio=1.0 picks only top anchors per cluster",
        set(int(i) for i in result_anchor_only) == {0, 1, 4, 5},
    )

    # ratio=0.0 -> all info, no anchor picks.
    result_info_only = strategy.query(4, current_round=0, anchor_score=anchor_score,
                                       info_score=info_score, anchor_ratio_schedule=(0.0,))
    check(
        "ratio=0.0 picks only top info samples per cluster",
        set(int(i) for i in result_info_only) == {2, 3, 6, 7},
    )

    # Dedup + global topup: make idx0 dominate BOTH channels in its cluster so
    # the anchor pick and info pick collide, forcing a global topup to reach n.
    anchor_dom = torch.zeros(pool_size)
    info_dom = torch.zeros(pool_size)
    anchor_dom[0], info_dom[0] = 9.0, 9.0   # idx0 wins both channels
    anchor_dom[1], info_dom[1] = 8.0, 8.0   # idx1 is the next-best on both
    anchor_dom[4], info_dom[4] = 1.0, 1.0
    anchor_dom[5], info_dom[5] = 0.5, 0.5
    result_dedup = strategy.query(4, current_round=0, anchor_score=anchor_dom,
                                   info_score=info_dom, anchor_ratio_schedule=(0.5,))
    check("dedup + global topup still returns exactly n samples", len(result_dedup) == 4)
    check("dominant sample idx0 is included", 0 in set(int(i) for i in result_dedup))

    # Missing anchor_score/info_score must raise, not silently misbehave.
    try:
        strategy.query(4, current_round=0)
        check("missing anchor_score/info_score raises ValueError", False)
    except ValueError:
        check("missing anchor_score/info_score raises ValueError", True)

    check("n=0 returns empty array", len(strategy.query(0, current_round=0,
                                                          anchor_score=anchor_score, info_score=info_score)) == 0)


def test_transport_route_sampling_noncontiguous_topup_and_known_discovery():
    """Regression test for two production-only failures hidden by an all-unlabeled toy pool.

    Absolute dataset indices differ from positions in the unlabeled subset, and
    sample 2 is globally predicted as known while its best novel head is class 3.
    It must remain queryable through the information/discovery channel.
    """
    pool_size = 10
    labeled_mask = np.ones(pool_size, dtype=bool)
    unlabeled = np.array([2, 5, 6, 8, 9])
    labeled_mask[unlabeled] = False
    strategy = TransportRouteSampling(
        _FakeDataset(labeled_mask), net=None, args=_FakeArgs(), device='cpu'
    )

    fake_logits = torch.zeros(pool_size, 5)
    # idx2: global known prediction (class 0), but tentative novel assignment 3.
    fake_logits[2, 0] = 10.0
    fake_logits[2, 3] = 9.0
    # Remaining candidates belong to tentative novel cluster 4.
    for idx in [5, 6, 8, 9]:
        fake_logits[idx, 4] = 10.0
    strategy.get_logits = lambda idxs: fake_logits[idxs]

    anchor_score = torch.zeros(pool_size)
    info_score = torch.zeros(pool_size)
    info_score[2] = 50.0
    info_score[5] = 100.0
    info_score[6] = 90.0
    info_score[8] = 1.0
    info_score[9] = 80.0
    # These pool-position scores deliberately prefer idx8 under the old bug:
    # remaining positions [3,4] were used instead of absolute indices [8,9].
    info_score[3] = 10.0
    info_score[4] = 0.0

    result = strategy.query(
        4,
        current_round=0,
        anchor_score=anchor_score,
        info_score=info_score,
        anchor_ratio_schedule=(0.0,),
    )
    result_set = set(int(i) for i in result)
    check(
        "globally known-predicted sample remains queryable via tentative novel assignment",
        2 in result_set,
    )
    check(
        "global top-up indexes route scores by absolute dataset index",
        9 in result_set and 8 not in result_set,
    )


if __name__ == '__main__':
    test_independent_query_active_value()
    test_two_way_transport_routing_identity()
    test_paired_view_density_masking()
    test_route_buffer_accumulation()
    test_anchor_ratio_schedule()
    test_transport_route_sampling_dual_channel()
    test_transport_route_sampling_noncontiguous_topup_and_known_discovery()
    print("\nALL TESTS PASSED")
