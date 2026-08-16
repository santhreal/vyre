//! Contracts for `vyre_driver::benchmark_pass_selection`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::benchmark_pass_selection::{
    select_benchmark_passes, select_benchmark_passes_with_scratch, BenchmarkPassCandidate,
    BenchmarkPassSelectionError, BenchmarkPassSelectionSample, BenchmarkPassSelectionScratch,
    BenchmarkPassSkipReason, SkippedBenchmarkPass,
};

#[test]
fn benchmark_pass_selection_picks_profitable_passes_by_value() {
    let plan = select_benchmark_passes(
        &[
            candidate(
                "device.adjacent-launch-fusion",
                1_000,
                4,
                0,
                100,
                64,
                18_000,
                true,
            ),
            candidate(
                "device.result-compaction",
                1,
                1,
                4_096,
                20,
                16,
                12_000,
                false,
            ),
            candidate(
                "device.megakernel-plan-cache",
                1,
                64,
                0,
                50,
                32,
                25_000,
                true,
            ),
        ],
        BenchmarkPassSelectionSample {
            frontier_items: 2_000,
            reuse_count: 128,
            avoidable_readback_bytes: 8_192,
            planning_budget_ns: 200,
            scratch_budget_bytes: 128,
        },
    )
    .expect("Fix: profitable passes should select");

    assert_eq!(plan.selected_pass_ids.len(), 3);
    assert!(plan
        .selected_pass_ids
        .contains(&"device.megakernel-plan-cache"));
    assert!(plan
        .selected_pass_ids
        .contains(&"device.adjacent-launch-fusion"));
    assert!(plan.selected_pass_ids.contains(&"device.result-compaction"));
    assert_eq!(plan.total_planning_cost_ns, 170);
    assert_eq!(plan.total_scratch_bytes, 112);
    assert!(plan.projected_speedup_bps > 50_000);
}

#[test]
fn benchmark_pass_selection_skips_unprofitable_passes_with_stable_reasons() {
    let plan = select_benchmark_passes(
        &[
            candidate(
                "device.adjacent-launch-fusion",
                1_000,
                4,
                0,
                10,
                8,
                15_000,
                false,
            ),
            candidate(
                "device.result-compaction",
                1,
                1,
                4_096,
                10,
                8,
                11_000,
                false,
            ),
        ],
        BenchmarkPassSelectionSample {
            frontier_items: 10,
            reuse_count: 1,
            avoidable_readback_bytes: 128,
            planning_budget_ns: 100,
            scratch_budget_bytes: 100,
        },
    )
    .expect("Fix: unprofitable optional passes should skip");

    assert_eq!(plan.selected_pass_ids, Vec::<&'static str>::new());
    assert_eq!(plan.skipped_passes.len(), 2);
    assert!(plan.skipped_passes.contains(&SkippedBenchmarkPass {
        pass_id: "device.adjacent-launch-fusion",
        reason: BenchmarkPassSkipReason::FrontierBelowThreshold,
    }));
    assert!(plan.skipped_passes.contains(&SkippedBenchmarkPass {
        pass_id: "device.result-compaction",
        reason: BenchmarkPassSkipReason::ReadbackBelowThreshold,
    }));
}

#[test]
fn benchmark_pass_selection_ranks_huge_values_without_saturation_ties() {
    let plan = select_benchmark_passes(
        &[
            candidate(
                "device.a-lexicographic-low-value",
                u64::MAX,
                u64::MAX,
                u64::MAX - 1,
                1,
                1,
                11_000,
                false,
            ),
            candidate(
                "device.z-lexicographic-high-value",
                u64::MAX,
                u64::MAX,
                u64::MAX,
                1,
                1,
                11_000,
                false,
            ),
        ],
        BenchmarkPassSelectionSample {
            frontier_items: u64::MAX,
            reuse_count: u64::MAX,
            avoidable_readback_bytes: u64::MAX,
            planning_budget_ns: 10,
            scratch_budget_bytes: 10,
        },
    )
    .expect("Fix: huge benchmark evidence should rank without saturating value ties");

    assert_eq!(
        plan.selected_pass_ids[0],
        "device.z-lexicographic-high-value",
        "Fix: pass ranking must use widened arithmetic; saturating u64 scoring would tie these candidates and incorrectly choose lexicographic order."
    );
}

#[test]
fn benchmark_pass_selection_rejects_missing_evidence_and_blocked_mandatory() {
    assert_eq!(
        select_benchmark_passes(
            &[candidate("device.bad", 1, 1, 0, 1, 1, 10_000, false)],
            sample(),
        )
        .expect_err("zero speedup evidence should fail"),
        BenchmarkPassSelectionError::MissingSpeedupEvidence {
            pass_id: "device.bad",
        }
    );
    assert_eq!(
        select_benchmark_passes(
            &[candidate("device.mandatory", 1, 1, 0, 101, 1, 11_000, true,)],
            sample(),
        )
        .expect_err("mandatory profitable pass cannot exceed budget"),
        BenchmarkPassSelectionError::MandatoryProfitablePassOverBudget {
            pass_id: "device.mandatory",
            reason: BenchmarkPassSkipReason::PlanningBudgetExceeded,
        }
    );
}

#[test]
fn benchmark_pass_selection_does_not_let_optional_passes_starve_mandatory_passes() {
    let plan = select_benchmark_passes(
        &[
            candidate(
                "device.optional-high-value",
                1,
                1,
                1_000_000,
                100,
                1,
                20_000,
                false,
            ),
            candidate("device.mandatory-low-value", 1, 1, 1, 100, 1, 11_000, true),
        ],
        BenchmarkPassSelectionSample {
            frontier_items: 1,
            reuse_count: 1,
            avoidable_readback_bytes: 1_000_000,
            planning_budget_ns: 100,
            scratch_budget_bytes: 8,
        },
    )
    .expect("Fix: mandatory profitable pass must reserve budget before optional passes");

    assert_eq!(plan.selected_pass_ids, vec!["device.mandatory-low-value"]);
    assert_eq!(
        plan.skipped_passes,
        vec![SkippedBenchmarkPass {
            pass_id: "device.optional-high-value",
            reason: BenchmarkPassSkipReason::PlanningBudgetExceeded,
        }]
    );
}

#[test]
fn benchmark_pass_selection_reuses_caller_owned_candidate_scratch() {
    let mut scratch =
        BenchmarkPassSelectionScratch::try_with_capacity(64).expect("Fix: scratch capacity");
    let names = [
        "device.synthetic.00",
        "device.synthetic.01",
        "device.synthetic.02",
        "device.synthetic.03",
        "device.synthetic.04",
        "device.synthetic.05",
        "device.synthetic.06",
        "device.synthetic.07",
        "device.synthetic.08",
        "device.synthetic.09",
        "device.synthetic.10",
        "device.synthetic.11",
        "device.synthetic.12",
        "device.synthetic.13",
        "device.synthetic.14",
        "device.synthetic.15",
    ];
    let mut wide = Vec::new();
    wide.try_reserve_exact(names.len())
        .expect("Fix: synthetic pass vector capacity");
    for (index, name) in names.iter().copied().enumerate() {
        wide.push(candidate(
            name,
            1,
            1,
            1,
            1,
            1,
            11_000 + u32::try_from(index).expect("Fix: synthetic pass index fits in u32"),
            false,
        ));
    }
    let first = select_benchmark_passes_with_scratch(
        &wide,
        BenchmarkPassSelectionSample {
            frontier_items: 64,
            reuse_count: 64,
            avoidable_readback_bytes: 64,
            planning_budget_ns: 128,
            scratch_budget_bytes: 128,
        },
        &mut scratch,
    )
    .expect("Fix: wide benchmark pass selection should plan with reusable scratch");
    let seen_capacity = scratch.seen_capacity();
    let ordered_index_capacity = scratch.ordered_index_capacity();

    assert_eq!(first.selected_pass_ids.len(), names.len());

    let second = select_benchmark_passes_with_scratch(
        &[
            candidate("device.reused.high", 1, 1, 1, 10, 8, 20_000, false),
            candidate("device.reused.low", 1, 1, 1, 10, 8, 12_000, false),
        ],
        sample(),
        &mut scratch,
    )
    .expect("Fix: smaller benchmark pass selection should reuse previous scratch");

    assert_eq!(second.selected_pass_ids[0], "device.reused.high");
    assert!(scratch.seen_capacity() >= seen_capacity);
    assert!(scratch.ordered_index_capacity() >= ordered_index_capacity);
}

#[test]
fn generated_benchmark_pass_profiles_preserve_budget_priority_and_ordering_contracts() {
    let mut scratch = BenchmarkPassSelectionScratch::default();
    for candidate_count in 1usize..=64 {
        for budget_multiplier in 1u64..=16 {
            let mut candidates = Vec::new();
            candidates
                .try_reserve_exact(candidate_count)
                .expect("Fix: generated candidate capacity");
            for index in 0..candidate_count {
                let mandatory = index % 5 == 0;
                candidates.push(candidate(
                    if mandatory {
                        "device.generated.mandatory"
                    } else {
                        "device.generated.optional"
                    },
                    1,
                    1,
                    u64::try_from(index % 4).expect("Fix: index fits"),
                    1 + u64::try_from(index % 3).expect("Fix: index fits"),
                    1,
                    11_000 + u32::try_from(index % 1_000).expect("Fix: index fits"),
                    mandatory,
                ));
                candidates[index].pass_id = generated_pass_id(index);
            }

            let plan = select_benchmark_passes_with_scratch(
                &candidates,
                BenchmarkPassSelectionSample {
                    frontier_items: 128,
                    reuse_count: 128,
                    avoidable_readback_bytes: 128,
                    planning_budget_ns: budget_multiplier * 64,
                    scratch_budget_bytes: budget_multiplier * 64,
                },
                &mut scratch,
            )
            .expect("Fix: generated benchmark pass selection profile should plan");

            let mut used_planning = 0u64;
            let mut used_scratch = 0u64;
            for pass_id in &plan.selected_pass_ids {
                let candidate = candidates
                    .iter()
                    .find(|candidate| candidate.pass_id == *pass_id)
                    .expect("Fix: selected pass must map to a generated candidate");
                used_planning += candidate.planning_cost_ns;
                used_scratch += candidate.scratch_bytes;
            }
            assert_eq!(plan.total_planning_cost_ns, used_planning);
            assert_eq!(plan.total_scratch_bytes, used_scratch);
            assert!(plan.total_planning_cost_ns <= budget_multiplier * 64);
            assert!(plan.total_scratch_bytes <= budget_multiplier * 64);
            assert!(plan.projected_speedup_bps >= 10_000);
        }
    }
}

fn generated_pass_id(index: usize) -> &'static str {
    const IDS: [&str; 64] = [
        "device.generated.00",
        "device.generated.01",
        "device.generated.02",
        "device.generated.03",
        "device.generated.04",
        "device.generated.05",
        "device.generated.06",
        "device.generated.07",
        "device.generated.08",
        "device.generated.09",
        "device.generated.10",
        "device.generated.11",
        "device.generated.12",
        "device.generated.13",
        "device.generated.14",
        "device.generated.15",
        "device.generated.16",
        "device.generated.17",
        "device.generated.18",
        "device.generated.19",
        "device.generated.20",
        "device.generated.21",
        "device.generated.22",
        "device.generated.23",
        "device.generated.24",
        "device.generated.25",
        "device.generated.26",
        "device.generated.27",
        "device.generated.28",
        "device.generated.29",
        "device.generated.30",
        "device.generated.31",
        "device.generated.32",
        "device.generated.33",
        "device.generated.34",
        "device.generated.35",
        "device.generated.36",
        "device.generated.37",
        "device.generated.38",
        "device.generated.39",
        "device.generated.40",
        "device.generated.41",
        "device.generated.42",
        "device.generated.43",
        "device.generated.44",
        "device.generated.45",
        "device.generated.46",
        "device.generated.47",
        "device.generated.48",
        "device.generated.49",
        "device.generated.50",
        "device.generated.51",
        "device.generated.52",
        "device.generated.53",
        "device.generated.54",
        "device.generated.55",
        "device.generated.56",
        "device.generated.57",
        "device.generated.58",
        "device.generated.59",
        "device.generated.60",
        "device.generated.61",
        "device.generated.62",
        "device.generated.63",
    ];
    IDS[index]
}

fn sample() -> BenchmarkPassSelectionSample {
    BenchmarkPassSelectionSample {
        frontier_items: 10,
        reuse_count: 10,
        avoidable_readback_bytes: 10,
        planning_budget_ns: 100,
        scratch_budget_bytes: 100,
    }
}

fn candidate(
    pass_id: &'static str,
    min_frontier_items: u64,
    min_reuse_count: u64,
    min_avoided_readback_bytes: u64,
    planning_cost_ns: u64,
    scratch_bytes: u64,
    expected_speedup_bps: u32,
    mandatory_when_profitable: bool,
) -> BenchmarkPassCandidate {
    BenchmarkPassCandidate {
        pass_id,
        min_frontier_items,
        min_reuse_count,
        min_avoided_readback_bytes,
        planning_cost_ns,
        scratch_bytes,
        expected_speedup_bps,
        mandatory_when_profitable,
    }
}
