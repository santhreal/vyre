//! Host dispatch for the queue-closure benchmark sequence.
//!
//! The IFDS dataflow and skewed-CSR frontier cases drive a reachability
//! closure over GPU-resident ping-pong queues. The input payload, the binding
//! layout, the seed reset program, the half-wave plan and the resident
//! dispatch are one fact each and are stated here.

use std::time::Instant;

use crate::api::case::{BenchContext, BenchError};
use crate::api::metric::elapsed_ns;
use crate::api::resident::{stated_launch, ResidentInputSet};
use vyre_driver::{ResidentDispatchStep, ResidentReadRange};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_queue_closure_inputs(
    frontier: &[u32],
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    active_sources: u64,
    queue_capacity: u32,
    context: &str,
    materialize_seed_queue: impl FnOnce(usize) -> Result<Vec<u32>, BenchError>,
) -> Result<Vec<Vec<u8>>, BenchError> {
    if u64::from(queue_capacity) < active_sources {
        return Err(BenchError::EnvironmentInvalid(format!(
            "{context} queue closure requires queue_capacity >= active_sources, got capacity={queue_capacity} active_sources={active_sources}. Fix: size ping-pong queues for the seed frontier."
        )));
    }
    let seed_queue_len = u32::try_from(active_sources).map_err(|_| {
        BenchError::EnvironmentInvalid(format!(
            "{context} queue closure active source count {active_sources} exceeds u32 indexing. Fix: split the seed queue."
        ))
    })?;
    let queue_bytes = (queue_capacity as usize)
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            BenchError::EnvironmentInvalid(format!(
                "{context} queue closure queue_capacity={queue_capacity} overflows host buffer sizing. Fix: split the frontier queue."
            ))
        })?;
    let seed_frontier = vyre_primitives::wire::pack_u32_slice(frontier);
    let seed_queue = materialize_seed_queue(seed_queue_len as usize)?;

    let mut inputs = vec![Vec::new(); QUEUE_CLOSURE_INPUT_COUNT];
    inputs[QUEUE_CLOSURE_SEED_FRONTIER_INDEX] = seed_frontier.clone();
    inputs[QUEUE_CLOSURE_SEED_QUEUE_INDEX] = vyre_primitives::wire::pack_u32_slice(&seed_queue);
    inputs[QUEUE_CLOSURE_SEED_LEN_INDEX] = vyre_primitives::wire::pack_u32_slice(&[seed_queue_len]);
    inputs[QUEUE_CLOSURE_QUEUE_A_INDEX] = vec![0_u8; queue_bytes];
    inputs[QUEUE_CLOSURE_LEN_A_INDEX] = vyre_primitives::wire::pack_u32_slice(&[0]);
    inputs[QUEUE_CLOSURE_QUEUE_B_INDEX] = vec![0_u8; queue_bytes];
    inputs[QUEUE_CLOSURE_LEN_B_INDEX] = vyre_primitives::wire::pack_u32_slice(&[0]);
    inputs[QUEUE_CLOSURE_EDGE_OFFSETS_INDEX] = vyre_primitives::wire::pack_u32_slice(edge_offsets);
    inputs[QUEUE_CLOSURE_EDGE_TARGETS_INDEX] = vyre_primitives::wire::pack_u32_slice(edge_targets);
    inputs[QUEUE_CLOSURE_EDGE_KIND_INDEX] = vyre_primitives::wire::pack_u32_slice(edge_kind_mask);
    inputs[QUEUE_CLOSURE_ACCUMULATOR_INDEX] = seed_frontier;
    Ok(inputs)
}

pub(crate) struct ResidentQueueClosureSpec<'a> {
    pub(crate) reset_program: &'a Program,
    pub(crate) clear_len_program: &'a Program,
    pub(crate) delta_program: &'a Program,
    pub(crate) frontier_words: u32,
    pub(crate) seed_queue_len: u32,
    pub(crate) baseline_output_len: usize,
    pub(crate) closure_iterations: u32,
    pub(crate) delta_grid: [u32; 3],
    pub(crate) workgroup: [u32; 3],
    pub(crate) context: &'static str,
}

pub(crate) struct QueueClosureSequenceRun {
    pub(crate) outputs: Vec<Vec<u8>>,
    pub(crate) wall_ns: u64,
}

/// Binding order of the queue-closure workload.
///
/// Both closure cases bind the same resources in the same order, because they
/// run the same reset and delta programs. The layout is one fact and lives
/// here, next to the dispatch that consumes it, rather than once per case.
pub(crate) const QUEUE_CLOSURE_SEED_FRONTIER_INDEX: usize = 0;
pub(crate) const QUEUE_CLOSURE_SEED_QUEUE_INDEX: usize = 1;
pub(crate) const QUEUE_CLOSURE_SEED_LEN_INDEX: usize = 2;
pub(crate) const QUEUE_CLOSURE_QUEUE_A_INDEX: usize = 3;
pub(crate) const QUEUE_CLOSURE_LEN_A_INDEX: usize = 4;
pub(crate) const QUEUE_CLOSURE_QUEUE_B_INDEX: usize = 5;
pub(crate) const QUEUE_CLOSURE_LEN_B_INDEX: usize = 6;
pub(crate) const QUEUE_CLOSURE_EDGE_OFFSETS_INDEX: usize = 7;
pub(crate) const QUEUE_CLOSURE_EDGE_TARGETS_INDEX: usize = 8;
pub(crate) const QUEUE_CLOSURE_EDGE_KIND_INDEX: usize = 9;
pub(crate) const QUEUE_CLOSURE_ACCUMULATOR_INDEX: usize = 10;
pub(crate) const QUEUE_CLOSURE_INPUT_COUNT: usize = 11;

pub(crate) fn build_queue_closure_reset_program(
    frontier_words: u32,
    seed_queue_len: u32,
    queue_capacity: u32,
    workgroup: [u32; 3],
) -> Program {
    let idx = Expr::InvocationId { axis: 0 };
    Program::wrapped(
        vec![
            BufferDecl::storage("frontier_seed", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(frontier_words.max(1)),
            BufferDecl::storage("seed_queue", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(seed_queue_len.max(1)),
            BufferDecl::storage("seed_len", 2, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("active_queue", 3, BufferAccess::ReadWrite, DataType::U32)
                .with_count(queue_capacity.max(1)),
            BufferDecl::storage("accumulator", 4, BufferAccess::ReadWrite, DataType::U32)
                .with_count(frontier_words.max(1)),
            BufferDecl::storage("queue_a_len", 5, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
            BufferDecl::storage("queue_b_len", 6, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        workgroup,
        vec![
            Node::if_then(
                Expr::lt(idx.clone(), Expr::u32(frontier_words)),
                vec![Node::store(
                    "accumulator",
                    idx.clone(),
                    Expr::load("frontier_seed", idx.clone()),
                )],
            ),
            Node::if_then(
                Expr::and(
                    Expr::lt(idx.clone(), Expr::u32(queue_capacity)),
                    Expr::and(
                        Expr::lt(idx.clone(), Expr::u32(seed_queue_len)),
                        Expr::lt(idx.clone(), Expr::load("seed_len", Expr::u32(0))),
                    ),
                ),
                vec![Node::store(
                    "active_queue",
                    idx.clone(),
                    Expr::load("seed_queue", idx.clone()),
                )],
            ),
            Node::if_then(
                Expr::eq(idx, Expr::u32(0)),
                vec![
                    Node::store(
                        "queue_a_len",
                        Expr::u32(0),
                        Expr::load("seed_len", Expr::u32(0)),
                    ),
                    Node::store("queue_b_len", Expr::u32(0), Expr::u32(0)),
                ],
            ),
        ],
    )
}

/// How a closure of `closure_iterations` half-waves folds into one prefix plus
/// a repeated four-step pair.
///
/// The delta program alternates direction every half-wave, so a pair of
/// half-waves is the shortest repeatable unit. An odd iteration count cannot be
/// expressed as pairs alone: the leading A-to-B half-wave is hoisted into the
/// prefix and the remainder divides evenly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct QueueClosureRepeatedPlan {
    pub(crate) leading_a_to_b_half_wave: bool,
    pub(crate) repeated_pair_count: u32,
}

impl QueueClosureRepeatedPlan {
    /// Half-waves the plan expands to. Must equal the requested iterations.
    pub(crate) const fn total_half_waves(self) -> u32 {
        self.repeated_pair_count
            .saturating_mul(2)
            .saturating_add(self.leading_a_to_b_half_wave as u32)
    }

    /// Dispatches the plan submits: one reset plus two per half-wave.
    pub(crate) const fn dispatch_count(self) -> u32 {
        1_u32.saturating_add(self.total_half_waves().saturating_mul(2))
    }
}

/// Fold an iteration count into its prefix-plus-repeated-pair plan.
pub(crate) const fn queue_closure_repeated_plan(
    closure_iterations: u32,
) -> QueueClosureRepeatedPlan {
    QueueClosureRepeatedPlan {
        leading_a_to_b_half_wave: closure_iterations & 1 == 1,
        repeated_pair_count: closure_iterations / 2,
    }
}

pub(crate) fn dispatch_resident_queue_closure_sequence(
    ctx: &BenchContext,
    prepared: ResidentQueueClosureSpec<'_>,
    resident: &ResidentInputSet,
) -> Result<QueueClosureSequenceRun, BenchError> {
    // Position of the accumulator inside the reset kernel's binding list,
    // derived so the readback slot cannot drift from the list below it.
    const RESET_ACCUMULATOR_RESOURCE: usize = {
        let mut slot = 0;
        while slot < RESET_RESOURCE_INDICES.len() {
            if RESET_RESOURCE_INDICES[slot] == QUEUE_CLOSURE_ACCUMULATOR_INDEX {
                break;
            }
            slot += 1;
        }
        assert!(
            slot < RESET_RESOURCE_INDICES.len(),
            "the reset kernel must bind the accumulator it reads back"
        );
        slot
    };
    const RESET_RESOURCE_INDICES: [usize; 7] = [
        QUEUE_CLOSURE_SEED_FRONTIER_INDEX,
        QUEUE_CLOSURE_SEED_QUEUE_INDEX,
        QUEUE_CLOSURE_SEED_LEN_INDEX,
        QUEUE_CLOSURE_QUEUE_A_INDEX,
        QUEUE_CLOSURE_ACCUMULATOR_INDEX,
        QUEUE_CLOSURE_LEN_A_INDEX,
        QUEUE_CLOSURE_LEN_B_INDEX,
    ];
    const CLEAR_A_RESOURCE_INDICES: [usize; 1] = [QUEUE_CLOSURE_LEN_A_INDEX];
    const CLEAR_B_RESOURCE_INDICES: [usize; 1] = [QUEUE_CLOSURE_LEN_B_INDEX];
    const DELTA_A_TO_B_RESOURCE_INDICES: [usize; 8] = [
        QUEUE_CLOSURE_QUEUE_A_INDEX,
        QUEUE_CLOSURE_LEN_A_INDEX,
        QUEUE_CLOSURE_EDGE_OFFSETS_INDEX,
        QUEUE_CLOSURE_EDGE_TARGETS_INDEX,
        QUEUE_CLOSURE_EDGE_KIND_INDEX,
        QUEUE_CLOSURE_ACCUMULATOR_INDEX,
        QUEUE_CLOSURE_QUEUE_B_INDEX,
        QUEUE_CLOSURE_LEN_B_INDEX,
    ];
    const DELTA_B_TO_A_RESOURCE_INDICES: [usize; 8] = [
        QUEUE_CLOSURE_QUEUE_B_INDEX,
        QUEUE_CLOSURE_LEN_B_INDEX,
        QUEUE_CLOSURE_EDGE_OFFSETS_INDEX,
        QUEUE_CLOSURE_EDGE_TARGETS_INDEX,
        QUEUE_CLOSURE_EDGE_KIND_INDEX,
        QUEUE_CLOSURE_ACCUMULATOR_INDEX,
        QUEUE_CLOSURE_QUEUE_A_INDEX,
        QUEUE_CLOSURE_LEN_A_INDEX,
    ];

    let resource_sets = [
        resident.resources_for_indices(
            &RESET_RESOURCE_INDICES,
            &format!("{} reset", prepared.context),
        )?,
        resident.resources_for_indices(
            &CLEAR_A_RESOURCE_INDICES,
            &format!("{} clear queue A length", prepared.context),
        )?,
        resident.resources_for_indices(
            &CLEAR_B_RESOURCE_INDICES,
            &format!("{} clear queue B length", prepared.context),
        )?,
        resident.resources_for_indices(
            &DELTA_A_TO_B_RESOURCE_INDICES,
            &format!("{} delta A to B", prepared.context),
        )?,
        resident.resources_for_indices(
            &DELTA_B_TO_A_RESOURCE_INDICES,
            &format!("{} delta B to A", prepared.context),
        )?,
    ];
    let reset_grid = [
        prepared
            .frontier_words
            .max(prepared.seed_queue_len)
            .div_ceil(prepared.workgroup[0])
            .max(1),
        1,
        1,
    ];
    let reset_step = ResidentDispatchStep {
        program: prepared.reset_program,
        resources: &resource_sets[0],
        launch: Some(stated_launch(prepared.reset_program, reset_grid)?),
    };
    let read_ranges = [ResidentReadRange {
        resource: &resource_sets[0][RESET_ACCUMULATOR_RESOURCE],
        byte_offset: 0,
        byte_len: prepared.baseline_output_len,
    }];
    let mut accumulator_output = Vec::with_capacity(prepared.baseline_output_len);
    let started = Instant::now();
    let plan = queue_closure_repeated_plan(prepared.closure_iterations);
    let clear_launch = stated_launch(prepared.clear_len_program, [1, 1, 1])?;
    let delta_launch = stated_launch(prepared.delta_program, prepared.delta_grid)?;
    let clear_a_step = || ResidentDispatchStep {
        program: prepared.clear_len_program,
        resources: &resource_sets[1],
        launch: Some(clear_launch),
    };
    let clear_b_step = || ResidentDispatchStep {
        program: prepared.clear_len_program,
        resources: &resource_sets[2],
        launch: Some(clear_launch),
    };
    let delta_a_to_b_step = || ResidentDispatchStep {
        program: prepared.delta_program,
        resources: &resource_sets[3],
        launch: Some(delta_launch),
    };
    let delta_b_to_a_step = || ResidentDispatchStep {
        program: prepared.delta_program,
        resources: &resource_sets[4],
        launch: Some(delta_launch),
    };

    if plan.leading_a_to_b_half_wave {
        let prefix_steps = [reset_step, clear_b_step(), delta_a_to_b_step()];
        let repeated_steps = [
            clear_a_step(),
            delta_b_to_a_step(),
            clear_b_step(),
            delta_a_to_b_step(),
        ];
        ctx.dispatch_resident_repeated_sequence_read_ranges_into(
            &prefix_steps,
            &repeated_steps,
            plan.repeated_pair_count,
            &read_ranges,
            &mut [&mut accumulator_output],
        )
    } else {
        let prefix_steps = [reset_step];
        let repeated_steps = [
            clear_b_step(),
            delta_a_to_b_step(),
            clear_a_step(),
            delta_b_to_a_step(),
        ];
        ctx.dispatch_resident_repeated_sequence_read_ranges_into(
            &prefix_steps,
            &repeated_steps,
            plan.repeated_pair_count,
            &read_ranges,
            &mut [&mut accumulator_output],
        )
    }
    .map_err(|error| BenchError::BackendFailed(error.to_string()))?;

    Ok(QueueClosureSequenceRun {
        outputs: vec![accumulator_output],
        wall_ns: elapsed_ns(started),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::mix32;

    /// The repeated-pair plan must expand to exactly the requested half-waves,
    /// in alternating direction, for every iteration count either queue closure
    /// case can produce.
    ///
    /// Both salts are the ones the IFDS and CSR cases each generated against
    /// while they carried their own copy of this plan, so collapsing them onto
    /// one owner did not narrow the input space.
    #[test]
    fn generated_repeated_plan_preserves_every_queue_closure_wave() {
        const CASES: u32 = 10_000;

        for salt in [0xC105_E7E5_u32, 0x6A17_0359] {
            let mut odd_cases = 0_u32;
            let mut repeated_pairs = 0_u64;

            for case in 0..CASES {
                let iterations = mix32(case ^ salt) % 16_385;
                let plan = queue_closure_repeated_plan(iterations);

                assert_eq!(
                    plan.total_half_waves(),
                    iterations,
                    "salt {salt:#x} case {case}"
                );
                assert_eq!(
                    plan.dispatch_count(),
                    1 + iterations.saturating_mul(2),
                    "dispatch count salt {salt:#x} case {case}"
                );
                assert_eq!(
                    plan.leading_a_to_b_half_wave,
                    iterations & 1 == 1,
                    "leading wave parity salt {salt:#x} case {case}"
                );
                assert_eq!(
                    plan.repeated_pair_count,
                    iterations / 2,
                    "pair count salt {salt:#x} case {case}"
                );
                assert_repeated_plan_expands_to_alternating_half_waves(case, iterations, plan);

                odd_cases += u32::from(plan.leading_a_to_b_half_wave);
                repeated_pairs += u64::from(plan.repeated_pair_count);
            }

            assert!(odd_cases > CASES / 3, "salt {salt:#x}");
            assert!(repeated_pairs > u64::from(CASES) * 1_000, "salt {salt:#x}");
        }
    }

    fn assert_repeated_plan_expands_to_alternating_half_waves(
        case: u32,
        iterations: u32,
        plan: QueueClosureRepeatedPlan,
    ) {
        let mut half_wave = 0_u32;
        if plan.leading_a_to_b_half_wave {
            assert_half_wave(case, half_wave, true);
            half_wave += 1;
        }

        for _ in 0..plan.repeated_pair_count {
            if plan.leading_a_to_b_half_wave {
                assert_half_wave(case, half_wave, false);
                half_wave += 1;
                assert_half_wave(case, half_wave, true);
            } else {
                assert_half_wave(case, half_wave, true);
                half_wave += 1;
                assert_half_wave(case, half_wave, false);
            }
            half_wave += 1;
        }

        assert_eq!(half_wave, iterations, "expanded wave count case {case}");
    }

    fn assert_half_wave(case: u32, half_wave: u32, a_to_b: bool) {
        assert_eq!(
            a_to_b,
            half_wave & 1 == 0,
            "half-wave direction case {case} wave {half_wave}"
        );
    }

    /// A zero-iteration closure still submits the reset dispatch and nothing
    /// else; the boundary the generated cases never reach.
    #[test]
    fn a_zero_iteration_closure_submits_only_the_reset() {
        let plan = queue_closure_repeated_plan(0);

        assert_eq!(plan.total_half_waves(), 0);
        assert_eq!(plan.dispatch_count(), 1);
        assert!(!plan.leading_a_to_b_half_wave);
    }
}
