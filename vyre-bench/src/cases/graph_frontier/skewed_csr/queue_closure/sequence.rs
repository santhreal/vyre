use crate::api::case::{BenchContext, BenchError};
use crate::api::resident::ResidentInputSet;
use crate::cases::queue_stage::{
    dispatch_resident_queue_closure_sequence as dispatch_shared_queue_closure_sequence,
    ResidentQueueClosureSpec,
};

use super::{GraphCsrSkewedQueueClosurePrepared, GRAPH_QUEUE_CLOSURE_WORKGROUP_SIZE};

pub(super) use crate::cases::queue_stage::QueueClosureSequenceRun;

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct QueueClosureRepeatedPlan {
    leading_a_to_b_half_wave: bool,
    repeated_pair_count: u32,
}

#[cfg(test)]
impl QueueClosureRepeatedPlan {
    const fn total_half_waves(self) -> u32 {
        self.repeated_pair_count
            .saturating_mul(2)
            .saturating_add(self.leading_a_to_b_half_wave as u32)
    }

    const fn dispatch_count(self) -> u32 {
        1_u32.saturating_add(self.total_half_waves().saturating_mul(2))
    }
}

#[cfg(test)]
const fn queue_closure_repeated_plan(closure_iterations: u32) -> QueueClosureRepeatedPlan {
    QueueClosureRepeatedPlan {
        leading_a_to_b_half_wave: closure_iterations & 1 == 1,
        repeated_pair_count: closure_iterations / 2,
    }
}

pub(super) fn dispatch_resident_queue_closure_sequence(
    ctx: &BenchContext,
    prepared: &GraphCsrSkewedQueueClosurePrepared,
    resident: &ResidentInputSet,
) -> Result<QueueClosureSequenceRun, BenchError> {
    dispatch_shared_queue_closure_sequence(
        ctx,
        ResidentQueueClosureSpec {
            reset_program: &prepared.reset_program,
            clear_len_program: &prepared.clear_len_program,
            delta_program: &prepared.delta_program,
            frontier_words: prepared.stats.frontier_words,
            seed_queue_len: prepared.seed_queue_len,
            baseline_output_len: prepared.baseline_output.len(),
            closure_iterations: prepared.closure_iterations,
            delta_grid: prepared.delta_grid,
            workgroup: GRAPH_QUEUE_CLOSURE_WORKGROUP_SIZE,
            context: "skewed CSR queue closure",
        },
        resident,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_graph_repeated_plan_preserves_every_queue_closure_wave() {
        const CASES: u32 = 10_000;
        let mut odd_cases = 0_u32;
        let mut repeated_pairs = 0_u64;

        for case in 0..CASES {
            let iterations = mix32(case ^ 0x6A17_0359) % 16_385;
            let plan = queue_closure_repeated_plan(iterations);

            assert_eq!(plan.total_half_waves(), iterations, "case {case}");
            assert_eq!(
                plan.dispatch_count(),
                1 + iterations.saturating_mul(2),
                "dispatch count case {case}"
            );
            assert_eq!(
                plan.leading_a_to_b_half_wave,
                iterations & 1 == 1,
                "leading wave parity case {case}"
            );
            assert_eq!(
                plan.repeated_pair_count,
                iterations / 2,
                "pair count case {case}"
            );
            assert_repeated_plan_expands_to_alternating_half_waves(case, iterations, plan);

            odd_cases += u32::from(plan.leading_a_to_b_half_wave);
            repeated_pairs += u64::from(plan.repeated_pair_count);
        }

        assert!(odd_cases > CASES / 3);
        assert!(repeated_pairs > u64::from(CASES) * 1_000);
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

    const fn mix32(mut value: u32) -> u32 {
        value ^= value >> 16;
        value = value.wrapping_mul(0x7FEB_352D);
        value ^= value >> 15;
        value = value.wrapping_mul(0x846C_A68B);
        value ^ (value >> 16)
    }
}
