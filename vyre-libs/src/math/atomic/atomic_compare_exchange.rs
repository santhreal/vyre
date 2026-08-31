//! Cat-B `atomic_compare_exchange_u32`. CPU ref: for each i, if
//! `state == expected[i]`, replace state with `desired[i]`; always
//! emit the pre-op state into `trace[i]`.

use vyre_foundation::ir::Program;

use super::build_atomic_compare_exchange;

const OP_ID: &str = "vyre-libs::math::atomic::atomic_compare_exchange_u32";

/// Sequential compare-and-exchange over pairs `(expected[i], desired[i])`.
#[must_use]
pub fn atomic_compare_exchange_u32(
    expected: &str,
    desired: &str,
    state: &str,
    trace: &str,
    n: u32,
) -> Program {
    build_atomic_compare_exchange(OP_ID, expected, desired, state, trace, n)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || atomic_compare_exchange_u32("expected", "desired", "state", "trace", 4),
        Some(|| {
            let to_bytes = vyre_primitives::wire::pack_u32_slice;
            vec![vec![
                to_bytes(&[10u32, 99, 20, 30]),
                to_bytes(&[11u32, 88, 21, 31]),
                to_bytes(&[10u32]),
            ]]
        }),
        Some(|| {
            // Final state=11, trace=[10,11,11,11].
            vec![vec![
                vec![0x0b, 0x00, 0x00, 0x00], // state: 11
                vec![
                    0x0a, 0x00, 0x00, 0x00, // trace[0]: 10
                    0x0b, 0x00, 0x00, 0x00, // trace[1]: 11
                    0x0b, 0x00, 0x00, 0x00, // trace[2]: 11
                    0x0b, 0x00, 0x00, 0x00, // trace[3]: 11
                ],
            ]]
        }),
    )
    .with_category("math")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::atomic::testutil::run_cas;

    #[test]
    fn swaps_when_expected_matches() {
        let expected = vec![10u32, 99, 20, 30];
        let desired = vec![11u32, 88, 21, 31];
        let initial = 10u32;
        let program = atomic_compare_exchange_u32(
            "expected",
            "desired",
            "state",
            "trace",
            expected.len() as u32,
        );
        let (final_state, trace) = run_cas(&program, &expected, &desired, initial);

        let mut cpu_state = initial;
        let mut cpu_trace = Vec::new();
        for (&e, &d) in expected.iter().zip(desired.iter()) {
            cpu_trace.push(cpu_state);
            if cpu_state == e {
                cpu_state = d;
            }
        }

        assert_eq!(final_state, cpu_state);
        assert_eq!(trace, cpu_trace);
    }
}
