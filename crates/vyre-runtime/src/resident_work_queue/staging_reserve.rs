//! Shared fallible staging reservations for megakernel host-side runtime paths.

use crate::PipelineError;
use vyre_runtime::reservation_policy::ReservationPolicy;

const MEGAKERNEL_STAGING: ReservationPolicy = ReservationPolicy::new(
    "megakernel runtime staging",
    "split the megakernel workload or reuse caller-owned staging",
);

pub(super) fn try_reserve_vec_capacity<T>(
    values: &mut Vec<T>,
    capacity: usize,
) -> Result<(), String> {
    MEGAKERNEL_STAGING
        .reserve_vec_to_capacity(values, capacity, "element")
        .map_err(|source| source.to_string())
}

pub(super) fn reserve_vec_capacity<T>(
    values: &mut Vec<T>,
    capacity: usize,
    label: &'static str,
) -> Result<(), PipelineError> {
    MEGAKERNEL_STAGING
        .reserve_vec_to_capacity(values, capacity, label)
        .map_err(|source| {
            PipelineError::Backend(format!(
                "megakernel {label} reservation failed for {capacity} element(s): {source}"
            ))
        })
}

// Inline: covers `reserve_vec_capacity`, which no integration test can name.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_vec_capacity_grows_after_clear() {
        let mut values = Vec::<u32>::with_capacity(4);
        values.extend_from_slice(&[1, 2, 3, 4]);
        values.clear();
        reserve_vec_capacity(&mut values, 8, "test vector")
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - cleared vector should grow to target capacity");
        assert!(values.capacity() >= 8);
        assert!(values.is_empty());
    }

    #[test]
    fn reserve_vec_capacity_reports_context_on_overflow() {
        let mut values = Vec::<u8>::new();
        let err = reserve_vec_capacity(&mut values, usize::MAX, "huge telemetry")
            .expect_err("oversized reservation should fail");
        let message = err.to_string();
        assert!(message.contains("huge telemetry"));
        assert!(message.contains("Fix:"));
    }
}
