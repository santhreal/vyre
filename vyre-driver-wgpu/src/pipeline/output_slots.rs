//! WGPU wrapper around backend-neutral output-slot resizing.

use vyre_driver::BackendError;

pub(crate) fn resize_vec_with<T, F>(
    vec: &mut Vec<T>,
    len: usize,
    make: F,
    label: &'static str,
) -> Result<(), BackendError>
where
    F: FnMut() -> T,
{
    vyre_driver::output_slots::resize_vec_with(
        vec,
        len,
        make,
        "WGPU pipeline",
        label,
        "split the dispatch batch before readback",
    )
}

#[cfg(test)]
mod tests {
    use super::resize_vec_with;

    /// The neutral resize policy is proved in `vyre_driver::output_slots`. What
    /// belongs here is the only thing this wrapper decides: which backend and
    /// which corrective action a failed reservation names. A verbatim copy of
    /// the policy test used to sit here and asserted neither.
    #[test]
    fn a_failed_reservation_names_the_wgpu_pipeline_and_its_corrective_action() {
        let mut slots: Vec<Vec<u8>> = Vec::new();
        let error = resize_vec_with(&mut slots, usize::MAX, Vec::new, "compiled output slots")
            .expect_err("Fix: a slot count that cannot be reserved must fail, not truncate.");
        let rendered = error.to_string();
        for expected in [
            "WGPU pipeline",
            "compiled output slots",
            "split the dispatch batch before readback",
        ] {
            assert!(
                rendered.contains(expected),
                "Fix: the WGPU slot-reservation diagnostic must contain `{expected}`, got: {rendered}"
            );
        }
    }

    #[test]
    fn resizing_through_the_wrapper_grows_shrinks_and_holds_at_the_boundaries() {
        let mut slots: Vec<Vec<u8>> = vec![vec![1], vec![2, 3]];

        resize_vec_with(&mut slots, 4, Vec::new, "grow")
            .expect("Fix: growing output slots must succeed.");
        assert_eq!(
            slots,
            vec![vec![1], vec![2, 3], Vec::new(), Vec::new()],
            "Fix: growth must preserve the existing prefix and add empty slots."
        );

        resize_vec_with(&mut slots, 4, Vec::new, "hold")
            .expect("Fix: resizing to the current length must succeed.");
        assert_eq!(
            slots.len(),
            4,
            "Fix: resizing to the current length must not change the slot count."
        );

        resize_vec_with(&mut slots, 1, Vec::new, "shrink")
            .expect("Fix: shrinking output slots must succeed.");
        assert_eq!(
            slots,
            vec![vec![1_u8]],
            "Fix: shrinking must drop stale trailing slots and keep the prefix."
        );

        resize_vec_with(&mut slots, 0, Vec::new, "empty")
            .expect("Fix: resizing to zero slots must succeed.");
        assert!(
            slots.is_empty(),
            "Fix: resizing to zero must leave no output slots."
        );
    }
}
