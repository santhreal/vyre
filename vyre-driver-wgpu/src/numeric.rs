use vyre_driver::numeric::BackendNumericPolicy;

/// Single WGPU numeric-boundary policy.
///
/// One label binding of the shared [`BackendNumericPolicy`], not a per-helper
/// wrapper fork. Mirrors `vyre-driver-cuda`'s `CUDA_NUMERIC` so there is one
/// numeric-policy pattern across every driver.
pub(crate) const WGPU_NUMERIC: BackendNumericPolicy = BackendNumericPolicy::new("WGPU");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgpu_numeric_policy_binds_the_backend_identity() {
        assert_eq!(WGPU_NUMERIC.backend(), "WGPU");
        assert_eq!(
            WGPU_NUMERIC
                .usize_to_u64(17, "numeric policy fixture")
                .expect("Fix: WGPU numeric policy must preserve valid host counts."),
            17
        );
    }
}
