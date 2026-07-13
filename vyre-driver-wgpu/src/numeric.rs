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

    const SOURCE: &str = include_str!("numeric.rs");

    #[test]
    fn wgpu_numeric_module_is_policy_binding_not_helper_fork() {
        assert_eq!(WGPU_NUMERIC.backend(), "WGPU");
        assert!(
            SOURCE.contains("BackendNumericPolicy::new(\"WGPU\")"),
            "WGPU numeric ownership must stay in vyre-driver::numeric"
        );
        let forbidden_wrapper = concat!("pub(crate) ", "fn");
        assert!(
            !SOURCE.contains(forbidden_wrapper),
            "WGPU must not reintroduce per-helper numeric wrappers"
        );
    }
}
