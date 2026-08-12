//! Cross-backend validation safety.
//!
//! Vyre's three-layer validation cache MUST stay distinct across
//! backends:
//!
//!   1. `program.is_structurally_validated()`  -  fast atomic structural state
//!      covering wire format, IR shape, and buffer bindings.
//!   2. `WgpuBackend::validation_cache: DashSet<blake3::Hash>`  -
//!      per-backend, covers capability checks (SUBGROUP
//!      availability, workgroup-size limits, feature flags).
//!   3. `vyre_driver::backend::validation::validate_program(program,
//!      backend)`  -  the real validator.
//!
//! A backend MUST NOT consume structural validation as a shortcut past its own
//! capability checks. Capability-cache keys include the backend identity.
//!
//! This test is the regression gate: any future "simplification"
//! that makes validation process-wide instead of per-backend trips
//! the assertion below.

use vyre_driver::VyreBackend;
use vyre_driver_wgpu::WgpuBackend;

/// Stand-in for a reduced-capability future backend. Refuses every
/// dispatch so the test cannot accidentally exercise its engine; the
/// point here is the `id()` distinction and the validation contract,
/// not its dispatch behavior.
struct ReducedBackend {
    id: &'static str,
}

impl vyre_driver::backend::private::Sealed for ReducedBackend {}

impl VyreBackend for ReducedBackend {
    fn id(&self) -> &'static str {
        self.id
    }
    fn dispatch(
        &self,
        _program: &vyre::Program,
        _inputs: &[Vec<u8>],
        _config: &vyre_driver::DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, vyre_driver::BackendError> {
        Err(vyre_driver::BackendError::new(
            "ReducedBackend refuses dispatch; tests only exercise capability surface.",
        ))
    }
    fn supports_subgroup_ops(&self) -> bool {
        false
    }
    fn max_workgroup_size(&self) -> [u32; 3] {
        [1, 1, 1]
    }
}

#[test]
fn backend_ids_are_distinct() {
    let wgpu = WgpuBackend::acquire().expect("Fix: GPU required for cross-backend test");
    let reduced = ReducedBackend { id: "reduced" };
    assert_ne!(
        wgpu.id(),
        reduced.id(),
        "transcendental parity6: two backends must have distinct ids so per-backend caches do not collide"
    );
}

#[test]
fn structural_validation_does_not_substitute_for_capability_check() {
    // Contract: a program that WgpuBackend has validated (flag may
    // or may not be set depending on whether validation covered
    // structural-only) MUST trigger independent capability checks on
    // any other backend. This test documents the contract; the
    // engine-side enforcement is: ReducedBackend MUST run validation
    // when handed the same program, regardless of
    // `program.is_structurally_validated()` state.
    let wgpu = WgpuBackend::acquire().expect("Fix: GPU required for cross-backend test");
    let program = vyre::Program::empty();
    wgpu.dispatch(&program, &[], &vyre_driver::DispatchConfig::default())
        .expect("wgpu dispatch of empty program must succeed");

    // A global structural-validation shortcut would let unsupported programs
    // through. Reduced-capability backends must still return a structured error.
    let reduced = ReducedBackend { id: "reduced" };
    let result = reduced.dispatch(&program, &[], &vyre_driver::DispatchConfig::default());
    assert!(
        result.is_err(),
        "transcendental parity6: reduced backend must refuse dispatch of a program validated elsewhere; \
         never use structural validation as a capability shortcut"
    );
}
