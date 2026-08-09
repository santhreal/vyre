//! Typed artifact invocation-grid contracts.

use vyre_driver::{BackendError, BindingSet};
use vyre_megakernel::Digest;

/// WHY: runtime workload geometry varies independently of immutable artifact identity,
/// but zero-sized grids must fail before target submission.
#[test]
fn invocation_grid_is_typed_validated_and_identity_neutral() {
    let artifact = Digest([7; 32]);
    let mut bindings = BindingSet::new(artifact);

    bindings
        .set_invocation_grid([3, 2, 1])
        .expect("positive invocation grid must be accepted");
    assert_eq!(bindings.artifact(), artifact);
    assert_eq!(bindings.invocation_grid(), Some([3, 2, 1]));

    let error = bindings
        .set_invocation_grid([3, 0, 1])
        .expect_err("zero invocation extent must fail closed");
    let BackendError::InvalidProgram { fix } = error else {
        panic!("zero invocation extent must be an invalid-program error");
    };
    assert!(fix.contains("axis 1") && fix.contains("must be positive"));
    assert_eq!(bindings.invocation_grid(), Some([3, 2, 1]));
}
