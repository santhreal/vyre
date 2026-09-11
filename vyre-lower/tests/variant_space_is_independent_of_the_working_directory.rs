//! The physical-IR variant space answers for the compiled source, not for the
//! directory the test process happens to stand in.
//!
//! WHY: every emitter contract suite enumerates `KernelOpKind` to derive the
//! space it states a decision over. Each suite used to resolve a workspace root
//! from the working directory and read `descriptor/mod.rs` off disk, so a run
//! whose working directory pointed at another checkout enumerated another
//! tree's enum, compared it against this tree's emitters, and passed. The
//! enumerator now reads the source that was compiled, which removes the
//! working directory from the answer.

use std::collections::BTreeSet;

use vyre_lower::variant_space::kernel_op_kind_variants;

/// A variant every physical-IR enum has carried since the enum existed, used
/// to prove the scan returned a real space rather than an empty one.
const ANCHOR_VARIANT: &str = "StoreGlobal";

#[test]
fn the_variant_space_is_the_same_from_any_working_directory() {
    let from_the_checkout = kernel_op_kind_variants();
    assert!(
        from_the_checkout.contains(ANCHOR_VARIANT),
        "Fix: the scan must find the declared variants: {from_the_checkout:?}"
    );

    let elsewhere = std::env::temp_dir();
    std::env::set_current_dir(&elsewhere).unwrap_or_else(|e| {
        panic!(
            "Fix: {} must be usable as a working directory: {e}",
            elsewhere.display()
        )
    });

    let from_elsewhere: BTreeSet<String> = kernel_op_kind_variants();
    assert_eq!(
        from_elsewhere,
        from_the_checkout,
        "Fix: the variant space is a property of the compiled source. Reading it \
         off disk relative to the working directory makes a run from {} enumerate \
         whatever enum that tree declares",
        elsewhere.display()
    );
}

#[test]
fn a_name_the_enum_does_not_declare_is_absent_from_the_space() {
    let space = kernel_op_kind_variants();
    for absent in [
        "EmitterDecision",
        "KernelOpKind",
        "MatrixMmaSpec",
        "Serialize",
    ] {
        assert!(
            !space.contains(absent),
            "Fix: `{absent}` is not a `KernelOpKind` variant. A scan that admits \
             surrounding declarations states a space wider than the enum, and a \
             contract over that space asserts decisions for members that do not exist"
        );
    }
}
