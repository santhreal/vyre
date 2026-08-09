#![forbid(unsafe_code)]
//! Universal Cat-A op harness registry + Region builder.
//!
//! **Registry Layering**: This file defines the `OpEntry` registry for Tier-3 Cat-A compositions.
//! It operates in parallel with the Tier-2.5 primitives registry (`vyre-primitives::harness::OpEntry`) and the Tier-2 hardware intrinsics registry (`vyre-intrinsics::harness::OpEntry`).
//! For an architectural overview of this three-registry split, see `vyre-harness/README.md`.
//!
//! Every Cat-A composition that participates in automated harness
//! checks registers one `OpEntry` through `inventory::submit!`. The
//! conform integration test at `tests/universal_harness.rs` discovers
//! every entry and validates: program validity, wire round-trip, CSE
//! stability, and (when available) CPU-oracle parity.
//!
//! The crate also re-exports the Region builder used by every Cat-A
//! library to wrap its produced `Vec<Node>` so optimizer passes treat
//! the library call as an opaque unit by default. See
//! [`region`] for `wrap`, `wrap_anonymous`, `wrap_child`,
//! `tag_program`.

pub mod fp_contract;
pub mod region;

pub use region::{reparent_program_children, tag_program, wrap, wrap_anonymous, wrap_child};

/// Re-exported so the [`vyre_op!`] macro can call into `inventory`
/// without callers needing to add it as a direct dependency.
#[doc(hidden)]
pub use inventory;


/// Canonical operation tier used by harness, catalog, and matrix gates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpTier {
    /// Foundation-level IR rewrite or built-in IR operation.
    FoundationIr,
    /// Tier-2 hardware intrinsic.
    Intrinsic,
    /// Tier-2.5 reusable primitive.
    Primitive,
    /// Tier-3 library composition.
    Libs,
    /// Runtime or driver-owned operation.
    Runtime,
    /// External consumer registered into the shared harness.
    External,
    /// Identifier does not match any accepted registry namespace.
    Unknown,
}

impl OpTier {
    /// Return the stable `OP_MATRIX.toml` tier spelling.
    #[must_use]
    pub const fn matrix_value(self) -> &'static str {
        match self {
            Self::FoundationIr => "foundation_ir",
            Self::Intrinsic => "intrinsic",
            Self::Primitive => "primitive",
            Self::Libs => "libs",
            Self::Runtime => "runtime",
            Self::External => "external",
            Self::Unknown => "unknown",
        }
    }
}

/// Classify an operation id by the canonical namespace contract.
#[must_use]
pub fn classify_op_id(id: &str) -> OpTier {
    if id.starts_with("vyre-intrinsics::hardware::") {
        OpTier::Intrinsic
    } else if id.starts_with("vyre-primitives::") {
        OpTier::Primitive
    } else if id.starts_with("vyre-libs::") {
        OpTier::Libs
    } else if id.starts_with("core.") || id.starts_with("io.") || id.starts_with("mem.") {
        OpTier::Runtime
    } else if is_external_crate_namespace(id) {
        OpTier::External
    } else {
        OpTier::Unknown
    }
}

fn is_external_crate_namespace(id: &str) -> bool {
    let Some((crate_name, _)) = id.split_once("::") else {
        return false;
    };
    !crate_name.is_empty() && !crate_name.starts_with("vyre-")
}

/// Canonical semantic operation registration.
pub use vyre_foundation::operation::OperationRegistration as OpEntry;

/// Deterministic fixture input cases.
pub type InputsFn = vyre_foundation::operation::OperationFixtures;
/// Deterministic expected-output fixtures.
pub type ExpectedFn = vyre_foundation::operation::OperationFixtures;

/// Return every canonical semantic operation registration linked into the binary.
pub fn all_entries() -> impl Iterator<Item = &'static OpEntry> {
    vyre_foundation::operation::OperationRegistry::global().iter()
}

/// Fixpoint contract for dataflow ops whose GPU body performs one
/// iteration per dispatch.
///
/// Submitting a `FixpointRegistration` alongside an `OpEntry` tells the
/// conform harness to call `backend.dispatch` in a loop until the
/// `converged_flag_buffer` reads zero before comparing against the CPU
/// reference. Without this registration such ops would always diverge
/// in a single-dispatch byte-identity test even though their lowering
/// is correct.
#[derive(Clone, Debug)]
pub struct FixpointContract {
    /// Name of the RW buffer whose bytes-interpreted-as-`u32` must
    /// equal zero for the fixpoint loop to terminate. Semantics: the
    /// GPU body writes `1` whenever any lane updated shared state;
    /// the driver clears it between iterations.
    pub converged_flag_buffer: &'static str,
    /// Hard cap on driver iterations before the loop bails out. Every
    /// fixpoint op MUST reach its answer in a known-bounded number of
    /// steps so the harness cannot hang.
    pub max_iterations: u32,
}

/// Link-time registration binding a fixpoint contract to an op id.
pub struct FixpointRegistration {
    /// Stable op id (`OpEntry::id`) this contract applies to.
    pub op_id: &'static str,
    /// Fixpoint contract parameters.
    pub contract: FixpointContract,
}

inventory::collect!(FixpointRegistration);

/// Look up the fixpoint contract registered for `op_id`, if any.
#[must_use]
pub fn fixpoint_contract(op_id: &str) -> Option<&'static FixpointContract> {
    inventory::iter::<FixpointRegistration>()
        .find(|registration| registration.op_id == op_id)
        .map(|registration| &registration.contract)
}

/// Convergence contract for ops whose GPU body performs one
/// iteration per dispatch and needs an external driver loop to
/// reach fixpoint before byte-identity comparison.
///
/// Submitting a `ConvergenceContract` alongside an `OpEntry` tells
/// the conform harness to dispatch the backend in a loop (transfer
/// step + `bitset_fixpoint` convergence check) until the changed
/// flag clears or the iteration budget is exhausted.
#[derive(Clone, Debug)]
pub struct ConvergenceContract {
    /// Stable op id (`OpEntry::id`) this contract applies to.
    pub op_id: &'static str,
    /// Hard cap on driver iterations before the loop bails out.
    pub max_iterations: u32,
}

inventory::collect!(ConvergenceContract);

/// Look up the convergence contract registered for `op_id`, if any.
#[must_use]
pub fn convergence_contract(op_id: &str) -> Option<&'static ConvergenceContract> {
    inventory::iter::<ConvergenceContract>().find(|contract| contract.op_id == op_id)
}

// Tolerance metadata and capability requirements are encoded in
// [`OpEntry::tolerance`] and checked by the conform lenses directly.
// There is no global exemption registry; every registered op must
// provide runnable `test_inputs` / `expected_output` fixtures or fail
// loudly with a diagnostic.

/// Declarative op registration shorthand (ROADMAP S8 generator half).
///
/// One source declares a Cat-A op; the macro expands to the matching
/// `inventory::submit!{ OpEntry { .. } }` so writers can't accidentally
/// drift from the canonical `OpEntry` shape.
///
/// ## Forms
///
/// ```ignore
/// // Minimal: just id + builder.
/// vyre_harness::vyre_op! {
///     id: "vyre-libs::math::matmul",
///     build: || matmul("a", "b", "out", 2, 2, 2),
/// }
///
/// // Full: explicit fixtures.
/// vyre_harness::vyre_op! {
///     id: "vyre-libs::math::matmul",
///     build: || matmul("a", "b", "out", 2, 2, 2),
///     test_inputs: || vec![vec![input_bytes(), output_zeros()]],
///     expected_output: || vec![vec![expected_bytes()]],
/// }
/// ```
///
/// `test_inputs` / `expected_output` default to `None` when omitted.
/// Every form expands to exactly one `inventory::submit!` of an
/// `OpEntry`, keeping the registration shape locked to the struct
/// definition above.
///
/// Future `OpEntry` field additions extend this macro with defaulted
/// arms so existing call sites stay green.
#[macro_export]
macro_rules! vyre_op {
    (
        id: $id:expr,
        build: $build:expr $(,)?
    ) => {
        $crate::vyre_op! {
            id: $id,
            build: $build,
            test_inputs: ::core::option::Option::None,
            expected_output: ::core::option::Option::None,
        }
    };
    (
        id: $id:expr,
        build: $build:expr,
        test_inputs: $inputs:expr,
        expected_output: $output:expr $(,)?
    ) => {
        $crate::inventory::submit! {
            $crate::OpEntry::new(
                $id,
                vyre_foundation::operation::OperationTier::External,
                ::core::option::Option::Some($build),
                $inputs,
                $output,
            )
        }
    };
}

#[cfg(test)]
mod tests {

    // ---------------- vyre_op! macro (S8 generator) ----------------

    fn _trivial_program() -> vyre_foundation::ir::Program {
        use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
        Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
        )
    }

    // Minimal-form invocation; expansion succeeds at compile time.
    crate::vyre_op! {
        id: "vyre-harness::test::trivial_minimal",
        build: _trivial_program,
    }

    // Full-form invocation with explicit fixtures.
    crate::vyre_op! {
        id: "vyre-harness::test::trivial_full",
        build: _trivial_program,
        test_inputs: ::core::option::Option::Some(|| vec![vec![vec![0u8; 4]]]),
        expected_output: ::core::option::Option::Some(|| vec![vec![vec![7u8, 0, 0, 0]]]),
    }

    #[test]
    fn vyre_op_macro_minimal_form_registers_entry() {
        let entry = crate::all_entries()
            .find(|e| e.id == "vyre-harness::test::trivial_minimal")
            .expect("Fix: vyre_op! minimal form must register an OpEntry");
        assert!(entry.test_inputs.is_none());
        assert!(entry.expected_output.is_none());
    }

    #[test]
    fn vyre_op_macro_full_form_registers_entry_with_fixtures() {
        let entry = crate::all_entries()
            .find(|e| e.id == "vyre-harness::test::trivial_full")
            .expect("Fix: vyre_op! full form must register an OpEntry");
        assert!(entry.test_inputs.is_some());
        assert!(entry.expected_output.is_some());
    }

    #[test]
    fn vyre_op_macro_build_fn_produces_program() {
        let entry = crate::all_entries()
            .find(|e| e.id == "vyre-harness::test::trivial_minimal")
            .expect("Fix: entry must exist");
        let program = entry
            .program()
            .expect("Fix: macro registration must retain its neutral builder");
        assert_ne!(program.entry().len(), 0);
    }
}
