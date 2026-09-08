//! The `::vyre` surface the expanded attribute writes its paths against.
//!
//! Each suite declares `extern crate self as vyre` and re-exports `ir` and
//! `optimizer` from here, so the expansion's `::vyre::optimizer::ProgramPass`
//! resolves inside the test binary. Every item those two modules hold has to be
//! `pub` for that resolution.

/// The `::vyre::ir` paths an expansion names.
pub mod ir {
    /// The program a pass transforms, reduced to the one field the stub
    /// fingerprint reads.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Program {
        /// Identity the stub fingerprint and the id-gated analysis read.
        pub id: u64,
    }
}

/// The `::vyre::optimizer` paths an expansion names.
pub mod optimizer {
    use super::ir::Program;

    /// The record a pass declares itself with, under the field names the
    /// attribute writes.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PassMetadata {
        /// Stable pass name.
        pub name: &'static str,
        /// Analyses the pass reads.
        pub requires: &'static [&'static str],
        /// Analyses the pass invalidates.
        pub invalidates: &'static [&'static str],
        /// Where in the pipeline the pass runs.
        pub phase: PassPhase,
        /// What the pass is allowed to change across a boundary.
        pub boundary_class: PassBoundaryClass,
        /// Capabilities the pass compiles against.
        pub requires_caps: &'static [&'static str],
        /// Whether the pass leaves the buffer ABI intact.
        pub preserves_abi: bool,
        /// Which cost model ranks the pass.
        pub cost_model_family: CostModelFamily,
    }

    /// Pipeline phase, mirroring the real enum's variant names because the
    /// attribute writes one of them into every expansion.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum PassPhase {
        /// No phase declared.
        Unclassified,
        /// Normalizes IR shape.
        Canonicalization,
        /// Rewrites scalar arithmetic.
        ScalarAlgebra,
        /// Rewrites loop structure.
        Loop,
        /// Rewrites memory access.
        Memory,
        /// Fuses nodes and eliminates common subexpressions.
        FusionCse,
        /// Places synchronization.
        Sync,
        /// Specializes against known facts.
        Specialization,
        /// Removes what earlier phases left.
        Cleanup,
        /// Rewrites dataflow structure.
        Dataflow,
        /// Builds the megakernel schedule.
        Megakernel,
    }

    /// What a pass may change across a boundary.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum PassBoundaryClass {
        /// No class declared.
        Unknown,
        /// Leaves the buffer ABI intact.
        AbiPreserving,
        /// Changes the buffer ABI.
        AbiChanging,
        /// Reads backend facts.
        BackendAware,
        /// Reads runtime facts.
        RuntimeAware,
        /// Holds only for one domain.
        DomainSpecific,
    }

    /// Which cost model ranks a pass.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum CostModelFamily {
        /// No family declared.
        Unknown,
        /// Scalar arithmetic cost.
        Scalar,
        /// Loop trip-count cost.
        Loop,
        /// Memory traffic cost.
        Memory,
        /// Fusion cost.
        Fusion,
        /// Synchronization cost.
        Sync,
        /// Dataflow cost.
        Dataflow,
        /// Megakernel schedule cost.
        Megakernel,
    }

    /// Device facts a pass may compile against. The macro names this type in
    /// every expansion, so the stub carries the fields the generated code and
    /// the test passes read, under the names the real record uses.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct AdapterCaps {
        /// Neutral backend id.
        pub backend: &'static str,
        /// Whether subgroup intrinsics may be emitted.
        pub supports_subgroup_ops: bool,
        /// Largest workgroup the device accepts.
        pub max_workgroup_size: [u32; 3],
    }

    impl AdapterCaps {
        /// Facts every device satisfies, for a pass compiled without a probe.
        #[must_use]
        pub const fn conservative() -> Self {
            Self {
                backend: "conservative",
                supports_subgroup_ops: false,
                max_workgroup_size: [256, 1, 1],
            }
        }
    }

    /// What an analysis decided about running the pass.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PassAnalysis {
        /// Whether the transform runs.
        pub should_run: bool,
    }

    impl PassAnalysis {
        /// Run the transform.
        pub const RUN: Self = Self { should_run: true };
        /// Skip the transform.
        pub const SKIP: Self = Self { should_run: false };
    }

    /// The program a transform produced, and whether it differs.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct PassResult {
        /// The program after the transform.
        pub program: Program,
        /// Whether the transform changed it.
        pub changed: bool,
    }

    /// A result stating `changed` explicitly.
    pub fn pass_result(program: Program, changed: bool) -> PassResult {
        PassResult { program, changed }
    }

    /// A result stating the program is unchanged.
    pub fn unchanged(program: Program) -> PassResult {
        pass_result(program, false)
    }

    /// The supertrait the attribute implements to seal `ProgramPass`.
    pub mod sealed {
        /// Implemented only by the attribute's expansion.
        pub trait Sealed {}
    }

    /// The trait every expansion implements.
    pub trait ProgramPass: sealed::Sealed + Send + Sync {
        /// The pass's declaration.
        fn metadata(&self) -> PassMetadata;
        /// Whether the transform runs on `program`.
        fn analyze(&self, program: &Program) -> PassAnalysis;
        /// Rewrite `program`.
        fn transform(&self, program: Program) -> PassResult;

        /// Mirrors the real trait: the default ignores the adapter, so a pass
        /// that does not override this compiles to the same program everywhere.
        /// The stub has to carry it because the macro emits an override for a
        /// pass declared adapter_dependent, and a stub trait without the member
        /// makes every such expansion an E0407 that names the trait rather than
        /// the generated code.
        fn transform_for_adapter(&self, program: Program, _caps: &AdapterCaps) -> PassResult {
            self.transform(program)
        }
        /// Identity of `program` under this pass.
        fn fingerprint(&self, program: &Program) -> u64;
    }

    /// The inventory row the attribute submits.
    pub struct ProgramPassRegistration {
        /// The registered pass's declaration.
        pub metadata: PassMetadata,
        /// Constructs the registered pass.
        pub factory: fn() -> Box<dyn ProgramPass>,
    }

    inventory::collect!(ProgramPassRegistration);

    /// The program identity the stub passes fingerprint against.
    pub fn fingerprint_program(program: &Program) -> u64 {
        program.id ^ 0x9e37_79b9_7f4a_7c15
    }
}

/// Inherent impl for a pass whose analysis is gated on a nonzero program id and
/// whose transform reports a change.
///
/// The `#[vyre_pass]` attribute stays at each use site, because its expansion
/// is what these targets are testing. Only the inherent body the attribute
/// forwards to is shared.
#[macro_export]
macro_rules! define_id_gated_pass_body {
    ($ty:ident) => {
        impl $ty {
            fn analyze_impl(program: &$crate::ir::Program) -> $crate::optimizer::PassAnalysis {
                if program.id == 0 {
                    $crate::optimizer::PassAnalysis::SKIP
                } else {
                    $crate::optimizer::PassAnalysis::RUN
                }
            }

            fn transform(program: $crate::ir::Program) -> $crate::optimizer::PassResult {
                $crate::optimizer::pass_result(program, true)
            }
        }
    };
}

/// Inherent impl for a pass that always runs and never changes the program.
#[macro_export]
macro_rules! define_always_run_pass_body {
    ($ty:ident) => {
        impl $ty {
            fn analyze_impl(_program: &$crate::ir::Program) -> $crate::optimizer::PassAnalysis {
                $crate::optimizer::PassAnalysis::RUN
            }

            fn transform(program: $crate::ir::Program) -> $crate::optimizer::PassResult {
                $crate::optimizer::unchanged(program)
            }
        }
    };
}

/// Inherent impl for a pass declared `analyze = "always"`, which supplies no
/// `analyze_impl` of its own.
#[macro_export]
macro_rules! define_unchanged_pass_body {
    ($ty:ident) => {
        impl $ty {
            fn transform(program: $crate::ir::Program) -> $crate::optimizer::PassResult {
                $crate::optimizer::unchanged(program)
            }
        }
    };
}
