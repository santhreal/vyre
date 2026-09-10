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

/// The `::vyre_foundation::numeric` paths an expansion names.
pub mod numeric {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct NumericContract;
    impl NumericContract {
        pub const EXACT: Self = Self;
    }
}

/// The `::vyre_foundation::geometry` paths an expansion names.
pub mod geometry {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct GeometryRequirements;
    impl GeometryRequirements {
        pub const fn agnostic() -> Self {
            Self
        }
    }
}

/// The `::vyre_foundation::operation` paths an expansion names.
pub mod operation {
    use super::ir::Program;

    pub type OperationFixtures = fn() -> Vec<Vec<Vec<u8>>>;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum OperationTier {
        Foundation,
        Intrinsic,
        Library,
        External,
        Unknown,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct SemanticDescriptor {
        pub id: &'static str,
        pub semantic_version: u32,
        pub signature: Option<&'static ()>,
        pub tier: OperationTier,
        pub category: Option<&'static str>,
        pub laws: &'static [&'static str],
        pub numeric: super::numeric::NumericContract,
        pub geometry_requirements: super::geometry::GeometryRequirements,
        pub explicit_effects: Option<()>,
        pub explicit_capabilities: Option<()>,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct LoweringProvider {
        pub id: &'static str,
        pub build: Option<fn() -> Program>,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct ConformanceProvider {
        pub id: &'static str,
        pub test_inputs: Option<OperationFixtures>,
        pub expected_output: Option<OperationFixtures>,
    }

    inventory::collect!(SemanticDescriptor);
    inventory::collect!(LoweringProvider);
    inventory::collect!(ConformanceProvider);
}

/// The accepted metadata strings the pass attribute maps onto variant names.
///
/// Each axis is one table pairing an accepted string with the variant it
/// names, the fixture pass that declares that string, and that pass's name.
/// The stub enum below, the coverage passes in `tests/pass_matrix.rs`, and the
/// assertion over them expand from these rows, so one row is the only place a
/// string and a variant are paired. `src/pass/mod.rs` owns the accepted set: a
/// value added there and not here leaves the stub without the variant, which
/// turns the `tests/ui/bad_pass_phase.stderr` accepted list red.
#[macro_export]
macro_rules! pass_axis_rows {
    (phase, $emit:ident) => {
        $emit! { PassPhase, phase,
            "unclassified" => Unclassified as PhaseUnclassified named "phase.unclassified",
            "canonicalization" => Canonicalization as PhaseCanonicalization named "phase.canonicalization",
            "scalar_algebra" => ScalarAlgebra as PhaseScalarAlgebra named "phase.scalar_algebra",
            "loop" => Loop as PhaseLoop named "phase.loop",
            "memory" => Memory as PhaseMemory named "phase.memory",
            "fusion_cse" => FusionCse as PhaseFusionCse named "phase.fusion_cse",
            "sync" => Sync as PhaseSync named "phase.sync",
            "specialization" => Specialization as PhaseSpecialization named "phase.specialization",
            "cleanup" => Cleanup as PhaseCleanup named "phase.cleanup",
            "dataflow" => Dataflow as PhaseDataflow named "phase.dataflow",
            "megakernel" => Megakernel as PhaseMegakernel named "phase.megakernel",
        }
    };
    (boundary_class, $emit:ident) => {
        $emit! { PassBoundaryClass, boundary_class,
            "unknown" => Unknown as BoundaryUnknown named "boundary.unknown",
            "abi_preserving" => AbiPreserving as BoundaryAbiPreserving named "boundary.abi_preserving",
            "abi_changing" => AbiChanging as BoundaryAbiChanging named "boundary.abi_changing",
            "backend_aware" => BackendAware as BoundaryBackendAware named "boundary.backend_aware",
            "runtime_aware" => RuntimeAware as BoundaryRuntimeAware named "boundary.runtime_aware",
            "domain_specific" => DomainSpecific as BoundaryDomainSpecific named "boundary.domain_specific",
        }
    };
    (cost_model_family, $emit:ident) => {
        $emit! { CostModelFamily, cost_model_family,
            "unknown" => Unknown as CostUnknown named "cost.unknown",
            "scalar" => Scalar as CostScalar named "cost.scalar",
            "loop" => Loop as CostLoop named "cost.loop",
            "memory" => Memory as CostMemory named "cost.memory",
            "fusion" => Fusion as CostFusion named "cost.fusion",
            "sync" => Sync as CostSync named "cost.sync",
            "dataflow" => Dataflow as CostDataflow named "cost.dataflow",
            "megakernel" => Megakernel as CostMegakernel named "cost.megakernel",
        }
    };
}

/// Declares one stub axis enum from its row table. The variant doc is the
/// accepted string that selects it.
#[macro_export]
macro_rules! declare_pass_axis_enum {
    (
        $enum_name:ident, $argument:ident,
        $($accepted:literal => $variant:ident as $fixture:ident named $pass_name:literal,)+
    ) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum $enum_name {
            $(#[doc = $accepted] $variant,)+
        }
    };
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

    pass_axis_rows!(phase, declare_pass_axis_enum);
    pass_axis_rows!(boundary_class, declare_pass_axis_enum);
    pass_axis_rows!(cost_model_family, declare_pass_axis_enum);

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
