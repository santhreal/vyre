#![allow(missing_docs)]

extern crate self as vyre;

mod support;

pub use support::{ir, optimizer};

use vyre_macros::vyre_pass;

/// Declare an always-running pass that sets exactly one metadata argument.
///
/// The three axes below differ only in which argument they write, so the
/// argument name is a parameter rather than three copies of the declaration.
macro_rules! define_single_argument_pass {
    ($ty:ident, $name:literal, $argument:ident = $value:literal) => {
        #[vyre_pass(name = $name, requires = [], invalidates = [], $argument = $value, analyze = "always")]
        pub struct $ty;

        crate::define_unchanged_pass_body!($ty);
    };
}

define_single_argument_pass!(
    PhaseUnclassified,
    "phase.unclassified",
    phase = "unclassified"
);
define_single_argument_pass!(
    PhaseCanonicalization,
    "phase.canonicalization",
    phase = "canonicalization"
);
define_single_argument_pass!(
    PhaseScalarAlgebra,
    "phase.scalar_algebra",
    phase = "scalar_algebra"
);
define_single_argument_pass!(PhaseLoop, "phase.loop", phase = "loop");
define_single_argument_pass!(PhaseMemory, "phase.memory", phase = "memory");
define_single_argument_pass!(PhaseFusionCse, "phase.fusion_cse", phase = "fusion_cse");
define_single_argument_pass!(PhaseSync, "phase.sync", phase = "sync");
define_single_argument_pass!(
    PhaseSpecialization,
    "phase.specialization",
    phase = "specialization"
);
define_single_argument_pass!(PhaseCleanup, "phase.cleanup", phase = "cleanup");
define_single_argument_pass!(PhaseDataflow, "phase.dataflow", phase = "dataflow");
define_single_argument_pass!(PhaseMegakernel, "phase.megakernel", phase = "megakernel");

define_single_argument_pass!(
    BoundaryUnknown,
    "boundary.unknown",
    boundary_class = "unknown"
);
define_single_argument_pass!(
    BoundaryAbiPreserving,
    "boundary.abi_preserving",
    boundary_class = "abi_preserving"
);
define_single_argument_pass!(
    BoundaryAbiChanging,
    "boundary.abi_changing",
    boundary_class = "abi_changing"
);
define_single_argument_pass!(
    BoundaryBackendAware,
    "boundary.backend_aware",
    boundary_class = "backend_aware"
);
define_single_argument_pass!(
    BoundaryRuntimeAware,
    "boundary.runtime_aware",
    boundary_class = "runtime_aware"
);
define_single_argument_pass!(
    BoundaryDomainSpecific,
    "boundary.domain_specific",
    boundary_class = "domain_specific"
);

define_single_argument_pass!(CostUnknown, "cost.unknown", cost_model_family = "unknown");
define_single_argument_pass!(CostScalar, "cost.scalar", cost_model_family = "scalar");
define_single_argument_pass!(CostLoop, "cost.loop", cost_model_family = "loop");
define_single_argument_pass!(CostMemory, "cost.memory", cost_model_family = "memory");
define_single_argument_pass!(CostFusion, "cost.fusion", cost_model_family = "fusion");
define_single_argument_pass!(CostSync, "cost.sync", cost_model_family = "sync");
define_single_argument_pass!(
    CostDataflow,
    "cost.dataflow",
    cost_model_family = "dataflow"
);
define_single_argument_pass!(
    CostMegakernel,
    "cost.megakernel",
    cost_model_family = "megakernel"
);

#[test]
fn vyre_pass_phase_matrix_emits_expected_metadata() {
    use optimizer::{PassPhase, ProgramPass};
    let cases: &[(&dyn ProgramPass, PassPhase)] = &[
        (&PhaseUnclassified, PassPhase::Unclassified),
        (&PhaseCanonicalization, PassPhase::Canonicalization),
        (&PhaseScalarAlgebra, PassPhase::ScalarAlgebra),
        (&PhaseLoop, PassPhase::Loop),
        (&PhaseMemory, PassPhase::Memory),
        (&PhaseFusionCse, PassPhase::FusionCse),
        (&PhaseSync, PassPhase::Sync),
        (&PhaseSpecialization, PassPhase::Specialization),
        (&PhaseCleanup, PassPhase::Cleanup),
        (&PhaseDataflow, PassPhase::Dataflow),
        (&PhaseMegakernel, PassPhase::Megakernel),
    ];
    for (pass, phase) in cases {
        let metadata = pass.metadata();
        assert_eq!(metadata.phase, *phase, "{}", metadata.name);
        assert_eq!(
            pass.analyze(&ir::Program { id: 0 }),
            optimizer::PassAnalysis::RUN
        );
    }
}

#[test]
fn vyre_pass_boundary_matrix_emits_expected_metadata() {
    use optimizer::{PassBoundaryClass, ProgramPass};
    let cases: &[(&dyn ProgramPass, PassBoundaryClass)] = &[
        (&BoundaryUnknown, PassBoundaryClass::Unknown),
        (&BoundaryAbiPreserving, PassBoundaryClass::AbiPreserving),
        (&BoundaryAbiChanging, PassBoundaryClass::AbiChanging),
        (&BoundaryBackendAware, PassBoundaryClass::BackendAware),
        (&BoundaryRuntimeAware, PassBoundaryClass::RuntimeAware),
        (&BoundaryDomainSpecific, PassBoundaryClass::DomainSpecific),
    ];
    for (pass, boundary_class) in cases {
        let metadata = pass.metadata();
        assert_eq!(
            metadata.boundary_class, *boundary_class,
            "{}",
            metadata.name
        );
    }
}

#[test]
fn vyre_pass_cost_model_matrix_emits_expected_metadata() {
    use optimizer::{CostModelFamily, ProgramPass};
    let cases: &[(&dyn ProgramPass, CostModelFamily)] = &[
        (&CostUnknown, CostModelFamily::Unknown),
        (&CostScalar, CostModelFamily::Scalar),
        (&CostLoop, CostModelFamily::Loop),
        (&CostMemory, CostModelFamily::Memory),
        (&CostFusion, CostModelFamily::Fusion),
        (&CostSync, CostModelFamily::Sync),
        (&CostDataflow, CostModelFamily::Dataflow),
        (&CostMegakernel, CostModelFamily::Megakernel),
    ];
    for (pass, cost_model_family) in cases {
        let metadata = pass.metadata();
        assert_eq!(
            metadata.cost_model_family, *cost_model_family,
            "{}",
            metadata.name
        );
    }
}
