#![allow(unreachable_pub)]

pub mod ir {
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Program {
        pub id: u64,
    }
}

pub mod optimizer {
    use super::ir::Program;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PassMetadata {
        pub name: &'static str,
        pub requires: &'static [&'static str],
        pub invalidates: &'static [&'static str],
        pub phase: PassPhase,
        pub boundary_class: PassBoundaryClass,
        pub requires_caps: &'static [&'static str],
        pub preserves_abi: bool,
        pub cost_model_family: CostModelFamily,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum PassPhase {
        Unclassified,
        Canonicalization,
        ScalarAlgebra,
        Loop,
        Memory,
        FusionCse,
        Sync,
        Specialization,
        Cleanup,
        Dataflow,
        Megakernel,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum PassBoundaryClass {
        Unknown,
        AbiPreserving,
        AbiChanging,
        BackendAware,
        RuntimeAware,
        DomainSpecific,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum CostModelFamily {
        Unknown,
        Scalar,
        Loop,
        Memory,
        Fusion,
        Sync,
        Dataflow,
        Megakernel,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PassAnalysis {
        pub should_run: bool,
    }

    impl PassAnalysis {
        pub const RUN: Self = Self { should_run: true };
        pub const SKIP: Self = Self { should_run: false };
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct PassResult {
        pub program: Program,
        pub changed: bool,
    }

    pub fn pass_result(program: Program, changed: bool) -> PassResult {
        PassResult { program, changed }
    }

    pub fn unchanged(program: Program) -> PassResult {
        pass_result(program, false)
    }

    pub mod private {
        pub trait Sealed {}
    }

    pub trait ProgramPass: private::Sealed + Send + Sync {
        fn metadata(&self) -> PassMetadata;
        fn analyze(&self, program: &Program) -> PassAnalysis;
        fn transform(&self, program: Program) -> PassResult;
        fn fingerprint(&self, program: &Program) -> u64;
    }

    pub struct ProgramPassRegistration {
        pub metadata: PassMetadata,
        pub factory: fn() -> Box<dyn ProgramPass>,
    }

    inventory::collect!(ProgramPassRegistration);

    pub fn fingerprint_program(program: &Program) -> u64 {
        program.id ^ 0x9e37_79b9_7f4a_7c15
    }
}
