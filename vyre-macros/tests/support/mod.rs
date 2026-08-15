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

    /// Device facts a pass may compile against. The macro names this type in
    /// every expansion, so the stub carries the fields the generated code and
    /// the test passes read, under the names the real record uses.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct AdapterCaps {
        pub backend: &'static str,
        pub supports_subgroup_ops: bool,
        pub max_workgroup_size: [u32; 3],
    }

    impl AdapterCaps {
        #[must_use]
        pub const fn conservative() -> Self {
            Self {
                backend: "conservative",
                supports_subgroup_ops: false,
                max_workgroup_size: [256, 1, 1],
            }
        }
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

    pub mod sealed {
        pub trait Sealed {}
    }

    pub trait ProgramPass: sealed::Sealed + Send + Sync {
        fn metadata(&self) -> PassMetadata;
        fn analyze(&self, program: &Program) -> PassAnalysis;
        fn transform(&self, program: Program) -> PassResult;

        /// Mirrors the real trait: the default discards the adapter, so a pass
        /// that does not override this compiles to the same program everywhere.
        /// The stub has to carry it because the macro emits an override for a
        /// pass declared adapter_dependent, and a stub trait without the member
        /// makes every such expansion an E0407 that names the trait rather than
        /// the generated code.
        fn transform_for_adapter(&self, program: Program, caps: &AdapterCaps) -> PassResult {
            let _ = caps;
            self.transform(program)
        }
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

/// Assert the metadata `#[vyre_pass]` emits when every optional argument is
/// omitted.
///
/// This is one contract, not one per test target: the defaults are what a pass
/// author gets for free, so a change to any of them is a change to the macro's
/// published behaviour and has to be visible in exactly one place.
pub fn assert_default_metadata(metadata: &optimizer::PassMetadata, name: &str) {
    assert_eq!(metadata.name, name);
    assert_eq!(metadata.requires, &[] as &[&str]);
    assert_eq!(metadata.invalidates, &[] as &[&str]);
    assert_eq!(metadata.phase, optimizer::PassPhase::Unclassified);
    assert_eq!(
        metadata.boundary_class,
        optimizer::PassBoundaryClass::Unknown
    );
    assert_eq!(metadata.requires_caps, &[] as &[&str]);
    assert!(metadata.preserves_abi);
    assert_eq!(
        metadata.cost_model_family,
        optimizer::CostModelFamily::Unknown
    );
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
