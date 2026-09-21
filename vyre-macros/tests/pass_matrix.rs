#![allow(missing_docs)]

use crate::expansion_fixtures;

pub use expansion_fixtures::{ir, optimizer};

use vyre_macros::vyre_pass;

/// Declares one always-running fixture pass per accepted string on an axis,
/// and asserts every row's string reaches its own variant.
///
/// The `#[vyre_pass]` attribute stays at the row, because its expansion is
/// what this target tests. A row added to `pass_axis_rows!` declares its own
/// pass and joins the assertion, so there is no second list to update.
macro_rules! declare_pass_axis_coverage {
    (
        $enum_name:ident, $argument:ident,
        $($accepted:literal => $variant:ident as $fixture:ident named $pass_name:literal,)+
    ) => {
        $(
            #[vyre_pass(
                name = $pass_name,
                requires = [],
                invalidates = [],
                $argument = $accepted,
                analyze = "always"
            )]
            pub struct $fixture;

            crate::define_unchanged_pass_body!($fixture);
        )+

        #[test]
        fn every_accepted_value_reaches_its_own_variant() {
            use optimizer::{$enum_name, ProgramPass};
            let cases: &[(&dyn ProgramPass, $enum_name)] =
                &[$((&$fixture, $enum_name::$variant),)+];
            for (pass, expected) in cases {
                let metadata = pass.metadata();
                assert_eq!(metadata.$argument, *expected, "{}", metadata.name);
                assert_eq!(
                    pass.analyze(&ir::Program { id: 0 }),
                    optimizer::PassAnalysis::RUN
                );
            }
        }
    };
}

pub mod phase_axis {
    use super::{ir, optimizer};
    use vyre_macros::vyre_pass;

    crate::pass_axis_rows!(phase, declare_pass_axis_coverage);
}

pub mod boundary_class_axis {
    use super::{ir, optimizer};
    use vyre_macros::vyre_pass;

    crate::pass_axis_rows!(boundary_class, declare_pass_axis_coverage);
}

pub mod cost_model_family_axis {
    use super::{ir, optimizer};
    use vyre_macros::vyre_pass;

    crate::pass_axis_rows!(cost_model_family, declare_pass_axis_coverage);
}

/// A pass that reads device facts. Its inherent `transform_for_adapter` reports
/// a change only when the adapter offers subgroup operations, so the assertion
/// below can tell which record the expansion forwarded.
#[vyre_pass(
    name = "adapter.subgroup_dependent",
    requires = [],
    invalidates = [],
    adapter_dependent = true,
    analyze = "always"
)]
pub struct AdapterDependent;

impl AdapterDependent {
    fn transform(program: ir::Program) -> optimizer::PassResult {
        optimizer::unchanged(program)
    }

    fn transform_for_adapter(
        program: ir::Program,
        caps: &optimizer::AdapterCaps,
    ) -> optimizer::PassResult {
        optimizer::pass_result(program, caps.supports_subgroup_ops)
    }
}

/// A pass that never declares `adapter_dependent`, whose expansion must discard
/// the record rather than reach for an inherent method it does not have.
#[vyre_pass(
    name = "adapter.independent",
    requires = [],
    invalidates = [],
    analyze = "always"
)]
pub struct AdapterIndependent;

crate::define_unchanged_pass_body!(AdapterIndependent);

#[test]
fn an_adapter_dependent_pass_receives_the_record_and_an_independent_one_ignores_it() {
    use optimizer::{AdapterCaps, ProgramPass};
    let program = ir::Program { id: 7 };
    let with_subgroups = AdapterCaps {
        supports_subgroup_ops: true,
        ..AdapterCaps::conservative()
    };
    let without_subgroups = AdapterCaps::conservative();

    assert!(
        AdapterDependent
            .transform_for_adapter(program.clone(), &with_subgroups)
            .changed,
        "a device-dependent pass must see the capability it branches on"
    );
    assert!(
        !AdapterDependent
            .transform_for_adapter(program.clone(), &without_subgroups)
            .changed
    );

    for caps in [with_subgroups, without_subgroups] {
        assert_eq!(
            AdapterIndependent.transform_for_adapter(program.clone(), &caps),
            AdapterIndependent.transform(program.clone()),
            "a pass that reads no device facts must answer the same on every device"
        );
    }
}
