//! Upper driver-contract checks for the neutral C statement-bounds program.

#![cfg(feature = "c-parser")]

use vyre_foundation::ir::Expr;
use vyre_libs::parsing::c::parse::structure_statement::c11_statement_bounds;

fn geometry(n: u32) -> (u32, [u32; 3]) {
    let program = c11_statement_bounds("tok_types", Expr::u32(n), "out_statements", "out_counts");
    let plan = vyre_driver::BindingPlan::build(&program)
        .expect("neutral statement-bounds declarations must produce a binding plan");
    let count = vyre_driver::dispatch_element_count_for_program(&program, &plan.bindings);
    let grid = vyre_driver::infer_dispatch_grid_for_count(count, program.workgroup_size())
        .expect("the declared one-dimensional workgroup must infer a grid");
    (count, grid)
}

#[test]
fn launch_geometry_preserves_positive_boundary_and_adversarial_routes() {
    assert_eq!(geometry(256), (512, [2, 1, 1]));
    assert_eq!(geometry(65_536), (131_072, [512, 1, 1]));
    assert_eq!(geometry(130_560).1[0], 1_020);
    assert_eq!(geometry(130_688).1[0], 1_021);
}

#[test]
fn unsupported_autotuner_widening_remains_explicitly_disabled() {
    let limits = vyre_driver::validation::LaunchGeometryLimits {
        backend: "cuda",
        max_threads_per_block: 1024,
        max_block_dim: [1024, 1024, 64],
        max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
        max_threads_per_sm: 0,
    };
    let config = vyre_driver::DispatchConfig::default();
    for n in [65_536_u32, 87_040, 130_560] {
        let program =
            c11_statement_bounds("tok_types", Expr::u32(n), "out_statements", "out_counts");
        assert!(program.non_composable_with_self);
        let plan = vyre_driver::BindingPlan::build(&program)
            .expect("neutral statement-bounds declarations must produce a binding plan");
        let count = vyre_driver::dispatch_element_count_for_program(&program, &plan.bindings);
        for mode in [
            vyre_driver::tuner::Mode::production_default(),
            vyre_driver::tuner::Mode::OffUseDefault,
        ] {
            let effective = vyre_driver::resolve_launch_workgroup_for_mode(
                &program, &config, limits, count, mode,
            );
            assert_eq!(effective, [256, 1, 1]);
        }
    }
}
