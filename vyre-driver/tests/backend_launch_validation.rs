//! Shared backend launch validation contracts.

use vyre_driver::{BackendError, DispatchConfig, VyreBackend};
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

struct GridLimitBackend;

impl vyre_driver::sealed::Sealed for GridLimitBackend {}

impl VyreBackend for GridLimitBackend {
    fn id(&self) -> &'static str {
        "grid-limit-test"
    }

    fn dispatch_borrowed(
        &self,
        _: &Program,
        _: &[&[u8]],
        _: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        Ok(Vec::new())
    }

    fn max_workgroup_size(&self) -> [u32; 3] {
        [256, 256, 64]
    }

    fn max_compute_invocations_per_workgroup(&self) -> u32 {
        256
    }

    fn max_compute_workgroups_per_dimension(&self) -> u32 {
        7
    }
}

fn tiny_program() -> Program {
    Program::wrapped(Vec::new(), [1, 1, 1], vec![Node::Return])
}

#[test]
fn validate_program_for_backend_rejects_grid_override_past_backend_axis_limit() {
    let backend = GridLimitBackend;
    let program = tiny_program();

    for (axis, grid) in [(0, [8, 1, 1]), (1, [1, 8, 1]), (2, [1, 1, 8])] {
        let mut config = DispatchConfig::default();
        config.grid_override = Some(grid);

        let err =
            vyre_driver::validation::validate_program_for_backend(&backend, &program, &config)
                .expect_err("Fix: grid_override above the backend per-dimension limit must fail.");
        let msg = err.to_string();
        assert!(
            msg.contains("Fix:"),
            "backend validation errors must remain actionable; got: {msg}"
        );
        assert!(
            msg.contains(&format!("axis {axis}")),
            "grid validation must identify the failing axis; got: {msg}"
        );
        assert!(
            msg.contains("max is 7"),
            "grid validation must include the backend limit; got: {msg}"
        );
    }
}

#[test]
fn backend_error_preserves_structured_validation_source() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32)],
        [0, 1, 1],
        Vec::new(),
    );
    let caps = vyre_driver::validation::ProgramValidationCaps {
        backend_id: "grid-limit-test",
        supports_subgroup_ops: false,
        supports_f16: false,
        supports_bf16: false,
        supports_indirect_dispatch: false,
        supports_distributed_collectives: false,
        supports_trap_propagation: false,
        supports_grid_sync: false,
        allows_host_grid_sync_split: false,
        max_workgroup_size: [256, 256, 64],
    };

    let error = vyre_driver::validation::validate_program_contract(
        &program,
        vyre_foundation::validate::ValidationOptions::default(),
        vyre_driver::default_supported_ops(),
        caps,
    )
    .expect_err("zero workgroup axis must fail validation");
    let BackendError::Validation { source } = error else {
        panic!("foundation validation must retain its structured source");
    };
    assert_eq!(source.code().as_str(), "V106");
    assert!(matches!(
        source.location(),
        vyre_foundation::validate::ValidationLocation::WorkgroupAxis(0)
    ));
}

#[test]
fn validate_program_for_backend_accepts_grid_override_at_backend_axis_limit() {
    let backend = GridLimitBackend;
    let program = tiny_program();
    let mut config = DispatchConfig::default();
    config.grid_override = Some([7, 7, 7]);

    vyre_driver::validation::validate_program_for_backend(&backend, &program, &config)
        .expect("Fix: grid_override equal to the backend per-dimension limit must be valid.");
}

#[test]
fn validate_program_for_backend_rejects_zero_grid_override_dimension() {
    let backend = GridLimitBackend;
    let program = tiny_program();
    let mut config = DispatchConfig::default();
    config.grid_override = Some([1, 0, 1]);

    let err = vyre_driver::validation::validate_program_for_backend(&backend, &program, &config)
        .expect_err("Fix: zero grid_override dimensions must fail before backend dispatch.");
    let msg = err.to_string();
    assert!(
        msg.contains("Fix:") && msg.contains("zero-sized grid dimensions"),
        "zero-grid validation must be actionable; got: {msg}"
    );
}

#[test]
fn validate_launch_geometry_rejects_per_axis_block_overflow() {
    let err = vyre_driver::validation::validate_launch_geometry(
        [1, 1, 65],
        [1, 1, 1],
        vyre_driver::validation::LaunchGeometryLimits {
            backend: "test",
            max_threads_per_block: 256,
            max_block_dim: [256, 256, 64],
            max_grid_dim: [1024, 1024, 1024],
            max_threads_per_sm: 0,
        },
    )
    .expect_err("Fix: per-axis block overflow must fail even when total threads are legal.");
    let msg = err.to_string();
    assert!(
        msg.contains("axis 2") && msg.contains("max is 64"),
        "per-axis launch validation must identify the failed axis and limit; got: {msg}"
    );
}

#[test]
fn launch_plan_prepares_geometry_and_param_words_once() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(1_000),
            BufferDecl::output("out", 1, DataType::U32).with_count(1_000),
        ],
        [128, 1, 1],
        vec![Node::store(
            "out",
            Expr::logical_index(0),
            Expr::load("input", Expr::logical_index(0)),
        )],
    );
    let bindings = vyre_driver::BindingPlan::build(&program)
        .expect("Fix: shared launch-plan test program must build a binding plan.");
    let mut config = DispatchConfig::default();
    config.workgroup_override = Some([128, 1, 1]);
    let launch = vyre_driver::LaunchPlan::from_bindings(
        &program,
        &bindings.bindings,
        &config,
        vyre_driver::validation::LaunchGeometryLimits {
            backend: "test",
            max_threads_per_block: 256,
            max_block_dim: [256, 256, 64],
            max_grid_dim: [1024, 1024, 1024],
            max_threads_per_sm: 0,
        },
    )
    .expect("Fix: shared launch plan must infer legal geometry.");

    assert_eq!(launch.element_count, 1_000);
    assert_eq!(launch.workgroup, [128, 1, 1]);
    assert_eq!(launch.grid, [8, 1, 1]);
    assert_eq!(launch.param_words[0], 1_000);
    assert_eq!(launch.param_words[1], 1_000);
    assert_eq!(launch.param_words[2], 1_000);
}

#[test]
fn launch_plan_rejects_zero_grid_override_before_driver_entry() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(4)],
        [64, 1, 1],
        vec![Node::Return],
    );
    let bindings = vyre_driver::BindingPlan::build(&program)
        .expect("Fix: shared launch-plan test program must build a binding plan.");
    let mut config = DispatchConfig::default();
    config.grid_override = Some([1, 0, 1]);

    let err = vyre_driver::LaunchPlan::from_bindings(
        &program,
        &bindings.bindings,
        &config,
        vyre_driver::validation::LaunchGeometryLimits {
            backend: "test",
            max_threads_per_block: 256,
            max_block_dim: [256, 256, 64],
            max_grid_dim: [1024, 1024, 1024],
            max_threads_per_sm: 0,
        },
    )
    .expect_err("Fix: shared launch preparation must reject zero grid overrides.");
    assert!(
        err.to_string().contains("non-zero"),
        "Fix: shared launch preparation must return actionable geometry errors; got: {err}"
    );
}

/// WHY: a dispatch that states a frozen launch and a tuner override states two
/// launches, and resolving them picks one and drops the other with nothing
/// reported. One of the two is then a shape nothing compiled against. Every
/// dispatch-shape field is covered, not the one field a caller happened to set;
/// `DispatchConfig::validate_launch_authority` destructures the struct field by
/// field, so a new dispatch-shape field cannot be added without a decision here.
#[test]
fn a_frozen_launch_beside_any_dispatch_shape_override_is_rejected() {
    let backend = GridLimitBackend;
    let program = tiny_program();
    let frozen = vyre_driver::LaunchDirective::stated([64, 1, 1], [2, 1, 1], 0)
        .expect("the stated fixture launch is positive");

    let overrides: [(&str, fn(&mut DispatchConfig)); 4] = [
        ("workgroup_override", |config| {
            config.workgroup_override = Some([64, 1, 1]);
        }),
        ("grid_override", |config| {
            config.grid_override = Some([2, 1, 1]);
        }),
        ("dispatch_elements", |config| {
            config.dispatch_elements = Some(128);
        }),
        ("dispatch_grid", |config| {
            config.dispatch_grid = Some([2, 1, 1]);
        }),
    ];

    for (field, state) in overrides {
        let mut config = DispatchConfig::default();
        config.launch = Some(frozen);
        state(&mut config);

        let err =
            vyre_driver::validation::validate_program_for_backend(&backend, &program, &config)
                .expect_err(
                    "Fix: a frozen launch beside an override must be rejected, not resolved.",
                );
        let msg = err.to_string();
        assert!(
            msg.contains(field),
            "the rejection must name the competing field; got: {msg}"
        );

        // The override alone, and the frozen launch alone, are both admissible.
        let mut alone = DispatchConfig::default();
        state(&mut alone);
        vyre_driver::validation::validate_program_for_backend(&backend, &program, &alone)
            .expect("Fix: a stated override alone must stay valid.");
    }

    let mut frozen_alone = DispatchConfig::default();
    frozen_alone.launch = Some(frozen);
    vyre_driver::validation::validate_program_for_backend(&backend, &program, &frozen_alone)
        .expect("Fix: a frozen launch alone must stay valid.");
}

/// WHY: backend axis limits are enforced against whatever grid is submitted, and
/// a frozen launch is submitted exactly. Reading only `grid_override` would let a
/// recorded grid past a limit reach the device driver.
#[test]
fn a_frozen_launch_grid_past_the_backend_axis_limit_is_rejected() {
    let backend = GridLimitBackend;
    let program = tiny_program();
    let mut config = DispatchConfig::default();
    config.launch = Some(
        vyre_driver::LaunchDirective::stated([1, 1, 1], [8, 1, 1], 0)
            .expect("the stated fixture launch is positive"),
    );

    let err = vyre_driver::validation::validate_program_for_backend(&backend, &program, &config)
        .expect_err("Fix: a frozen grid above the backend axis limit must fail.");
    let msg = err.to_string();
    assert!(
        msg.contains("axis 0") && msg.contains("max is 7"),
        "the rejection must name the failing axis and the limit; got: {msg}"
    );
}
