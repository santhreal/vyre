use vyre_foundation::operation::OperationRegistration;

macro_rules! bitset_and_entry {
    ($module:ident, $build:expr) => {
        inventory::submit! {
            OperationRegistration::library(
                super::$module::OP_ID,
                $build,
                Some(|| {
                    vec![vec![
                        vec![12, 0, 0, 0],
                        vec![10, 0, 0, 0],
                    ]]
                }),
                Some(|| {
                    vec![vec![vec![8, 0, 0, 0]]]
                }),
            )
            .with_category("security")
        }
    };
}

macro_rules! bitset_and_not_entry {
    ($module:ident, $build:expr) => {
        inventory::submit! {
            OperationRegistration::library(
                super::$module::OP_ID,
                $build,
                Some(|| {
                    vec![vec![
                        vec![15, 0, 0, 0],
                        vec![12, 0, 0, 0],
                    ]]
                }),
                Some(|| {
                    vec![vec![vec![3, 0, 0, 0]]]
                }),
            )
            .with_category("security")
        }
    };
}

bitset_and_entry!(auth_check_dominates, || {
    super::auth_check_dominates::auth_check_dominates(4, "a", "b", "out")
});
bitset_and_entry!(buffer_size_check, || {
    super::buffer_size_check::buffer_size_check(4, "a", "b", "out")
});
bitset_and_entry!(lock_dominates, || {
    super::lock_dominates::lock_dominates(4, "a", "b", "out")
});
bitset_and_entry!(path_canonical, || {
    super::path_canonical::path_canonical(4, "a", "b", "out")
});
bitset_and_entry!(sanitizer_dominates, || {
    super::sanitizer_dominates::sanitizer_dominates(4, "a", "b", "out")
});
bitset_and_entry!(sql_param_bound, || {
    super::sql_param_bound::sql_param_bound(4, "a", "b", "out")
});
bitset_and_entry!(xss_escape, || {
    super::xss_escape::xss_escape(4, "a", "b", "out")
});

bitset_and_not_entry!(format_string_check, || {
    super::format_string_check::format_string_check(4, "a", "b", "out")
});
bitset_and_not_entry!(taint_kill, || {
    super::taint_kill::taint_kill(4, "a", "b", "out")
});
bitset_and_not_entry!(unchecked_return, || {
    super::unchecked_return::unchecked_return(4, "a", "b", "out")
});

inventory::submit! {
    OperationRegistration::library(
        super::sink_intersection::OP_ID,
        || super::sink_intersection::sink_intersection(4, "a", "b", "scratch", "out"),
        Some(|| vec![vec![
            vec![12, 0, 0, 0],
            vec![10, 0, 0, 0],
        ]]),
        Some(|| vec![vec![
            vec![8, 0, 0, 0],
            vec![1, 0, 0, 0],
        ]]),
    )
    .with_category("security")
}

inventory::submit! {
    OperationRegistration::library(
        super::integer_overflow_arith::OP_ID,
        || {
            super::integer_overflow_arith::integer_overflow_arith(
                4, "arith", "reach", "guards", "scratch", "out",
            )
        },
        Some(|| vec![vec![
            vec![15, 0, 0, 0],
            vec![12, 0, 0, 0],
            vec![8, 0, 0, 0],
        ]]),
        Some(|| vec![vec![
            vec![12, 0, 0, 0],
            vec![4, 0, 0, 0],
        ]]),
    )
    .with_category("security")
}
macro_rules! reach_flow_entry {
    ($op_id:expr, $build:expr, $inputs_fn:expr, $expected_fn:expr) => {
        inventory::submit! {
            OperationRegistration::library(
                $op_id,
                $build,
                Some($inputs_fn),
                Some($expected_fn),
            )
            .with_category("security")
        }
        inventory::submit! {
            crate::operation_catalog::ConvergenceContract {
                op_id: $op_id,
                max_iterations: super::flow_composition::FLOW_MAX_ITERATIONS,
            }
        }
    };
}

reach_flow_entry!(
    super::flows_to::OP_ID,
    || super::flows_to::flows_to(
        crate::graph::program_graph::ProgramGraphShape::new(4, 3),
        "fin",
        "fout"
    ),
    super::flow_composition::forward_reach_fixture_inputs,
    super::flow_composition::forward_reach_fixture_expected
);

reach_flow_entry!(
    super::taint_flow::OP_ID,
    || super::taint_flow::taint_flow(
        crate::graph::program_graph::ProgramGraphShape::new(4, 3),
        "fin",
        "fout"
    ),
    super::flow_composition::forward_reach_fixture_inputs,
    super::flow_composition::forward_reach_fixture_expected
);

reach_flow_entry!(
    super::bounded_by_comparison::OP_ID,
    || super::bounded_by_comparison::bounded_by_comparison(
        crate::graph::program_graph::ProgramGraphShape::new(4, 4),
        "fin",
        "fout"
    ),
    super::flow_composition::dominance_fixture_inputs,
    super::flow_composition::dominance_fixture_expected
);

reach_flow_entry!(
    super::dominance_predecessors::OP_ID,
    || super::dominance_predecessors::dominance_predecessors(
        crate::graph::program_graph::ProgramGraphShape::new(4, 4),
        "fin",
        "fout"
    ),
    super::flow_composition::dominance_fixture_inputs,
    super::flow_composition::dominance_fixture_expected
);

inventory::submit! {
    OperationRegistration::library(
        super::flows_to_to_sink::OP_ID,
        || super::flows_to_to_sink::flows_to_to_sink(crate::graph::program_graph::ProgramGraphShape::new(4, 3), "source", "sink", "reach", "hits", "out_scalar"),
        Some(super::flow_composition::dataflow_hit_fixture_inputs),
        Some(super::flow_composition::dataflow_hit_fixture_expected),
    )
    .with_category("security")
}

inventory::submit! {
    OperationRegistration::library(
        super::taint_pollution::OP_ID,
        || super::taint_pollution::taint_pollution(crate::graph::program_graph::ProgramGraphShape::new(4, 3), "source", "label_set", "reach", "hits", "out_scalar"),
        Some(super::flow_composition::dataflow_hit_fixture_inputs),
        Some(super::flow_composition::dataflow_hit_fixture_expected),
    )
    .with_category("security")
}

inventory::submit! {
    OperationRegistration::library(
        super::sanitized_by::OP_ID,
        || super::sanitized_by::sanitized_by(crate::graph::program_graph::ProgramGraphShape::new(4, 3), "fin", "san", "fout"),
        Some(super::sanitized_by::sanitized_by_fixture_inputs),
        Some(super::sanitized_by::sanitized_by_fixture_expected),
    )
    .with_category("security")
}

inventory::submit! {
    crate::operation_catalog::ConvergenceContract {
        op_id: super::sanitized_by::OP_ID,
        max_iterations: 4096,
    }
}

inventory::submit! {
    OperationRegistration::library(
        super::flows_to_with_sanitizer::OP_ID,
        || super::flows_to_with_sanitizer::flows_to_with_sanitizer(crate::graph::program_graph::ProgramGraphShape::new(4, 3), "source", "sink", "sanitizer", "clean", "reach", "alive", "hits", "out_scalar"),
        Some(super::flows_to_with_sanitizer::flows_to_with_sanitizer_fixture_inputs),
        Some(super::flows_to_with_sanitizer::flows_to_with_sanitizer_fixture_expected),
    )
    .with_category("security")
}
