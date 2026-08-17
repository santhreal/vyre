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
