//! Generated live CUDA/reference differential matrix for data-dependent control semantics.

mod common;
#[path = "common/generated_control_values.rs"]
mod generated_control_values;

use common::{
    assert_f32_matrix_sweep, assert_u32_matrix_sweep, bool_bytes, f32_bytes,
    generated_lane_program, generated_mixed_bool_values as generated_bool_values,
    guarded_generated_store, i32_bytes, live_backend, u32_bytes, GeneratedMatrixCase,
    GENERATED_LANE_COUNT as LANE_COUNT,
};
use generated_control_values::{
    generated_f32_values, generated_i32_values, generated_u32_values, MAX_F32_ULP,
};
use vyre_foundation::ir::{DataType, Expr, Node, Program};

/// A select case: `out[idx] = build(flag[idx], lhs[idx], rhs[idx])`. The operand
/// and flag types come from the sweep, so one table shape covers every dtype.
#[derive(Clone, Copy)]
struct SelectCase {
    name: &'static str,
    build: fn(Expr, Expr, Expr) -> Expr,
}

fn u32_select_eq_flag(flag: Expr, lhs: Expr, rhs: Expr) -> Expr {
    Expr::select(Expr::eq(flag, Expr::u32(0)), lhs, rhs)
}

fn u32_select_lt_min(_flag: Expr, lhs: Expr, rhs: Expr) -> Expr {
    Expr::select(Expr::lt(lhs.clone(), rhs.clone()), lhs, rhs)
}

fn u32_select_bit_mixed(flag: Expr, lhs: Expr, rhs: Expr) -> Expr {
    Expr::select(
        Expr::ne(Expr::bitand(flag, Expr::u32(1)), Expr::u32(0)),
        Expr::bitxor(lhs.clone(), rhs.clone()),
        Expr::add(lhs, rhs),
    )
}

fn u32_select_nested(flag: Expr, lhs: Expr, rhs: Expr) -> Expr {
    let low_flag = Expr::lt(flag.clone(), Expr::u32(0x8000_0000));
    let low_value = Expr::select(Expr::lt(lhs.clone(), rhs.clone()), lhs.clone(), rhs.clone());
    let high_value = Expr::select(
        Expr::gt(lhs.clone(), rhs.clone()),
        Expr::sub(lhs, rhs),
        flag,
    );
    Expr::select(low_flag, low_value, high_value)
}

fn i32_select_lt_min(_flag: Expr, lhs: Expr, rhs: Expr) -> Expr {
    Expr::select(Expr::lt(lhs.clone(), rhs.clone()), lhs, rhs)
}

fn i32_select_ge_delta(_flag: Expr, lhs: Expr, rhs: Expr) -> Expr {
    Expr::select(
        Expr::ge(lhs.clone(), rhs.clone()),
        Expr::sub(lhs.clone(), rhs.clone()),
        Expr::add(lhs, rhs),
    )
}

fn i32_select_flag_sign(flag: Expr, lhs: Expr, rhs: Expr) -> Expr {
    Expr::select(
        Expr::lt(flag, Expr::i32(0)),
        Expr::sub(Expr::i32(0), lhs),
        rhs,
    )
}

fn f32_select_nan(_flag: Expr, lhs: Expr, rhs: Expr) -> Expr {
    Expr::select(Expr::is_nan(lhs.clone()), rhs, lhs)
}

fn f32_select_finite(_flag: Expr, lhs: Expr, rhs: Expr) -> Expr {
    Expr::select(
        Expr::is_finite(lhs.clone()),
        Expr::add(lhs, rhs.clone()),
        rhs,
    )
}

fn f32_select_ordered(_flag: Expr, lhs: Expr, rhs: Expr) -> Expr {
    Expr::select(
        Expr::lt(lhs.clone(), rhs.clone()),
        Expr::mul(lhs.clone(), rhs.clone()),
        Expr::sub(lhs, rhs),
    )
}

fn bool_select_flag(flag: Expr, lhs: Expr, rhs: Expr) -> Expr {
    Expr::select(flag, lhs, rhs)
}

fn bool_select_eq(flag: Expr, lhs: Expr, rhs: Expr) -> Expr {
    Expr::select(Expr::eq(lhs.clone(), rhs), flag, lhs)
}

fn bool_select_nested(flag: Expr, lhs: Expr, rhs: Expr) -> Expr {
    let then_value = Expr::select(lhs.clone(), flag.clone(), rhs.clone());
    let else_value = Expr::select(rhs, lhs, flag.clone());
    Expr::select(flag, then_value, else_value)
}

const U32_SELECT_CASES: &[SelectCase] = &[
    SelectCase {
        name: "u32_select_eq_flag",
        build: u32_select_eq_flag,
    },
    SelectCase {
        name: "u32_select_lt_min",
        build: u32_select_lt_min,
    },
    SelectCase {
        name: "u32_select_bit_mixed",
        build: u32_select_bit_mixed,
    },
    SelectCase {
        name: "u32_select_nested",
        build: u32_select_nested,
    },
];

const I32_SELECT_CASES: &[SelectCase] = &[
    SelectCase {
        name: "i32_select_lt_min",
        build: i32_select_lt_min,
    },
    SelectCase {
        name: "i32_select_ge_delta",
        build: i32_select_ge_delta,
    },
    SelectCase {
        name: "i32_select_flag_sign",
        build: i32_select_flag_sign,
    },
];

const F32_SELECT_CASES: &[SelectCase] = &[
    SelectCase {
        name: "f32_select_nan",
        build: f32_select_nan,
    },
    SelectCase {
        name: "f32_select_finite",
        build: f32_select_finite,
    },
    SelectCase {
        name: "f32_select_ordered",
        build: f32_select_ordered,
    },
];

const BOOL_SELECT_CASES: &[SelectCase] = &[
    SelectCase {
        name: "bool_select_flag",
        build: bool_select_flag,
    },
    SelectCase {
        name: "bool_select_eq",
        build: bool_select_eq,
    },
    SelectCase {
        name: "bool_select_nested",
        build: bool_select_nested,
    },
];

#[test]
fn generated_u32_select_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let flag = generated_u32_values(0x1020_3040);
    let lhs = generated_u32_values(0xa5a5_5a5a);
    let rhs = generated_u32_values(0x5a5a_a5a5);

    assert_u32_matrix_sweep(
        &backend,
        "u32 select",
        "every adversarial lane active",
        U32_SELECT_CASES.iter().map(|case| GeneratedMatrixCase {
            name: case.name,
            program: select_program(case, DataType::U32, DataType::U32),
            inputs: vec![u32_bytes(&flag), u32_bytes(&lhs), u32_bytes(&rhs)],
        }),
    );
}

#[test]
fn generated_i32_select_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let flag = generated_i32_values(0x1020_3040);
    let lhs = generated_i32_values(0x1357_9bdf);
    let rhs = generated_i32_values(0xfdb9_7531);

    assert_u32_matrix_sweep(
        &backend,
        "i32 select",
        "every adversarial lane active",
        I32_SELECT_CASES.iter().map(|case| GeneratedMatrixCase {
            name: case.name,
            program: select_program(case, DataType::I32, DataType::I32),
            inputs: vec![i32_bytes(&flag), i32_bytes(&lhs), i32_bytes(&rhs)],
        }),
    );
}

#[test]
fn generated_f32_select_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let flag = generated_u32_values(0x3333_cccc);
    let lhs = generated_f32_values(0x1234_abcd);
    let rhs = generated_f32_values(0xdcba_4321);

    assert_f32_matrix_sweep(
        &backend,
        "f32 select",
        "every adversarial lane active",
        MAX_F32_ULP,
        F32_SELECT_CASES.iter().map(|case| GeneratedMatrixCase {
            name: case.name,
            program: select_program(case, DataType::U32, DataType::F32),
            inputs: vec![u32_bytes(&flag), f32_bytes(&lhs), f32_bytes(&rhs)],
        }),
    );
}

#[test]
fn generated_bool_select_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let flag = generated_bool_values(0x3333_cccc);
    let lhs = generated_bool_values(0x1234_abcd);
    let rhs = generated_bool_values(0xdcba_4321);

    assert_u32_matrix_sweep(
        &backend,
        "bool select",
        "predicate select and bool output storage active",
        BOOL_SELECT_CASES.iter().map(|case| GeneratedMatrixCase {
            name: case.name,
            program: select_program(case, DataType::Bool, DataType::Bool),
            inputs: vec![bool_bytes(&flag), bool_bytes(&lhs), bool_bytes(&rhs)],
        }),
    );
}

#[test]
fn generated_data_dependent_if_then_overwrite_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let flag = generated_u32_values(0xface_cafe);
    let lhs = generated_u32_values(0x0123_4567);
    let rhs = generated_u32_values(0x89ab_cdef);

    assert_u32_matrix_sweep(
        &backend,
        "if_then",
        "every adversarial lane active",
        [GeneratedMatrixCase {
            name: "u32_if_then_overwrite",
            program: u32_if_then_overwrite_program(),
            inputs: vec![u32_bytes(&flag), u32_bytes(&lhs), u32_bytes(&rhs)],
        }]
        .into_iter(),
    );
}

/// `out[idx] = build(flag[idx], lhs[idx], rhs[idx])`, with the output taking the
/// operand type: a select cannot change the type of the value it chooses.
fn select_program(case: &SelectCase, flag_type: DataType, operand_type: DataType) -> Program {
    let idx = Expr::var("idx");
    let value = (case.build)(
        Expr::load("flag", idx.clone()),
        Expr::load("lhs", idx.clone()),
        Expr::load("rhs", idx),
    );
    generated_lane_program(
        &[
            ("flag", flag_type),
            ("lhs", operand_type.clone()),
            ("rhs", operand_type.clone()),
        ],
        operand_type,
        guarded_generated_store(value),
    )
}

/// `out[idx] = rhs[idx]`, then overwritten with `lhs[idx]` on odd-flag lanes.
/// The nested store is the contract: a lowering that hoists or reorders the
/// inner `if_then` leaves the first value behind.
fn u32_if_then_overwrite_program() -> Program {
    generated_lane_program(
        &[
            ("flag", DataType::U32),
            ("lhs", DataType::U32),
            ("rhs", DataType::U32),
        ],
        DataType::U32,
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::u32(LANE_COUNT as u32)),
                vec![
                    Node::store("out", Expr::var("idx"), Expr::load("rhs", Expr::var("idx"))),
                    Node::if_then(
                        Expr::ne(
                            Expr::bitand(Expr::load("flag", Expr::var("idx")), Expr::u32(1)),
                            Expr::u32(0),
                        ),
                        vec![Node::store(
                            "out",
                            Expr::var("idx"),
                            Expr::load("lhs", Expr::var("idx")),
                        )],
                    ),
                ],
            ),
        ],
    )
}
