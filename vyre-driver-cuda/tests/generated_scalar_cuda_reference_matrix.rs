//! Generated live CUDA/reference differential matrix for scalar IR semantics.

mod harness;

use harness::{
    assert_u32_matrix_sweep, bool_bytes, bool_word, compare_word, eq_word, ge_word,
    generated_lane_program, gt_word, guarded_generated_store, le_word, live_backend, lt_word,
    ne_word, u32_bytes, GeneratedMatrixCase, GENERATED_LANE_COUNT as LANE_COUNT,
};
use vyre_foundation::ir::{DataType, Expr, Program};

const ADVERSARIAL_SEEDS: &[u32] = &[
    0,
    1,
    2,
    3,
    7,
    31,
    32,
    63,
    127,
    128,
    255,
    256,
    1023,
    1024,
    0x7fff,
    0x8000,
    0xffff,
    0x1_0000,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_fffe,
    0xffff_ffff,
    0x5555_5555,
    0xaaaa_aaaa,
    0x0123_4567,
    0x89ab_cdef,
    0xfedc_ba98,
];

#[derive(Clone, Copy)]
struct BinaryCase {
    name: &'static str,
    rhs: RhsKind,
    build: fn(Expr, Expr) -> Expr,
}

/// A binary case whose right operand needs no adversarial shaping, which is
/// every Bool case: both operands come from the same predicate corpus.
#[derive(Clone, Copy)]
struct BoolBinaryCase {
    name: &'static str,
    build: fn(Expr, Expr) -> Expr,
}

#[derive(Clone, Copy)]
struct UnaryCase {
    name: &'static str,
    build: fn(Expr) -> Expr,
}

#[derive(Clone, Copy)]
enum RhsKind {
    Mixed,
    Divisor,
    Shift,
}

fn bool_and_word(left: Expr, right: Expr) -> Expr {
    compare_word(left, right, Expr::and)
}

fn bool_or_word(left: Expr, right: Expr) -> Expr {
    compare_word(left, right, Expr::or)
}

fn bool_not_word(value: Expr) -> Expr {
    bool_word(Expr::not(value))
}

const BINARY_CASES: &[BinaryCase] = &[
    BinaryCase {
        name: "add",
        rhs: RhsKind::Mixed,
        build: Expr::add,
    },
    BinaryCase {
        name: "sub",
        rhs: RhsKind::Mixed,
        build: Expr::sub,
    },
    BinaryCase {
        name: "mul",
        rhs: RhsKind::Mixed,
        build: Expr::mul,
    },
    BinaryCase {
        name: "div_total",
        rhs: RhsKind::Divisor,
        build: Expr::div,
    },
    BinaryCase {
        name: "mod_total",
        rhs: RhsKind::Divisor,
        build: Expr::rem,
    },
    BinaryCase {
        name: "mulhi",
        rhs: RhsKind::Mixed,
        build: Expr::mulhi,
    },
    BinaryCase {
        name: "abs_diff",
        rhs: RhsKind::Mixed,
        build: Expr::abs_diff,
    },
    BinaryCase {
        name: "bitand",
        rhs: RhsKind::Mixed,
        build: Expr::bitand,
    },
    BinaryCase {
        name: "bitor",
        rhs: RhsKind::Mixed,
        build: Expr::bitor,
    },
    BinaryCase {
        name: "bitxor",
        rhs: RhsKind::Mixed,
        build: Expr::bitxor,
    },
    BinaryCase {
        name: "shl_masked",
        rhs: RhsKind::Shift,
        build: Expr::shl,
    },
    BinaryCase {
        name: "shr_masked",
        rhs: RhsKind::Shift,
        build: Expr::shr,
    },
    BinaryCase {
        name: "min",
        rhs: RhsKind::Mixed,
        build: Expr::min,
    },
    BinaryCase {
        name: "max",
        rhs: RhsKind::Mixed,
        build: Expr::max,
    },
    BinaryCase {
        name: "eq",
        rhs: RhsKind::Mixed,
        build: eq_word,
    },
    BinaryCase {
        name: "ne",
        rhs: RhsKind::Mixed,
        build: ne_word,
    },
    BinaryCase {
        name: "lt",
        rhs: RhsKind::Mixed,
        build: lt_word,
    },
    BinaryCase {
        name: "le",
        rhs: RhsKind::Mixed,
        build: le_word,
    },
    BinaryCase {
        name: "gt",
        rhs: RhsKind::Mixed,
        build: gt_word,
    },
    BinaryCase {
        name: "ge",
        rhs: RhsKind::Mixed,
        build: ge_word,
    },
];

const UNARY_CASES: &[UnaryCase] = &[
    UnaryCase {
        name: "bitnot",
        build: Expr::bitnot,
    },
    UnaryCase {
        name: "popcount",
        build: Expr::popcount,
    },
    UnaryCase {
        name: "clz",
        build: Expr::clz,
    },
    UnaryCase {
        name: "ctz",
        build: Expr::ctz,
    },
    UnaryCase {
        name: "reverse_bits",
        build: Expr::reverse_bits,
    },
];

const BOOL_BINARY_CASES: &[BoolBinaryCase] = &[
    BoolBinaryCase {
        name: "bool_and",
        build: bool_and_word,
    },
    BoolBinaryCase {
        name: "bool_or",
        build: bool_or_word,
    },
    BoolBinaryCase {
        name: "bool_eq",
        build: eq_word,
    },
    BoolBinaryCase {
        name: "bool_ne",
        build: ne_word,
    },
];

const BOOL_UNARY_CASES: &[UnaryCase] = &[UnaryCase {
    name: "bool_not",
    build: bool_not_word,
}];

#[test]
fn generated_binary_scalar_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let lhs = adversarial_values(0x1357_2468);

    assert_u32_matrix_sweep(
        &backend,
        "binary",
        "every adversarial lane active",
        BINARY_CASES.iter().map(|case| GeneratedMatrixCase {
            name: case.name,
            program: binary_program(case.build, DataType::U32),
            inputs: vec![
                u32_bytes(&lhs),
                u32_bytes(&adversarial_rhs(case.rhs, &lhs, 0x9e37_79b9)),
            ],
        }),
    );
}

#[test]
fn generated_unary_scalar_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let input = adversarial_values(0xfeed_babe);

    assert_u32_matrix_sweep(
        &backend,
        "unary",
        "every adversarial lane active",
        UNARY_CASES.iter().map(|case| GeneratedMatrixCase {
            name: case.name,
            program: unary_program(case.build, DataType::U32),
            inputs: vec![u32_bytes(&input)],
        }),
    );
}

#[test]
fn generated_bool_scalar_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let lhs = adversarial_bool_values(0x1357_2468);
    let rhs = adversarial_bool_values(0x9e37_79b9);

    // The two tables are swept separately so each proves its own lane coverage.
    // A combined total lets one table over-count and cover the other's shortfall.
    assert_u32_matrix_sweep(
        &backend,
        "bool scalar",
        "predicate ALU active",
        BOOL_BINARY_CASES.iter().map(|case| GeneratedMatrixCase {
            name: case.name,
            program: binary_program(case.build, DataType::Bool),
            inputs: vec![bool_bytes(&lhs), bool_bytes(&rhs)],
        }),
    );
    assert_u32_matrix_sweep(
        &backend,
        "bool scalar",
        "bool memory active",
        BOOL_UNARY_CASES.iter().map(|case| GeneratedMatrixCase {
            name: case.name,
            program: unary_program(case.build, DataType::Bool),
            inputs: vec![bool_bytes(&lhs)],
        }),
    );
}

/// `out[idx] = build(lhs[idx], rhs[idx])` over two `input_type` buffers.
fn binary_program(build: fn(Expr, Expr) -> Expr, input_type: DataType) -> Program {
    let idx = Expr::var("idx");
    let value = build(
        Expr::load("lhs", idx.clone()),
        Expr::load("rhs", idx.clone()),
    );
    generated_lane_program(
        &[("lhs", input_type.clone()), ("rhs", input_type)],
        DataType::U32,
        guarded_generated_store(value),
    )
}

/// `out[idx] = build(input[idx])` over one `input_type` buffer.
fn unary_program(build: fn(Expr) -> Expr, input_type: DataType) -> Program {
    let value = build(Expr::load("input", Expr::var("idx")));
    generated_lane_program(
        &[("input", input_type)],
        DataType::U32,
        guarded_generated_store(value),
    )
}

fn adversarial_values(salt: u32) -> Vec<u32> {
    (0..LANE_COUNT)
        .map(|lane| {
            let seed = ADVERSARIAL_SEEDS[lane % ADVERSARIAL_SEEDS.len()];
            let lane_word = lane as u32;
            let mixed = lane_word
                .wrapping_mul(0x9e37_79b9)
                .rotate_left((lane_word & 31) + 1)
                ^ salt.rotate_right(lane_word & 31);
            seed ^ mixed
        })
        .collect()
}

fn adversarial_rhs(kind: RhsKind, lhs: &[u32], salt: u32) -> Vec<u32> {
    let mixed = adversarial_values(salt);
    mixed
        .into_iter()
        .enumerate()
        .map(|(lane, value)| match kind {
            RhsKind::Mixed if lane % 11 == 0 => lhs[lane],
            RhsKind::Mixed => value,
            RhsKind::Divisor if lane % 13 == 0 => 0,
            RhsKind::Divisor if lane % 17 == 0 => 1,
            RhsKind::Divisor => value,
            RhsKind::Shift => value & 31,
        })
        .collect()
}

fn adversarial_bool_values(salt: u32) -> Vec<bool> {
    (0..LANE_COUNT)
        .map(|lane| {
            let lane = lane as u32;
            let mixed = lane.wrapping_mul(0x045d_9f3b).rotate_left((lane & 7) + 1)
                ^ salt.rotate_right(lane & 31);
            (mixed & 0b1011) == 0b0001 || lane % 17 == 0
        })
        .collect()
}
