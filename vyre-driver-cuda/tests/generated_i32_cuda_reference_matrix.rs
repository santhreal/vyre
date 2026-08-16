//! Generated live CUDA/reference differential matrix for signed i32 IR semantics.

mod harness;

use harness::{
    assert_u32_matrix_sweep, eq_word, ge_word, generated_lane_program, gt_word,
    guarded_generated_store, i32_bytes, le_word, live_backend, lt_word, ne_word,
    GeneratedMatrixCase, GENERATED_LANE_COUNT as LANE_COUNT,
};
use vyre_foundation::ir::{DataType, Expr, Program};

const ADVERSARIAL_I32_SEEDS: &[i32] = &[
    0,
    1,
    -1,
    2,
    -2,
    3,
    -3,
    7,
    -7,
    31,
    -31,
    32,
    -32,
    127,
    -127,
    128,
    -128,
    255,
    -255,
    1024,
    -1024,
    i16::MAX as i32,
    i16::MIN as i32,
    i32::MAX,
    i32::MIN,
    0x5555_5555,
    0x2aaa_aaaa,
    0x0123_4567,
    -0x0123_4567,
];

#[derive(Clone)]
struct I32BinaryCase {
    name: &'static str,
    rhs: I32RhsKind,
    output: DataType,
    build: fn(Expr, Expr) -> Expr,
}

#[derive(Clone)]
struct I32UnaryCase {
    name: &'static str,
    output: DataType,
    build: fn(Expr) -> Expr,
}

#[derive(Clone, Copy)]
enum I32RhsKind {
    Mixed,
    DefinedDivisor,
}

fn wrapping_negate_i32(value: Expr) -> Expr {
    Expr::sub(Expr::i32(0), value)
}

const I32_BINARY_CASES: &[I32BinaryCase] = &[
    I32BinaryCase {
        name: "i32_add",
        rhs: I32RhsKind::Mixed,
        output: DataType::I32,
        build: Expr::add,
    },
    I32BinaryCase {
        name: "i32_sub",
        rhs: I32RhsKind::Mixed,
        output: DataType::I32,
        build: Expr::sub,
    },
    I32BinaryCase {
        name: "i32_mul",
        rhs: I32RhsKind::Mixed,
        output: DataType::I32,
        build: Expr::mul,
    },
    I32BinaryCase {
        name: "i32_div_defined",
        rhs: I32RhsKind::DefinedDivisor,
        output: DataType::I32,
        build: Expr::div,
    },
    I32BinaryCase {
        name: "i32_mod_defined",
        rhs: I32RhsKind::DefinedDivisor,
        output: DataType::U32,
        build: Expr::rem,
    },
    I32BinaryCase {
        name: "i32_bitand",
        rhs: I32RhsKind::Mixed,
        output: DataType::U32,
        build: Expr::bitand,
    },
    I32BinaryCase {
        name: "i32_bitor",
        rhs: I32RhsKind::Mixed,
        output: DataType::U32,
        build: Expr::bitor,
    },
    I32BinaryCase {
        name: "i32_bitxor",
        rhs: I32RhsKind::Mixed,
        output: DataType::U32,
        build: Expr::bitxor,
    },
    I32BinaryCase {
        name: "i32_min",
        rhs: I32RhsKind::Mixed,
        output: DataType::I32,
        build: Expr::min,
    },
    I32BinaryCase {
        name: "i32_max",
        rhs: I32RhsKind::Mixed,
        output: DataType::I32,
        build: Expr::max,
    },
    I32BinaryCase {
        name: "i32_eq",
        rhs: I32RhsKind::Mixed,
        output: DataType::U32,
        build: eq_word,
    },
    I32BinaryCase {
        name: "i32_ne",
        rhs: I32RhsKind::Mixed,
        output: DataType::U32,
        build: ne_word,
    },
    I32BinaryCase {
        name: "i32_lt",
        rhs: I32RhsKind::Mixed,
        output: DataType::U32,
        build: lt_word,
    },
    I32BinaryCase {
        name: "i32_le",
        rhs: I32RhsKind::Mixed,
        output: DataType::U32,
        build: le_word,
    },
    I32BinaryCase {
        name: "i32_gt",
        rhs: I32RhsKind::Mixed,
        output: DataType::U32,
        build: gt_word,
    },
    I32BinaryCase {
        name: "i32_ge",
        rhs: I32RhsKind::Mixed,
        output: DataType::U32,
        build: ge_word,
    },
];

const I32_UNARY_CASES: &[I32UnaryCase] = &[
    I32UnaryCase {
        name: "i32_negate",
        output: DataType::I32,
        build: wrapping_negate_i32,
    },
    I32UnaryCase {
        name: "i32_bitnot",
        output: DataType::I32,
        build: Expr::bitnot,
    },
    I32UnaryCase {
        name: "i32_popcount",
        output: DataType::I32,
        build: Expr::popcount,
    },
    I32UnaryCase {
        name: "i32_clz",
        output: DataType::I32,
        build: Expr::clz,
    },
    I32UnaryCase {
        name: "i32_ctz",
        output: DataType::I32,
        build: Expr::ctz,
    },
    I32UnaryCase {
        name: "i32_reverse_bits",
        output: DataType::I32,
        build: Expr::reverse_bits,
    },
];

#[test]
fn generated_i32_binary_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let lhs = adversarial_i32_values(0x3141_5926);

    assert_u32_matrix_sweep(
        &backend,
        "i32 binary",
        "every adversarial lane active",
        I32_BINARY_CASES.iter().map(|case| GeneratedMatrixCase {
            name: case.name,
            program: i32_binary_program(case),
            inputs: vec![
                i32_bytes(&lhs),
                i32_bytes(&adversarial_i32_rhs(case.rhs, &lhs, 0x2718_2818)),
            ],
        }),
    );
}

#[test]
fn generated_i32_unary_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let input = adversarial_i32_values(0x1618_0339);

    assert_u32_matrix_sweep(
        &backend,
        "i32 unary",
        "every adversarial lane active",
        I32_UNARY_CASES.iter().map(|case| GeneratedMatrixCase {
            name: case.name,
            program: i32_unary_program(case),
            inputs: vec![i32_bytes(&input)],
        }),
    );
}

/// `out[idx] = build(lhs[idx], rhs[idx])` over two i32 buffers.
fn i32_binary_program(case: &I32BinaryCase) -> Program {
    let idx = Expr::var("idx");
    let value = (case.build)(
        Expr::load("lhs", idx.clone()),
        Expr::load("rhs", idx.clone()),
    );
    generated_lane_program(
        &[("lhs", DataType::I32), ("rhs", DataType::I32)],
        case.output.clone(),
        guarded_generated_store(value),
    )
}

/// `out[idx] = build(input[idx])` over one i32 buffer.
fn i32_unary_program(case: &I32UnaryCase) -> Program {
    let value = (case.build)(Expr::load("input", Expr::var("idx")));
    generated_lane_program(
        &[("input", DataType::I32)],
        case.output.clone(),
        guarded_generated_store(value),
    )
}

fn adversarial_i32_values(salt: u32) -> Vec<i32> {
    (0..LANE_COUNT)
        .map(|lane| {
            let seed = ADVERSARIAL_I32_SEEDS[lane % ADVERSARIAL_I32_SEEDS.len()] as u32;
            let lane_word = lane as u32;
            let mixed = lane_word
                .wrapping_mul(0x9e37_79b9)
                .rotate_left((lane_word & 31) + 1)
                ^ salt.rotate_right(lane_word & 31);
            (seed ^ mixed) as i32
        })
        .collect()
}

fn adversarial_i32_rhs(kind: I32RhsKind, lhs: &[i32], salt: u32) -> Vec<i32> {
    adversarial_i32_values(salt)
        .into_iter()
        .enumerate()
        .map(|(lane, value)| match kind {
            I32RhsKind::Mixed if lane % 11 == 0 => lhs[lane],
            I32RhsKind::Mixed => value,
            I32RhsKind::DefinedDivisor => defined_i32_divisor(lhs[lane], value, lane),
        })
        .collect()
}

fn defined_i32_divisor(lhs: i32, value: i32, lane: usize) -> i32 {
    let candidate = if value == 0 {
        (lane as i32 % 31) + 1
    } else {
        value
    };
    if lhs == i32::MIN && candidate == -1 {
        1
    } else {
        candidate
    }
}
