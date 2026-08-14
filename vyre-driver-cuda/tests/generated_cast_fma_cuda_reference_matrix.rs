//! Generated live CUDA/reference differential matrix for cast and fused arithmetic semantics.

mod common;

use common::{
    assert_f32_matrix_sweep, assert_u32_matrix_sweep, bool_bytes, f32_bytes,
    generated_bool_cast_values, generated_f32_cast_values, generated_f32_fma_values,
    generated_i32_cast_values, generated_lane_program, generated_u32_cast_values,
    guarded_generated_store, i32_bytes, live_backend, u32_bytes, GeneratedMatrixCase,
    GENERATED_LANE_COUNT as LANE_COUNT,
};
use vyre_foundation::ir::{DataType, Expr, Program};

const MAX_F32_ULP: u32 = 1;

#[derive(Clone)]
struct CastCase {
    name: &'static str,
    input_type: DataType,
    output_type: DataType,
    input: CastInput,
    build: fn(Expr) -> Expr,
}

#[derive(Clone, Copy)]
enum CastInput {
    U32,
    I32,
    F32,
    Bool,
}

fn cast_to_u32(value: Expr) -> Expr {
    Expr::cast(DataType::U32, value)
}

fn cast_to_i32(value: Expr) -> Expr {
    Expr::cast(DataType::I32, value)
}

fn cast_to_f32(value: Expr) -> Expr {
    Expr::cast(DataType::F32, value)
}

fn cast_to_bool_word(value: Expr) -> Expr {
    Expr::select(
        Expr::cast(DataType::Bool, value),
        Expr::u32(1),
        Expr::u32(0),
    )
}

const CAST_CASES: &[CastCase] = &[
    CastCase {
        name: "cast_u32_to_i32",
        input_type: DataType::U32,
        output_type: DataType::I32,
        input: CastInput::U32,
        build: cast_to_i32,
    },
    CastCase {
        name: "cast_i32_to_u32",
        input_type: DataType::I32,
        output_type: DataType::U32,
        input: CastInput::I32,
        build: cast_to_u32,
    },
    CastCase {
        name: "cast_u32_to_f32_numeric",
        input_type: DataType::U32,
        output_type: DataType::F32,
        input: CastInput::U32,
        build: cast_to_f32,
    },
    CastCase {
        name: "cast_i32_to_f32_numeric",
        input_type: DataType::I32,
        output_type: DataType::F32,
        input: CastInput::I32,
        build: cast_to_f32,
    },
    CastCase {
        name: "cast_bool_to_u32",
        input_type: DataType::Bool,
        output_type: DataType::U32,
        input: CastInput::Bool,
        build: cast_to_u32,
    },
    CastCase {
        name: "cast_bool_to_i32",
        input_type: DataType::Bool,
        output_type: DataType::I32,
        input: CastInput::Bool,
        build: cast_to_i32,
    },
    CastCase {
        name: "cast_bool_to_f32_numeric",
        input_type: DataType::Bool,
        output_type: DataType::F32,
        input: CastInput::Bool,
        build: cast_to_f32,
    },
    CastCase {
        name: "cast_f32_to_u32",
        input_type: DataType::F32,
        output_type: DataType::U32,
        input: CastInput::F32,
        build: cast_to_u32,
    },
    CastCase {
        name: "cast_f32_to_i32",
        input_type: DataType::F32,
        output_type: DataType::I32,
        input: CastInput::F32,
        build: cast_to_i32,
    },
    CastCase {
        name: "cast_u32_to_bool_word",
        input_type: DataType::U32,
        output_type: DataType::U32,
        input: CastInput::U32,
        build: cast_to_bool_word,
    },
    CastCase {
        name: "cast_i32_to_bool_word",
        input_type: DataType::I32,
        output_type: DataType::U32,
        input: CastInput::I32,
        build: cast_to_bool_word,
    },
    CastCase {
        name: "cast_f32_to_bool_word",
        input_type: DataType::F32,
        output_type: DataType::U32,
        input: CastInput::F32,
        build: cast_to_bool_word,
    },
];

/// The four cast source corpora, one per input dtype.
struct CastCorpora {
    words: Vec<u32>,
    signed: Vec<i32>,
    floats: Vec<f32>,
    predicates: Vec<bool>,
}

impl CastCorpora {
    fn generate() -> Self {
        Self {
            words: generated_u32_cast_values(LANE_COUNT),
            signed: generated_i32_cast_values(LANE_COUNT),
            floats: generated_f32_cast_values(LANE_COUNT),
            predicates: generated_bool_cast_values(LANE_COUNT),
        }
    }

    fn bytes(&self, input: CastInput) -> Vec<u8> {
        match input {
            CastInput::U32 => u32_bytes(&self.words),
            CastInput::I32 => i32_bytes(&self.signed),
            CastInput::F32 => f32_bytes(&self.floats),
            CastInput::Bool => bool_bytes(&self.predicates),
        }
    }
}

#[test]
fn generated_cast_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let corpora = CastCorpora::generate();
    // The output dtype picks the comparison, so the table is swept in two
    // groups: f32 results under the arithmetic ULP bound, every other result
    // bit-exact. Each group asserts its own lane coverage, so neither can hide
    // a shortfall in the other behind a combined total.
    let (float_results, word_results): (Vec<&CastCase>, Vec<&CastCase>) = CAST_CASES
        .iter()
        .partition(|case| matches!(case.output_type, DataType::F32));

    assert_u32_matrix_sweep(
        &backend,
        "cast",
        "every lane active",
        word_results.into_iter().map(|case| GeneratedMatrixCase {
            name: case.name,
            program: cast_program(case),
            inputs: vec![corpora.bytes(case.input)],
        }),
    );
    assert_f32_matrix_sweep(
        &backend,
        "cast",
        "every lane active",
        MAX_F32_ULP,
        float_results.into_iter().map(|case| GeneratedMatrixCase {
            name: case.name,
            program: cast_program(case),
            inputs: vec![corpora.bytes(case.input)],
        }),
    );
}

#[test]
fn generated_f32_fma_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let a = generated_f32_fma_values(LANE_COUNT, 0x1234_5678);
    let b = generated_f32_fma_values(LANE_COUNT, 0x9abc_def0);
    let c = generated_f32_fma_values(LANE_COUNT, 0x0fed_cba9);

    assert_f32_matrix_sweep(
        &backend,
        "f32 fma",
        "every lane active",
        MAX_F32_ULP,
        [GeneratedMatrixCase {
            name: "f32_fma",
            program: f32_fma_program(),
            inputs: vec![f32_bytes(&a), f32_bytes(&b), f32_bytes(&c)],
        }]
        .into_iter(),
    );
}

/// `out[idx] = build(input[idx])` for one cast case.
fn cast_program(case: &CastCase) -> Program {
    let value = (case.build)(Expr::load("input", Expr::var("idx")));
    generated_lane_program(
        &[("input", case.input_type.clone())],
        case.output_type.clone(),
        guarded_generated_store(value),
    )
}

/// `out[idx] = fma(a[idx], b[idx], c[idx])`, the single-rounding contract the
/// reference interpreter and both CUDA paths have to agree on.
fn f32_fma_program() -> Program {
    let idx = Expr::var("idx");
    let value = Expr::fma(
        Expr::load("a", idx.clone()),
        Expr::load("b", idx.clone()),
        Expr::load("c", idx),
    );
    generated_lane_program(
        &[
            ("a", DataType::F32),
            ("b", DataType::F32),
            ("c", DataType::F32),
        ],
        DataType::F32,
        guarded_generated_store(value),
    )
}
