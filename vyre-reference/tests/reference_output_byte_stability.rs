//! Byte-stability golden for what the reference backend computes.
//!
//! The reference interpreter is the oracle every backend is diffed against, so
//! a refactor inside it is only safe if the bytes it returns do not move. This
//! pins two surfaces:
//!
//! - every registered dual-reference facet, over a fixed hostile input set;
//! - a fixed `Program` corpus run through [`vyre_reference::reference_eval`].
//!
//! The facet section is enumerated from the registry at run time, so adding a
//! dual reference turns this red until the golden records what it computes.
//!
//! The section format, the comparison, and the hex rendering live in
//! `vyre_lower::artifact_golden`. This file supplies only the reference corpus.

#![forbid(unsafe_code)]

use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_lower::artifact_golden::{assert_matches_golden, hex_words, write_golden};
use vyre_reference::value::Value;
use vyre_reference::{dual_op_ids, reference_eval, resolve_dual};

/// Line that opens each case's section, matching `artifact_golden`.
const CASE_MARKER: &str = "===== ";

/// Header written above the golden, matching `artifact_golden`.
const HEADER: &str = "\
# Emitted-artifact byte-stability golden.
#
# One section per shared success-corpus case, in corpus order. Regenerate with
# the `bless_*` test in the file that reads this golden, then review the diff:
# a change here is a change in what the backend emits.
";

/// Fixed hostile seeds every dual facet is evaluated over.
const FACET_SEEDS: [u32; 8] = [
    0,
    1,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_ffff,
    0x0000_ffff,
    0xdead_beef,
    0x0f0f_0f0f,
];

/// Lane count shared by every program case.
const LANES: usize = 8;

/// Workgroup width shared by every program case.
const WORKGROUP_SIZE_X: u32 = 8;

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/reference_outputs.txt")
}

/// Widen one seed into the byte input a dual facet consumes.
///
/// Facets read a fixed-width prefix, so the widest supported operand set is
/// supplied once and narrower facets ignore the tail.
fn facet_input(seed: u32) -> Vec<u8> {
    let left = seed.wrapping_mul(0x85eb_ca6b).rotate_left((seed ^ 0x13) & 31);
    let right = seed
        .wrapping_mul(0xc2b2_ae35)
        .rotate_right((seed ^ 0x29) & 31);
    let mut input = Vec::with_capacity(48);
    for word in [left, right, left ^ right, left.wrapping_add(right)] {
        input.extend_from_slice(&word.to_le_bytes());
    }
    for word in [
        (u64::from(left) << 32) | u64::from(right),
        (u64::from(right) << 32) | u64::from(left),
    ] {
        input.extend_from_slice(&word.to_le_bytes());
    }
    input.extend_from_slice(&f32::from_bits(left).to_le_bytes());
    input.extend_from_slice(&f32::from_bits(right).to_le_bytes());
    input
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Hostile u32 lane values reused by the integer program cases.
const U32_LANES: [u32; LANES] = [
    0,
    1,
    2,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_ffff,
    0xdead_beef,
    0x0f0f_0f0f,
];

/// Hostile i32 lane values, including both signed extremes.
const I32_LANES: [i32; LANES] = [0, 1, -1, i32::MAX, i32::MIN, -7, 12_345, -12_345];

/// Signed divisors paired lane-for-lane with [`I32_LANES`].
///
/// Signed division is a partial function: zero divisors and `i32::MIN / -1`
/// have no defined backend semantics and the reference rejects them, so the
/// divisor lanes exclude both rather than the interesting numerator lanes.
const I32_DIVISORS: [i32; LANES] = [1, -1, 3, -7, 5, 2, -2, 11];

/// Hostile f32 lane values: both zeroes, a subnormal, and both infinities.
const F32_LANES: [f32; LANES] = [
    0.0,
    -0.0,
    1.0,
    -1.5,
    f32::MIN_POSITIVE / 2.0,
    f32::MAX,
    f32::INFINITY,
    f32::NEG_INFINITY,
];

/// One pinned interpreter case: a program plus the inputs it is bound to.
struct ProgramCase {
    id: &'static str,
    program: Program,
    inputs: Vec<Vec<u8>>,
}

/// Guard the store on the lane bound so an oversized grid cannot write past the
/// declared buffer, matching how every dispatched program in the tree is shaped.
fn guarded_store(value: Expr) -> Vec<Node> {
    vec![
        Node::let_bind("idx", Expr::gid_x()),
        Node::if_then(
            Expr::lt(Expr::var("idx"), Expr::u32(LANES as u32)),
            vec![Node::store("out", Expr::var("idx"), value)],
        ),
    ]
}

fn binary_case(
    id: &'static str,
    dtype: DataType,
    out: DataType,
    build: fn(Expr, Expr) -> Expr,
    lhs: Vec<u8>,
    rhs: Vec<u8>,
) -> ProgramCase {
    let idx = Expr::var("idx");
    let value = build(
        Expr::load("lhs", idx.clone()),
        Expr::load("rhs", idx.clone()),
    );
    ProgramCase {
        id,
        program: Program::wrapped(
            vec![
                BufferDecl::read("lhs", 0, dtype.clone()).with_count(LANES as u32),
                BufferDecl::read("rhs", 1, dtype).with_count(LANES as u32),
                BufferDecl::output("out", 2, out).with_count(LANES as u32),
            ],
            [WORKGROUP_SIZE_X, 1, 1],
            guarded_store(value),
        ),
        inputs: vec![lhs, rhs],
    }
}

fn unary_case(
    id: &'static str,
    dtype: DataType,
    out: DataType,
    build: fn(Expr) -> Expr,
    src: Vec<u8>,
) -> ProgramCase {
    let value = build(Expr::load("input", Expr::var("idx")));
    ProgramCase {
        id,
        program: Program::wrapped(
            vec![
                BufferDecl::read("input", 0, dtype).with_count(LANES as u32),
                BufferDecl::output("out", 1, out).with_count(LANES as u32),
            ],
            [WORKGROUP_SIZE_X, 1, 1],
            guarded_store(value),
        ),
        inputs: vec![src],
    }
}

/// The pinned interpreter corpus, in golden order.
fn program_cases() -> Vec<ProgramCase> {
    let u32_in = u32_bytes(&U32_LANES);
    let i32_in = i32_bytes(&I32_LANES);
    let f32_in = f32_bytes(&F32_LANES);
    let i32_divisors = i32_bytes(&I32_DIVISORS);
    let mut cases = vec![
        binary_case("u32_add", DataType::U32, DataType::U32, Expr::add, u32_in.clone(), u32_in.clone()),
        binary_case("u32_sub", DataType::U32, DataType::U32, Expr::sub, u32_in.clone(), u32_in.clone()),
        binary_case("u32_mul", DataType::U32, DataType::U32, Expr::mul, u32_in.clone(), u32_in.clone()),
        binary_case("u32_div", DataType::U32, DataType::U32, Expr::div, u32_in.clone(), u32_in.clone()),
        binary_case("u32_rem", DataType::U32, DataType::U32, Expr::rem, u32_in.clone(), u32_in.clone()),
        binary_case("u32_mulhi", DataType::U32, DataType::U32, Expr::mulhi, u32_in.clone(), u32_in.clone()),
        binary_case("u32_shl", DataType::U32, DataType::U32, Expr::shl, u32_in.clone(), u32_in.clone()),
        binary_case("u32_shr", DataType::U32, DataType::U32, Expr::shr, u32_in.clone(), u32_in.clone()),
        binary_case("u32_bitxor", DataType::U32, DataType::U32, Expr::bitxor, u32_in.clone(), u32_in.clone()),
        binary_case("u32_min", DataType::U32, DataType::U32, Expr::min, u32_in.clone(), u32_in.clone()),
        binary_case("u32_max", DataType::U32, DataType::U32, Expr::max, u32_in.clone(), u32_in.clone()),
        binary_case("u32_abs_diff", DataType::U32, DataType::U32, Expr::abs_diff, u32_in.clone(), u32_in.clone()),
        binary_case("i32_add", DataType::I32, DataType::I32, Expr::add, i32_in.clone(), i32_in.clone()),
        binary_case("i32_div", DataType::I32, DataType::I32, Expr::div, i32_in.clone(), i32_divisors.clone()),
        binary_case("i32_rem", DataType::I32, DataType::I32, Expr::rem, i32_in.clone(), i32_divisors.clone()),
        binary_case("i32_min", DataType::I32, DataType::I32, Expr::min, i32_in.clone(), i32_divisors.clone()),
        binary_case("i32_max", DataType::I32, DataType::I32, Expr::max, i32_in.clone(), i32_divisors.clone()),
        binary_case("f32_add", DataType::F32, DataType::F32, Expr::add, f32_in.clone(), f32_in.clone()),
        binary_case("f32_mul", DataType::F32, DataType::F32, Expr::mul, f32_in.clone(), f32_in.clone()),
        binary_case("f32_div", DataType::F32, DataType::F32, Expr::div, f32_in.clone(), f32_in.clone()),
        binary_case("f32_min", DataType::F32, DataType::F32, Expr::min, f32_in.clone(), f32_in.clone()),
        binary_case("f32_max", DataType::F32, DataType::F32, Expr::max, f32_in.clone(), f32_in.clone()),
        unary_case("u32_bitnot", DataType::U32, DataType::U32, Expr::bitnot, u32_in.clone()),
        unary_case("u32_popcount", DataType::U32, DataType::U32, Expr::popcount, u32_in.clone()),
        unary_case("u32_clz", DataType::U32, DataType::U32, Expr::clz, u32_in.clone()),
        unary_case("u32_ctz", DataType::U32, DataType::U32, Expr::ctz, u32_in.clone()),
        unary_case("u32_reverse_bits", DataType::U32, DataType::U32, Expr::reverse_bits, u32_in.clone()),
        unary_case(
            "u32_to_f32",
            DataType::U32,
            DataType::F32,
            |value| Expr::cast(DataType::F32, value),
            u32_in.clone(),
        ),
        unary_case(
            "f32_to_i32",
            DataType::F32,
            DataType::I32,
            |value| Expr::cast(DataType::I32, value),
            f32_in.clone(),
        ),
        unary_case(
            "i32_to_u32",
            DataType::I32,
            DataType::U32,
            |value| Expr::cast(DataType::U32, value),
            i32_in.clone(),
        ),
        binary_case(
            "u32_lt_predicate",
            DataType::U32,
            DataType::U32,
            |left, right| Expr::select(Expr::lt(left, right), Expr::u32(1), Expr::u32(0)),
            u32_in.clone(),
            u32_in.clone(),
        ),
        binary_case(
            "u32_fused_select_chain",
            DataType::U32,
            DataType::U32,
            |left, right| {
                Expr::select(
                    Expr::ge(left.clone(), right.clone()),
                    Expr::sub(left.clone(), right.clone()),
                    Expr::add(left, right),
                )
            },
            u32_in.clone(),
            u32_in.clone(),
        ),
    ];
    cases.push(ProgramCase {
        id: "u32_branch_divergence",
        program: Program::wrapped(
            vec![
                BufferDecl::read("input", 0, DataType::U32).with_count(LANES as u32),
                BufferDecl::output("out", 1, DataType::U32).with_count(LANES as u32),
            ],
            [WORKGROUP_SIZE_X, 1, 1],
            vec![
                Node::let_bind("idx", Expr::gid_x()),
                Node::if_then(
                    Expr::lt(Expr::var("idx"), Expr::u32(LANES as u32)),
                    vec![Node::if_then_else(
                        Expr::lt(Expr::load("input", Expr::var("idx")), Expr::u32(3)),
                        vec![Node::store("out", Expr::var("idx"), Expr::u32(1))],
                        vec![Node::store(
                            "out",
                            Expr::var("idx"),
                            Expr::add(Expr::load("input", Expr::var("idx")), Expr::u32(7)),
                        )],
                    )],
                ),
            ],
        ),
        inputs: vec![u32_in.clone()],
    });
    cases.push(ProgramCase {
        id: "u32_loop_accumulate",
        program: Program::wrapped(
            vec![
                BufferDecl::read("input", 0, DataType::U32).with_count(LANES as u32),
                BufferDecl::output("out", 1, DataType::U32).with_count(LANES as u32),
            ],
            [WORKGROUP_SIZE_X, 1, 1],
            vec![
                Node::let_bind("idx", Expr::gid_x()),
                Node::if_then(
                    Expr::lt(Expr::var("idx"), Expr::u32(LANES as u32)),
                    vec![
                        Node::store("out", Expr::var("idx"), Expr::u32(0)),
                        Node::loop_for(
                            "step",
                            Expr::u32(0),
                            Expr::u32(4),
                            vec![Node::store(
                                "out",
                                Expr::var("idx"),
                                Expr::add(
                                    Expr::load("out", Expr::var("idx")),
                                    Expr::mul(
                                        Expr::load("input", Expr::var("idx")),
                                        Expr::var("step"),
                                    ),
                                ),
                            )],
                        ),
                    ],
                ),
            ],
        ),
        inputs: vec![u32_in.clone()],
    });
    cases.push(ProgramCase {
        id: "u32_workgroup_scratch_exchange",
        program: Program::wrapped(
            vec![
                BufferDecl::read("input", 0, DataType::U32).with_count(LANES as u32),
                BufferDecl::workgroup("scratch", LANES as u32, DataType::U32),
                BufferDecl::output("out", 1, DataType::U32).with_count(LANES as u32),
            ],
            [WORKGROUP_SIZE_X, 1, 1],
            vec![
                Node::let_bind("idx", Expr::gid_x()),
                Node::if_then(
                    Expr::lt(Expr::var("idx"), Expr::u32(LANES as u32)),
                    vec![Node::store(
                        "scratch",
                        Expr::var("idx"),
                        Expr::load("input", Expr::var("idx")),
                    )],
                ),
                Node::barrier(),
                Node::if_then(
                    Expr::lt(Expr::var("idx"), Expr::u32(LANES as u32)),
                    vec![Node::store(
                        "out",
                        Expr::var("idx"),
                        Expr::add(
                            Expr::load("scratch", Expr::var("idx")),
                            Expr::load(
                                "scratch",
                                Expr::rem(
                                    Expr::add(Expr::var("idx"), Expr::u32(1)),
                                    Expr::u32(LANES as u32),
                                ),
                            ),
                        ),
                    )],
                ),
            ],
        ),
        inputs: vec![u32_in.clone()],
    });
    cases
}

/// Render every registered dual facet plus the pinned interpreter corpus.
fn render_corpus() -> String {
    let mut text = String::from(HEADER);
    for op_id in dual_op_ids() {
        let (reference_a, reference_b) =
            resolve_dual(op_id).expect("Fix: a registered dual facet must resolve");
        writeln!(text, "{CASE_MARKER}dual::{op_id}").expect("string write");
        for seed in FACET_SEEDS {
            let input = facet_input(seed);
            let output_a = reference_a(&input);
            let output_b = reference_b(&input);
            assert_eq!(
                output_a, output_b,
                "Fix: dual references for {op_id} diverged at seed {seed:#010x}"
            );
            writeln!(text, "seed {seed:#010x}").expect("string write");
            text.push_str(&hex_words(&output_a));
        }
    }
    for case in program_cases() {
        writeln!(text, "{CASE_MARKER}program::{}", case.id).expect("string write");
        let values = case
            .inputs
            .iter()
            .map(|bytes| Value::Bytes(Arc::from(bytes.clone().into_boxed_slice())))
            .collect::<Vec<_>>();
        let outputs = reference_eval(&case.program, &values).unwrap_or_else(|error| {
            panic!(
                "Fix: pinned reference corpus case `{}` must evaluate: {error}",
                case.id
            )
        });
        for (index, output) in outputs.iter().enumerate() {
            writeln!(text, "output {index}").expect("string write");
            text.push_str(&hex_words(&output.to_bytes()));
        }
    }
    text
}

/// WHY: the reference interpreter is the conformance oracle. A change in the
/// bytes it computes is a change in what every backend is graded against, so it
/// must never happen as a side effect of a refactor.
#[test]
fn reference_outputs_match_the_pinned_corpus() {
    assert_matches_golden(&golden_path(), &render_corpus());
}

/// WHY: reference evaluation must be a pure function of program and input. A
/// renderer that depended on iteration order or an address would pass the
/// golden once and fail the next run.
#[test]
fn reference_outputs_are_deterministic_across_runs() {
    assert_eq!(render_corpus(), render_corpus());
}

/// WHY: a pinned corpus that no longer names every registered dual facet would
/// silently stop covering whichever one was added last.
#[test]
fn pinned_corpus_covers_every_registered_dual_facet() {
    let golden =
        std::fs::read_to_string(golden_path()).expect("pinned reference corpus must exist");
    for op_id in dual_op_ids() {
        assert!(
            golden.contains(&format!("{CASE_MARKER}dual::{op_id}\n")),
            "Fix: pinned reference corpus is missing dual facet `{op_id}`; re-bless it."
        );
    }
}

#[test]
#[ignore = "bless: rewrites the pinned reference-output corpus"]
fn bless_pinned_reference_corpus() {
    write_golden(&golden_path(), &render_corpus());
}
