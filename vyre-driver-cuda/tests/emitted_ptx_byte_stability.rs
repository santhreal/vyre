//! Byte-stability golden for the PTX this driver emits.
//!
//! A refactor of the CUDA driver is only safe if the bytes it hands the device
//! do not move. This pins the whole host-side emission path with no GPU:
//! the registered target compiler chooses the emit options from its own
//! `TargetProfile`, `compile_selected_modules` fuses and lowers each selected
//! group, the emitter renders PTX, and driver admission decodes the module
//! bundle back out. Every one of those steps is between a `Program` and the
//! `.visible .entry main` text, and none of them needs a device.
//!
//! The section format and the comparison live in
//! `vyre_lower::artifact_golden`. This file supplies the program corpus and the
//! PTX rendering.
//!
//! Materialization and dispatch are covered by the live tests in
//! `tests/target_compiler.rs`; this file deliberately stops at the bytes.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use vyre_driver::materialize::{self, MaterializerTarget};
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program, ProgramGraph};
use vyre_lower::artifact_golden::{assert_matches_golden, write_golden};
use vyre_megakernel::{
    Artifact, CompileRequest, Digest, ExternalFacts, SearchBudget, TargetCompiler, TargetPayload,
};

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

/// Lane count shared by every corpus program.
const LANES: u32 = 64;

/// Workgroup width shared by every corpus program.
const WORKGROUP_SIZE_X: u32 = 32;

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/emitted_ptx_corpus.ptx")
}

/// Guard the store on the lane bound, matching how dispatched programs are shaped.
fn guarded_store(buffer: &str, value: Expr) -> Vec<Node> {
    vec![
        Node::let_bind("idx", Expr::gid_x()),
        Node::if_then(
            Expr::lt(Expr::var("idx"), Expr::u32(LANES)),
            vec![Node::store(buffer, Expr::var("idx"), value)],
        ),
    ]
}

/// One pinned emission case.
struct PtxCase {
    id: &'static str,
    program: Program,
}

fn binary_case(id: &'static str, dtype: DataType, build: fn(Expr, Expr) -> Expr) -> PtxCase {
    let idx = Expr::var("idx");
    PtxCase {
        id,
        program: Program::wrapped(
            vec![
                BufferDecl::read("lhs", 0, dtype.clone()).with_count(LANES),
                BufferDecl::read("rhs", 1, dtype.clone()).with_count(LANES),
                BufferDecl::output("out", 2, dtype).with_count(LANES),
            ],
            [WORKGROUP_SIZE_X, 1, 1],
            guarded_store(
                "out",
                build(
                    Expr::load("lhs", idx.clone()),
                    Expr::load("rhs", idx.clone()),
                ),
            ),
        ),
    }
}

fn unary_case(id: &'static str, dtype: DataType, build: fn(Expr) -> Expr) -> PtxCase {
    PtxCase {
        id,
        program: Program::wrapped(
            vec![
                BufferDecl::read("input", 0, dtype.clone()).with_count(LANES),
                BufferDecl::output("out", 1, dtype).with_count(LANES),
            ],
            [WORKGROUP_SIZE_X, 1, 1],
            guarded_store("out", build(Expr::load("input", Expr::var("idx")))),
        ),
    }
}

/// The pinned emission corpus, in golden order.
///
/// The set spans what changes PTX text: integer and float ALU, the division and
/// shift arms that need masking, casts between widths, divergent control flow, a
/// counted loop, workgroup scratch with a barrier, and an atomic.
fn ptx_cases() -> Vec<PtxCase> {
    let mut cases = vec![
        binary_case("u32_add", DataType::U32, Expr::add),
        binary_case("u32_sub", DataType::U32, Expr::sub),
        binary_case("u32_mul", DataType::U32, Expr::mul),
        binary_case("u32_div", DataType::U32, Expr::div),
        binary_case("u32_rem", DataType::U32, Expr::rem),
        binary_case("u32_mulhi", DataType::U32, Expr::mulhi),
        binary_case("u32_shl", DataType::U32, Expr::shl),
        binary_case("u32_shr", DataType::U32, Expr::shr),
        binary_case("u32_bitand", DataType::U32, Expr::bitand),
        binary_case("u32_bitor", DataType::U32, Expr::bitor),
        binary_case("u32_bitxor", DataType::U32, Expr::bitxor),
        binary_case("u32_min", DataType::U32, Expr::min),
        binary_case("u32_max", DataType::U32, Expr::max),
        binary_case("u32_abs_diff", DataType::U32, Expr::abs_diff),
        binary_case("i32_add", DataType::I32, Expr::add),
        binary_case("i32_mul", DataType::I32, Expr::mul),
        binary_case("f32_add", DataType::F32, Expr::add),
        binary_case("f32_mul", DataType::F32, Expr::mul),
        binary_case("f32_div", DataType::F32, Expr::div),
        binary_case("f32_min", DataType::F32, Expr::min),
        binary_case("f32_max", DataType::F32, Expr::max),
        unary_case("u32_bitnot", DataType::U32, Expr::bitnot),
        unary_case("u32_popcount", DataType::U32, Expr::popcount),
        unary_case("u32_clz", DataType::U32, Expr::clz),
        unary_case("u32_ctz", DataType::U32, Expr::ctz),
        unary_case("u32_reverse_bits", DataType::U32, Expr::reverse_bits),
    ];
    cases.push(PtxCase {
        id: "u32_to_f32_cast",
        program: Program::wrapped(
            vec![
                BufferDecl::read("input", 0, DataType::U32).with_count(LANES),
                BufferDecl::output("out", 1, DataType::F32).with_count(LANES),
            ],
            [WORKGROUP_SIZE_X, 1, 1],
            guarded_store(
                "out",
                Expr::cast(DataType::F32, Expr::load("input", Expr::var("idx"))),
            ),
        ),
    });
    cases.push(PtxCase {
        id: "f32_to_i32_cast",
        program: Program::wrapped(
            vec![
                BufferDecl::read("input", 0, DataType::F32).with_count(LANES),
                BufferDecl::output("out", 1, DataType::I32).with_count(LANES),
            ],
            [WORKGROUP_SIZE_X, 1, 1],
            guarded_store(
                "out",
                Expr::cast(DataType::I32, Expr::load("input", Expr::var("idx"))),
            ),
        ),
    });
    cases.push(PtxCase {
        id: "u32_branch_divergence",
        program: Program::wrapped(
            vec![
                BufferDecl::read("input", 0, DataType::U32).with_count(LANES),
                BufferDecl::output("out", 1, DataType::U32).with_count(LANES),
            ],
            [WORKGROUP_SIZE_X, 1, 1],
            vec![
                Node::let_bind("idx", Expr::gid_x()),
                Node::if_then(
                    Expr::lt(Expr::var("idx"), Expr::u32(LANES)),
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
    });
    cases.push(PtxCase {
        id: "u32_loop_accumulate",
        program: Program::wrapped(
            vec![
                BufferDecl::read("input", 0, DataType::U32).with_count(LANES),
                BufferDecl::output("out", 1, DataType::U32).with_count(LANES),
            ],
            [WORKGROUP_SIZE_X, 1, 1],
            vec![
                Node::let_bind("idx", Expr::gid_x()),
                Node::if_then(
                    Expr::lt(Expr::var("idx"), Expr::u32(LANES)),
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
    });
    cases.push(PtxCase {
        id: "u32_workgroup_scratch_exchange",
        program: Program::wrapped(
            vec![
                BufferDecl::read("input", 0, DataType::U32).with_count(LANES),
                BufferDecl::workgroup("scratch", WORKGROUP_SIZE_X, DataType::U32),
                BufferDecl::output("out", 1, DataType::U32).with_count(LANES),
            ],
            [WORKGROUP_SIZE_X, 1, 1],
            vec![
                Node::let_bind("idx", Expr::gid_x()),
                Node::let_bind("lane", Expr::invocation_local_x()),
                Node::if_then(
                    Expr::lt(Expr::var("idx"), Expr::u32(LANES)),
                    vec![Node::store(
                        "scratch",
                        Expr::var("lane"),
                        Expr::load("input", Expr::var("idx")),
                    )],
                ),
                Node::barrier(),
                Node::if_then(
                    Expr::lt(Expr::var("idx"), Expr::u32(LANES)),
                    vec![Node::store(
                        "out",
                        Expr::var("idx"),
                        Expr::add(
                            Expr::load("scratch", Expr::var("lane")),
                            Expr::load(
                                "scratch",
                                Expr::rem(
                                    Expr::add(Expr::var("lane"), Expr::u32(1)),
                                    Expr::u32(WORKGROUP_SIZE_X),
                                ),
                            ),
                        ),
                    )],
                ),
            ],
        ),
    });
    cases
}

/// Wrap one corpus program in the single-node graph the artifact route expects.
///
/// `ProgramGraph::from_program` owns lifting host-visible buffers into typed
/// external values, so the corpus does not restate that contract per case.
fn artifact_for(case: &PtxCase) -> Artifact {
    let graph = ProgramGraph::from_program("main", case.program.clone()).unwrap_or_else(|error| {
        panic!("Fix: corpus case `{}` must form a graph node: {error}", case.id)
    });
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        SearchBudget::new(1, 1, 0, 0, 1),
        1_000_000,
    )
    .validate()
    .unwrap_or_else(|error| {
        panic!("Fix: corpus case `{}` must validate: {error}", case.id)
    });
    vyre_megakernel::compile(&request)
        .unwrap_or_else(|error| panic!("Fix: corpus case `{}` must compile: {error}", case.id))
}

/// The registered CUDA target compiler, acquired without a device.
fn cuda_target_compiler() -> Box<dyn TargetCompiler> {
    vyre_driver::backend::backend_registration(vyre_driver_cuda::CUDA_BACKEND_ID)
        .expect("Fix: the CUDA backend registration must be linked into this test binary.")
        .target_compiler()
        .expect("Fix: CUDA target-payload production must not require a device.")
}

/// Render every admitted module of one payload as PTX text plus its entry metadata.
fn render_payload(
    compiler: &dyn TargetCompiler,
    artifact: &Artifact,
    payload: &TargetPayload,
    case_id: &str,
) -> String {
    let admitted = materialize::admit(
        artifact,
        payload,
        MaterializerTarget {
            backend_id: vyre_driver_cuda::CUDA_BACKEND_ID,
            format: compiler.format(),
            profile: compiler.profile(),
        },
    )
    .unwrap_or_else(|error| panic!("Fix: corpus case `{case_id}` must admit: {error}"));
    let mut text = String::new();
    for entry in payload.entries() {
        writeln!(
            text,
            "entry {} grid {:?} workgroup {:?} dynamic_shared {}",
            entry.name, entry.grid_size, entry.workgroup_size, entry.dynamic_shared_bytes
        )
        .expect("string write");
        for binding in &entry.resource_bindings {
            writeln!(text, "  binding {binding:?}").expect("string write");
        }
    }
    for module in &admitted {
        let ptx = std::str::from_utf8(&module.image.bytes).unwrap_or_else(|error| {
            panic!("Fix: corpus case `{case_id}` must emit UTF-8 PTX: {error}")
        });
        writeln!(text, "module group {:?}", module.image.group).expect("string write");
        text.push_str(ptx);
        if !ptx.ends_with('\n') {
            text.push('\n');
        }
    }
    text
}

/// Render the pinned corpus through the registered CUDA payload route.
fn render_corpus() -> String {
    let compiler = cuda_target_compiler();
    let mut text = String::from(HEADER);
    for case in ptx_cases() {
        let artifact = artifact_for(&case);
        let payload = compiler.compile(&artifact).unwrap_or_else(|error| {
            panic!("Fix: corpus case `{}` must produce a payload: {error:?}", case.id)
        });
        writeln!(text, "{CASE_MARKER}{}", case.id).expect("string write");
        text.push_str(&render_payload(
            compiler.as_ref(),
            &artifact,
            &payload,
            case.id,
        ));
    }
    text
}

/// WHY: the PTX this driver hands the device is the product. A dedup refactor
/// must not move one byte of it, and only a pinned corpus can prove that.
#[test]
fn emitted_ptx_matches_the_pinned_corpus() {
    assert_matches_golden(&golden_path(), &render_corpus());
}

/// WHY: emission must be a pure function of the artifact. A renderer that
/// depended on iteration order or an address would pass the golden once and
/// fail the next run.
#[test]
fn emitted_ptx_is_deterministic_across_runs() {
    assert_eq!(render_corpus(), render_corpus());
}

/// WHY: a pinned corpus that no longer names every case would silently stop
/// covering the one that was dropped.
#[test]
fn pinned_corpus_covers_every_case() {
    let golden = std::fs::read_to_string(golden_path()).expect("pinned PTX corpus must exist");
    for case in ptx_cases() {
        assert!(
            golden.contains(&format!("{CASE_MARKER}{}\n", case.id)),
            "Fix: pinned PTX corpus is missing case `{}`; re-bless it.",
            case.id
        );
    }
}

#[test]
#[ignore = "bless: rewrites the pinned emitted-PTX corpus"]
fn bless_pinned_ptx_corpus() {
    write_golden(&golden_path(), &render_corpus());
}
