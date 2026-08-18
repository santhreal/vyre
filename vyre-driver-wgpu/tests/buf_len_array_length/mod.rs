//! Q3 reproducer: `Expr::buf_len(buffer)` lowers to `naga::ArrayLength`
//! on the wgpu/Vulkan path. ArrayLength must equal the bound storage
//! buffer's element count at dispatch time. The cat_a_gpu_differential
//! pass on 2026-05-02 surfaced a regression where the unbounded
//! `vyre-libs::hash::fnv1a64` registration (loop bound = buf_len)
//! caused the GPU loop to run zero iterations, returning the unchanged
//! FNV1A64_OFFSET.
//!
//! These tests build the smallest possible Program that exercises
//! `Expr::buf_len` at runtime and assert that the dispatched output
//! reflects the actual bound buffer length. They are written to fail
//! before a Q3 fix lands and pass after, so the workaround in
//! `vyre_libs::hash::fnv1a` (using `fnv1a64_program_n` instead of
//! `fnv1a64_program`) can be reverted with confidence.
//!
//! Lane: `driver_wgpu` (per `docs/optimization/OWNERSHIP.toml`).

use std::sync::{Arc, OnceLock};

use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

fn backend() -> &'static WgpuBackend {
    static BACKEND: OnceLock<WgpuBackend> = OnceLock::new();
    BACKEND.get_or_init(|| {
        WgpuBackend::acquire().expect(
            "Fix: GPU adapter required for buf_len_array_length tests. Run on a host with a working wgpu adapter.",
        )
    })
}

pub(super) fn wrapped_storage_program(body: Node) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![body],
    )
}

pub(super) fn triple_nested_region(inner_body: Vec<Node>, label_stem: &str) -> Node {
    let mid = Node::Region {
        generator: Ident::from(format!("vyre-primitives::test::{label_stem}_inner")),
        source_region: None,
        body: Arc::new(inner_body),
    };
    let outer = Node::Region {
        generator: Ident::from(format!("vyre-primitives::test::{label_stem}_mid")),
        source_region: Some(Ident::from(format!("vyre-libs::test::{label_stem}_outer"))),
        body: Arc::new(vec![mid]),
    };
    Node::Region {
        generator: Ident::from(format!("vyre-libs::test::{label_stem}_outer")),
        source_region: None,
        body: Arc::new(vec![outer]),
    }
}

/// Build a Program whose body writes `buf_len(input)` to `out[0]`.
/// `input` is declared without a static count, so the lowering uses
/// `naga::ArrayLength` to read the bound buffer's element count at
/// runtime. `out` is one u32 with explicit count = 1.
fn dispatch_and_read_first_word(program: &Program, input_bytes: Vec<u8>) -> u32 {
    dispatch_and_read_first_word_with_lowering(program, input_bytes, false)
}

/// Like [`dispatch_and_read_first_word`] but routes the program through
/// the same `vyre_foundation::optimizer::optimize` pass
/// that `cat_a_gpu_differential::lower_for_gpu` uses. The catalog
/// failure cases hit that path; pure direct dispatch does not.
fn dispatch_and_read_first_word_lowered(program: &Program, input_bytes: Vec<u8>) -> u32 {
    dispatch_and_read_first_word_with_lowering(program, input_bytes, true)
}

fn dispatch_and_read_first_word_with_lowering(
    program: &Program,
    input_bytes: Vec<u8>,
    lower: bool,
) -> u32 {
    let lowered;
    let prog = if lower {
        lowered = vyre_foundation::optimizer::optimize(program.clone())
            .expect("registered optimizer must converge");
        &lowered
    } else {
        program
    };
    let inputs = vec![input_bytes, vec![0u8; 4]];
    let outputs = backend()
        .dispatch(prog, &inputs, &DispatchConfig::default())
        .expect("Fix: backend.dispatch must succeed for the buf_len writer program");
    let raw = &outputs[0];
    assert!(
        raw.len() >= 4,
        "Fix: output buffer too small to read a u32 result"
    );
    u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])
}

fn dispatch_and_read_words(program: &Program, input_bytes: Vec<u8>) -> Vec<u32> {
    let inputs = vec![input_bytes, vec![0u8; 16]];
    let outputs = backend()
        .dispatch(program, &inputs, &DispatchConfig::default())
        .expect("Fix: backend.dispatch must succeed for the word writer program");
    outputs[0]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn dispatch_and_read_words_with_inputs(program: &Program, inputs: Vec<Vec<u8>>) -> Vec<u32> {
    let outputs = backend()
        .dispatch(program, &inputs, &DispatchConfig::default())
        .expect("Fix: backend.dispatch must succeed for the word writer program");
    outputs[0]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// The byte address invocation `w` handles for lane `k`: `w * 4 + k`.
///
/// Every packing program in this family lays four byte lanes across one output
/// word, so this address, the grid gate below, and the atomic-or accumulate
/// lane are the shape all of them share. Restating them per program is what let
/// two of these files carry the same forty lines twice.
fn lane_addr(k: u32) -> Expr {
    Expr::add(Expr::mul(Expr::var("w"), Expr::u32(4)), Expr::u32(k))
}

/// An unclamped one-byte load from `input`, widened to `u32`.
fn input_byte(addr: Expr) -> Expr {
    Expr::cast(DataType::U32, Expr::load("input", addr))
}

/// A `buf_len`-clamped one-byte load from `input`, masked to a single byte.
///
/// `input` carries no static count, so `buf_len` lowers to `naga::ArrayLength`
/// and the clamp is what keeps a lane in bounds when the bound buffer is
/// shorter than the lane grid.
fn clamped_input_byte(addr: Expr) -> Expr {
    let len = Expr::buf_len("input");
    let safe_addr = Expr::select(
        Expr::lt(addr.clone(), len.clone()),
        addr,
        Expr::saturating_sub(len, Expr::u32(1)),
    );
    Expr::bitand(input_byte(safe_addr), Expr::u32(0xFF))
}

/// `in_byte_{k}`, either a space when `comment` holds, or `byte` otherwise.
///
/// Written as declare-then-assign rather than a `select` on purpose: an
/// `Assign` to an outer-scope binding from inside both arms of an
/// `if_then_else` is the carrier shape these contracts exist to pin.
fn assigned_byte_nodes(k: u32, comment: Expr, byte: Expr) -> Vec<Node> {
    vec![
        Node::let_bind(format!("in_byte_{k}"), Expr::u32(0)),
        Node::if_then_else(
            comment,
            vec![Node::assign(
                &format!("in_byte_{k}"),
                Expr::u32(b' ' as u32),
            )],
            vec![Node::assign(&format!("in_byte_{k}"), byte)],
        ),
    ]
}

/// `out_pos_{k}`, `out_word_idx_{k}` and `out_shift_{k}` from `off_{k}`: the
/// destination byte index, the word holding it, and its bit shift in that word.
fn scatter_position_nodes(k: u32) -> Vec<Node> {
    vec![
        Node::let_bind(
            format!("out_pos_{k}"),
            Expr::saturating_sub(Expr::var(format!("off_{k}")), Expr::u32(1)),
        ),
        Node::let_bind(
            format!("out_word_idx_{k}"),
            Expr::div(Expr::var(format!("out_pos_{k}")), Expr::u32(4)),
        ),
        Node::let_bind(
            format!("out_shift_{k}"),
            Expr::mul(
                Expr::rem(Expr::var(format!("out_pos_{k}")), Expr::u32(4)),
                Expr::u32(8),
            ),
        ),
    ]
}

/// Fold `value` into `out[index]` with an atomic or, binding the prior word.
fn atomic_or_lane(k: u32, index: Expr, value: Expr) -> Node {
    Node::let_bind(format!("prev_{k}"), Expr::atomic_or("out", index, value))
}

/// `let w = InvocationId(0); if w < words { lanes }`, the grid gate every
/// packing program here shares.
fn invocation_gated(words: u32, lanes: Vec<Node>) -> Vec<Node> {
    vec![
        Node::let_bind("w", Expr::InvocationId { axis: 0 }),
        Node::if_then(Expr::lt(Expr::var("w"), Expr::u32(words)), lanes),
    ]
}

/// The four lanes of one output word, concatenated.
fn four_lanes(lane: impl Fn(u32) -> Vec<Node>) -> Vec<Node> {
    (0..4).flat_map(lane).collect()
}

/// A packing program: a runtime-length `input` of bytes, `extra` buffers, and a
/// read-write `out` of `words` words the lanes accumulate into.
fn packing_program(words: u32, extra: Vec<BufferDecl>, lanes: Vec<Node>) -> Program {
    let out_binding = 1 + extra.len() as u32;
    let mut buffers = vec![BufferDecl::storage(
        "input",
        0,
        BufferAccess::ReadOnly,
        DataType::U8,
    )];
    buffers.extend(extra);
    buffers.push(
        BufferDecl::storage("out", out_binding, BufferAccess::ReadWrite, DataType::U32)
            .with_count(words),
    );
    Program::wrapped(buffers, [256, 1, 1], invocation_gated(words, lanes))
}

mod basic_len_contracts;
mod dynamic_pack_contracts;
mod fnv_loop_contracts;
mod region_loop_contracts;
mod scatter_contracts;
