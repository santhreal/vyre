use std::collections::{BTreeMap, BTreeSet};

use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;
use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_driver::VyreBackend;
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::visit::{expr_children, walk_nodes};

/// The read-only ProgramGraph buffer names, in binding order.
///
/// Imported from the crate that declares them, so the guards below and the
/// declarations every graph op emits are one list. A second copy here would
/// drift silently on a rename: the guards would stop recognizing the bundle and
/// resume randomizing topology buffers, which is the defect they exist to
/// prevent. The module is reachable because this crate already depends on
/// `vyre-libs` with `reasoning`, and `reasoning` enables `graph`.
use vyre_libs::graph::program_graph::READ_ONLY_BUFFER_NAMES;

/// Buffers whose contents reach a loop bound in this program.
///
/// A trip count read from a buffer is only as bounded as the value in it: one
/// random `u32` asks for four billion iterations, which the interpreter runs
/// and the sweep reports as a hang rather than as a parity result. The set is
/// derived from the program IR instead of from buffer names, because a name
/// list only covers the topology names someone thought of:
/// `security::flows_to_with_sanitizer` carries its row offsets under names the
/// CSR list does not match, and the next op to do that would be missed the
/// same way.
pub(crate) fn trip_count_buffers(program: &Program) -> BTreeSet<String> {
    let mut loaded: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut derived: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    walk_nodes(program, |node| {
        let (Node::Let { name, value } | Node::Assign { name, value }) = node else {
            return;
        };
        let (buffers, vars) = expression_sources(value);
        loaded.entry(name.to_string()).or_default().extend(buffers);
        derived.entry(name.to_string()).or_default().extend(vars);
    });
    // A bound often reads a variable bound from another variable, so the
    // buffers a name stands for are closed over the whole chain.
    loop {
        let mut grew = false;
        for (name, parents) in &derived {
            let inherited: BTreeSet<String> = parents
                .iter()
                .filter_map(|parent| loaded.get(parent))
                .flatten()
                .cloned()
                .collect();
            let own = loaded.entry(name.clone()).or_default();
            for buffer in inherited {
                grew |= own.insert(buffer);
            }
        }
        if !grew {
            break;
        }
    }
    let mut sources = BTreeSet::new();
    walk_nodes(program, |node| {
        let Node::Loop { from, to, .. } = node else {
            return;
        };
        for bound in [from, to] {
            let (buffers, vars) = expression_sources(bound);
            sources.extend(buffers);
            for var in vars {
                if let Some(buffers) = loaded.get(&var) {
                    sources.extend(buffers.iter().cloned());
                }
            }
        }
    });
    sources
}

/// The buffers an expression loads and the variables it reads.
fn expression_sources(expr: &Expr) -> (BTreeSet<String>, BTreeSet<String>) {
    let mut buffers = BTreeSet::new();
    let mut vars = BTreeSet::new();
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match current {
            Expr::Load { buffer, .. } => {
                buffers.insert(buffer.to_string());
            }
            Expr::Var(name) => {
                vars.insert(name.to_string());
            }
            _ => {}
        }
        stack.extend(expr_children(current).iter());
    }
    (buffers, vars)
}

/// Deterministic per-op seed derived from the stable op id.
pub(crate) fn op_seed(op_id: &str) -> u64 {
    let hash = blake3::hash(op_id.as_bytes());
    let b = hash.as_bytes();
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

pub(crate) fn missing_capability_reason(
    backend: &WgpuBackend,
    program: &Program,
) -> Option<String> {
    let required = vyre_foundation::program_caps::scan(program);
    let check = vyre_foundation::program_caps::check_backend_capabilities(
        backend.id(),
        &vyre_driver::validation::ProgramValidationCaps::from_backend(backend).support(),
        &required,
    );
    check.err().map(|missing| missing.to_string())
}

pub(crate) fn randomize_buffer(op_id: &str, program: &Program, buffer_idx: usize) -> bool {
    let Some(buffer) = program.buffers().get(buffer_idx) else {
        return false;
    };
    if buffer.access() != BufferAccess::ReadOnly {
        return false;
    }
    if is_program_graph_topology(program, buffer_idx) {
        return false;
    }
    if is_csr_topology_name(buffer.name()) {
        return false;
    }
    if trip_count_buffers(program).contains(buffer.name()) {
        return false;
    }

    match op_id {
        "vyre-libs::decode::inflate_stored_block" => false,
        "vyre-libs::parsing::c_lexer" => buffer_idx == 0,
        "vyre-libs::parsing::c_keyword" | "vyre-libs::parsing::c_keyword_packed_haystack" => {
            buffer_idx == 4 || buffer_idx == 5
        }
        "vyre-libs::reduce::segment_reduce_sum" => buffer_idx == 0,
        "vyre-libs::pattern::bracket_match" | "vyre-libs::parsing::ast_shunting_yard" => false,
        "vyre-libs::nn::attention::quest_paging" => buffer.name() == "q",
        id if id.contains("::conv1d") => buffer.name() != "params",
        _ => true,
    }
}

/// True when the buffer at `buffer_idx` is a read-only ProgramGraph bundle
/// member.
///
/// Matched by name against the ABI set `vyre-libs` exports rather than by
/// position. A primitive may declare its own buffers before the bundle, which
/// leaves the bundle at no fixed index: `security::flows_to_with_sanitizer`
/// emits its sanitizer difference first and carries the bundle at 3..=7, so a
/// guard reading indices 0..5 randomized its node array, edge-kind mask and
/// node tags, while `security::flows_to` over the same topology kept all five
/// because its traversal declares the bundle first.
fn is_program_graph_topology(program: &Program, buffer_idx: usize) -> bool {
    program
        .buffers()
        .get(buffer_idx)
        .is_some_and(|buffer| READ_ONLY_BUFFER_NAMES.contains(&buffer.name()))
}

fn is_csr_topology_name(name: &str) -> bool {
    name.ends_with("_offsets")
        || name.ends_with("_targets")
        || name.contains("edge_offsets")
        || name.contains("edge_targets")
}

fn program_graph_node_count(program: &Program) -> Option<u32> {
    program
        .buffers()
        .iter()
        .find(|buffer| buffer.name().contains("edge_offsets"))
        .and_then(|buffer| buffer.count().checked_sub(1))
        .filter(|&node_count| node_count > 0)
}

/// True when the buffer at `buffer_idx` is a node-indexed bitset the graph ABI
/// expects to hold no bit at or above the node count.
///
/// Every buffer outside the read-only bundle is a candidate, because the bundle
/// is the only part of the ABI whose contents a primitive does not own.
pub(crate) fn is_program_graph_frontier(program: &Program, buffer_idx: usize) -> bool {
    let Some(node_count) = program_graph_node_count(program) else {
        return false;
    };
    let Some(buffer) = program.buffers().get(buffer_idx) else {
        return false;
    };
    let words = node_count.div_ceil(32).max(1);
    !is_program_graph_topology(program, buffer_idx)
        && buffer.element() == DataType::U32
        && buffer.count() == words
}

pub(crate) fn random_program_graph_frontier(
    program: &Program,
    len: usize,
    runner: &mut TestRunner,
) -> Vec<u8> {
    let node_count = program_graph_node_count(program).expect("frontier implies graph shape");
    let words = node_count.div_ceil(32).max(1) as usize;
    assert_eq!(
        len,
        words * core::mem::size_of::<u32>(),
        "ProgramGraph frontier fixture length must match declared node count"
    );

    let strategy = proptest::collection::vec(any::<u32>(), words);
    let tree = strategy
        .new_tree(runner)
        .expect("frontier bitset strategy must yield a tree");
    let mut values = tree.current();
    let used_bits = node_count % 32;
    if used_bits != 0 {
        let mask = (1u32 << used_bits) - 1;
        if let Some(last) = values.last_mut() {
            *last &= mask;
        }
    }
    values
        .into_iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn random_u32_vec(
    words: usize,
    range: core::ops::RangeInclusive<u32>,
    runner: &mut TestRunner,
) -> Vec<u32> {
    let strategy = proptest::collection::vec(range, words);
    let tree = strategy
        .new_tree(runner)
        .expect("bounded u32 vector strategy must yield a tree");
    tree.current()
}

fn u32_words_to_bytes(words: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(words)
}

pub(crate) fn random_amg_v_cycle_inputs(
    fixture_case: &[Vec<u8>],
    runner: &mut TestRunner,
) -> Vec<Vec<u8>> {
    let fine_diag = random_u32_vec(4, 2..=8, runner);
    let mut coarse_diag = random_u32_vec(2, 2..=8, runner);
    let initial_x = vec![0u32; 4];

    let mut a = vec![0u32; 16];
    for row in 0..4 {
        for col in 0..4 {
            a[row * 4 + col] = if row == col { fine_diag[row] << 16 } else { 0 };
        }
    }
    let rhs = vec![0u32; 4];

    let r_mat = [1u32 << 16, 0, 0, 0, 0, 0, 1u32 << 16, 0];
    let p_mat = [1u32 << 16, 0, 1u32 << 15, 0, 0, 1u32 << 16, 0, 1u32 << 15];
    let a_c = [
        coarse_diag.remove(0) << 16,
        0,
        0,
        coarse_diag.remove(0) << 16,
    ];
    let omega = [1u32 << 15];

    let mut inputs = Vec::with_capacity(fixture_case.len());
    inputs.push(u32_words_to_bytes(&a));
    inputs.push(u32_words_to_bytes(&rhs));
    inputs.push(u32_words_to_bytes(&initial_x));
    inputs.push(u32_words_to_bytes(&r_mat));
    inputs.push(u32_words_to_bytes(&p_mat));
    inputs.push(u32_words_to_bytes(&a_c));
    inputs.push(u32_words_to_bytes(&omega));
    inputs.extend(fixture_case.iter().skip(7).cloned());
    inputs
}

pub(crate) fn random_buffer_for(
    op_id: &str,
    buffer: &BufferDecl,
    len: usize,
    runner: &mut TestRunner,
) -> Vec<u8> {
    if op_id.contains("newton_schulz") && buffer.element() == DataType::F32 && len % 4 == 0 {
        let strategy = proptest::collection::vec(0u32..=1_000, len / 4);
        let tree = strategy
            .new_tree(runner)
            .expect("bounded Newton-Schulz f32 strategy must yield a tree");
        return tree
            .current()
            .into_iter()
            .map(|value| value as f32 / 1_000.0)
            .flat_map(f32::to_le_bytes)
            .collect();
    }

    if op_id.contains("amg_v_cycle") && buffer.element() == DataType::U32 && len % 4 == 0 {
        let strategy = proptest::collection::vec(0u32..=4_096, len / 4);
        let tree = strategy
            .new_tree(runner)
            .expect("bounded AMG fixed-point strategy must yield a tree");
        return tree
            .current()
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
    }

    if buffer.element() == DataType::F32 && len % 4 == 0 {
        let strategy = proptest::collection::vec(0u32..=4096, len / 4);
        let tree = strategy
            .new_tree(runner)
            .expect("finite f32 strategy must yield a tree");
        return tree
            .current()
            .into_iter()
            .flat_map(|value| (value as f32).to_le_bytes())
            .collect();
    }

    let strategy = proptest::collection::vec(any::<u8>(), len);
    let tree = strategy
        .new_tree(runner)
        .expect("vec strategy must yield a tree");
    tree.current()
}

fn ordered_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// Compare CPU and GPU outputs. On mismatch returns a string naming the
/// op, the seed, and the first divergent byte so the bug can be replayed
/// deterministically.
pub(crate) fn compare_outputs(
    op_id: &str,
    program: &Program,
    cpu: &[Vec<u8>],
    gpu: &[Vec<u8>],
    tolerance: u32,
    seed: u64,
) -> Result<(), String> {
    if cpu.len() != gpu.len() {
        return Err(format!(
            "{op_id} seed={seed}: CPU produced {} buffers, GPU produced {}",
            cpu.len(),
            gpu.len()
        ));
    }
    let output_indices = program.output_buffer_indices();
    if output_indices.len() != cpu.len() {
        return Err(format!(
            "{op_id} seed={seed}: program declares {} output buffers, CPU produced {}",
            output_indices.len(),
            cpu.len()
        ));
    }
    for (buf_idx, ((c, g), buffer_index)) in cpu
        .iter()
        .zip(gpu.iter())
        .zip(output_indices.iter().copied())
        .enumerate()
    {
        if c.len() != g.len() {
            return Err(format!(
                "{op_id} seed={seed}: buffer #{buf_idx} length diverged CPU={} GPU={}",
                c.len(),
                g.len()
            ));
        }
        let element = program.buffers()[buffer_index as usize].element();
        if element != DataType::F32 || tolerance == 0 {
            for (byte_idx, (cb, gb)) in c.iter().zip(g.iter()).enumerate() {
                if cb != gb {
                    return Err(format!(
                        "{op_id} seed={seed}: byte mismatch at buffer #{buf_idx} byte {byte_idx} \
                         CPU=0x{cb:02x} GPU=0x{gb:02x}"
                    ));
                }
            }
        } else {
            if c.len() % 4 != 0 {
                return Err(format!(
                    "{op_id} seed={seed}: buffer #{buf_idx} length {} not 4-byte aligned for f32 ULP compare",
                    c.len()
                ));
            }
            for (lane, (c_chunk, g_chunk)) in c.chunks_exact(4).zip(g.chunks_exact(4)).enumerate() {
                let c_bits = u32::from_le_bytes(c_chunk.try_into().expect("4 bytes"));
                let g_bits = u32::from_le_bytes(g_chunk.try_into().expect("4 bytes"));
                let c_f = f32::from_bits(c_bits);
                let g_f = f32::from_bits(g_bits);

                // NaN payloads must match exactly regardless of tolerance.
                if c_f.is_nan() || g_f.is_nan() {
                    if c_bits != g_bits {
                        let byte_idx = lane * 4;
                        return Err(format!(
                            "{op_id} seed={seed}: NaN payload mismatch at buffer #{buf_idx} byte {byte_idx} \
                             (lane {lane}) CPU bits=0x{c_bits:08x} GPU bits=0x{g_bits:08x}"
                        ));
                    }
                    continue;
                }

                let diff = ordered_bits(c_f).abs_diff(ordered_bits(g_f));
                if diff > tolerance {
                    let byte_idx = lane * 4;
                    return Err(format!(
                        "{op_id} seed={seed}: ULP divergence at buffer #{buf_idx} byte {byte_idx} \
                         (lane {lane}) diff={diff} > tolerance={tolerance} \
                         CPU bits=0x{c_bits:08x} GPU bits=0x{g_bits:08x}"
                    ));
                }
            }
        }
    }
    Ok(())
}

/// The read-only ProgramGraph bundle is producer data every graph op consumes
/// under one ABI, so the sweep holds all five members at their fixture whatever
/// order the op declares them in.
///
/// Derived from the live catalog rather than from a list of graph ops, so an op
/// added later that declares its own buffers before the bundle is covered
/// without an edit here. The bundle reaches the catalog only under the `graph`
/// and `security` selections, so the non-vacuous proof of the guard itself is
/// `neither_graph_detector_reads_a_buffer_position` below; this case is the
/// sweep over whatever real ops the selection registers.
#[test]
fn no_program_graph_bundle_member_is_ever_randomized() {
    let mut programs = 0usize;
    for entry in vyre_libs::operation_catalog::fixture_entries() {
        let Some(program) = entry.program() else {
            continue;
        };
        programs += 1;
        for (index, buffer) in program.buffers().iter().enumerate() {
            if !READ_ONLY_BUFFER_NAMES.contains(&buffer.name()) {
                continue;
            }
            assert!(
                !randomize_buffer(entry.id, &program, index),
                "{}: ProgramGraph bundle member `{}` at index {index} is randomized. \
                 Fix: hold every name in READ_ONLY_BUFFER_NAMES, at any declaration order.",
                entry.id,
                buffer.name()
            );
        }
    }
    assert!(
        programs > 0,
        "the operation catalog registered nothing. Fix: this sweep asserts \
         nothing, so the catalog or its feature selection is broken."
    );
}

/// Both graph detectors answer on buffer identity and shape, never on buffer
/// position.
///
/// A primitive may declare its own buffers before the read-only bundle, and one
/// does: `security::flows_to_with_sanitizer` emits its sanitizer difference
/// first, which put its bundle at index 3 and its `source` frontier at 0. The
/// case is built here rather than taken from that op so it holds under every
/// feature selection this crate is tested with.
#[test]
fn neither_graph_detector_reads_a_buffer_position() {
    let words = 1_u32;
    let node_count = 4_u32;
    let word = |name: &str, binding: u32, count: u32| {
        BufferDecl::storage(name, binding, BufferAccess::ReadOnly, DataType::U32).with_count(count)
    };
    let mut buffers = vec![
        // A frontier ahead of the bundle, the shape that broke both guards.
        word("source", 0, words),
        word("sanitizer", 1, words),
    ];
    for (offset, name) in READ_ONLY_BUFFER_NAMES.iter().enumerate() {
        let count = if name.contains("edge_offsets") {
            node_count + 1
        } else {
            node_count
        };
        buffers.push(word(name, 2 + offset as u32, count));
    }
    let program = Program::wrapped(
        buffers,
        [vyre_foundation::ir::PORTABLE_WORKGROUP_INVOCATIONS, 1, 1],
        Vec::new(),
    );

    for (index, name) in READ_ONLY_BUFFER_NAMES.iter().enumerate() {
        let slot = index + 2;
        assert!(
            is_program_graph_topology(&program, slot),
            "bundle member `{name}` at index {slot} is not recognized as topology"
        );
        assert!(
            !randomize_buffer("test::position", &program, slot),
            "bundle member `{name}` at index {slot} is randomized"
        );
    }
    for (slot, name) in [(0, "source"), (1, "sanitizer")] {
        assert!(
            is_program_graph_frontier(&program, slot),
            "`{name}` at index {slot} is not recognized as a node bitset, so it \
             would be randomized as an arbitrary word and set bits above the \
             node count"
        );
    }
}
