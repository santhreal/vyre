// RELEASE PROOF LANE 4  -  pairwise op composition proptest.
//
// Generates random pairwise compositions from the Cat-A operator catalog,
// fuses them via `vyre_foundation::execution_plan::fusion::fuse_programs`,
// and asserts CPU-reference vs GPU-backend parity on the fused Program.
//
// **Proving test** (`pairwise_composition_parity`): only compatible dtype
// signatures are exercised; every passing case proves that sequential
// buffer-wired composition is sound.
//
// **Adversarial test** (`pairwise_composition_adversarial`): unfiltered pairs
// hit `try_compose`.  Incompatible pairs must return `Err`  -  never panic,
// never produce a silent-wrong Program.
//
// Coverage: `vyre_libs::harness::all_entries()`.

use std::collections::HashSet;
use std::sync::{Arc, LazyLock};

use proptest::prelude::*;
use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::execution_plan::fusion::fuse_programs;
use vyre_libs::harness::fp_contract;
use vyre_reference::value::Value;

// ------------------------------------------------------------------
// Catalog
// ------------------------------------------------------------------

/// Unified view of a harness entry so the proptest can treat the whole
/// catalog as a flat vector.
struct UnifiedEntry {
    id: &'static str,
    build: fn() -> Program,
    #[allow(clippy::type_complexity)]
    test_inputs: Option<fn() -> Vec<Vec<Vec<u8>>>>,
}

fn all_entries_vec() -> Vec<UnifiedEntry> {
    let mut out = Vec::new();
    for e in vyre_libs::harness::all_entries() {
        out.push(UnifiedEntry {
            id: e.id,
            build: e.build,
            test_inputs: e.test_inputs,
        });
    }
    out
}

fn entry_count() -> usize {
    all_entries_vec().len()
}

fn entry_by_index(idx: usize) -> &'static UnifiedEntry {
    static ENTRIES: LazyLock<Vec<UnifiedEntry>> = LazyLock::new(all_entries_vec);
    ENTRIES.get(idx).expect("Fix: entry index out of bounds")
}

// ------------------------------------------------------------------
// GPU backend probe (lazy, fatal when absent)
// ------------------------------------------------------------------

fn gpu() -> &'static WgpuBackend {
    static GPU: LazyLock<WgpuBackend> = LazyLock::new(|| {
        WgpuBackend::acquire().unwrap_or_else(|error| {
            panic!(
                "Fix: pairwise GPU parity could not acquire WGPU backend on a GPU-required host: {error}"
            )
        })
    });
    &GPU
}

/// Validate a fused program against the backend it is about to run on.
///
/// A bare `Program::validate` assumes no backend and rejects every subgroup
/// expression with V041, which says in as many words to validate with the
/// backend instead. Composing two ops that each use subgroup operations is not
/// an error when the device supports them, and this harness dispatches to a
/// device it has already acquired, so the backend-aware form is the only one
/// that describes the program actually under test.
fn validate_for_backend(program: &Program) -> Result<(), String> {
    let options = vyre_foundation::validate::ValidationOptions::default().with_backend(gpu());
    let report = vyre_foundation::validate::validate_with_options(program, options);
    if report.errors.is_empty() {
        return Ok(());
    }
    Err(report
        .errors
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; "))
}

fn missing_capability_reason(program: &Program) -> Option<String> {
    let required = vyre_foundation::program_caps::scan(program);
    let backend = gpu();
    vyre_foundation::program_caps::check_backend_capabilities(
        backend.id(),
        backend.supports_subgroup_ops(),
        backend.supports_f16(),
        backend.supports_bf16(),
        backend.supports_indirect_dispatch(),
        true,
        backend.supports_distributed_collectives(),
        backend.max_workgroup_size(),
        &required,
    )
    .err()
    .map(|e| e.to_string())
}

// ------------------------------------------------------------------
// Buffer renaming helpers (composition wiring)
// ------------------------------------------------------------------

fn rename_buffer_in_expr(expr: &Expr, old: &str, new: &str) -> Expr {
    match expr {
        Expr::Load { buffer, index } => Expr::Load {
            buffer: if buffer.as_str() == old {
                new.into()
            } else {
                buffer.clone()
            },
            index: Box::new(rename_buffer_in_expr(index, old, new)),
        },
        Expr::BufLen { buffer } => Expr::BufLen {
            buffer: if buffer.as_str() == old {
                new.into()
            } else {
                buffer.clone()
            },
        },
        Expr::Atomic {
            op,
            buffer,
            index,
            expected,
            value,
            ordering,
        } => Expr::Atomic {
            op: *op,
            buffer: if buffer.as_str() == old {
                new.into()
            } else {
                buffer.clone()
            },
            index: Box::new(rename_buffer_in_expr(index, old, new)),
            expected: expected
                .as_ref()
                .map(|e| Box::new(rename_buffer_in_expr(e, old, new))),
            value: Box::new(rename_buffer_in_expr(value, old, new)),
            ordering: *ordering,
        },
        Expr::BinOp { op, left, right } => Expr::BinOp {
            op: *op,
            left: Box::new(rename_buffer_in_expr(left, old, new)),
            right: Box::new(rename_buffer_in_expr(right, old, new)),
        },
        Expr::UnOp { op, operand } => Expr::UnOp {
            op: op.clone(),
            operand: Box::new(rename_buffer_in_expr(operand, old, new)),
        },
        Expr::Call { op_id, args } => Expr::Call {
            op_id: op_id.clone(),
            args: args
                .iter()
                .map(|a| rename_buffer_in_expr(a, old, new))
                .collect(),
        },
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => Expr::Select {
            cond: Box::new(rename_buffer_in_expr(cond, old, new)),
            true_val: Box::new(rename_buffer_in_expr(true_val, old, new)),
            false_val: Box::new(rename_buffer_in_expr(false_val, old, new)),
        },
        Expr::Cast { target, value } => Expr::Cast {
            target: target.clone(),
            value: Box::new(rename_buffer_in_expr(value, old, new)),
        },
        Expr::Fma { a, b, c } => Expr::Fma {
            a: Box::new(rename_buffer_in_expr(a, old, new)),
            b: Box::new(rename_buffer_in_expr(b, old, new)),
            c: Box::new(rename_buffer_in_expr(c, old, new)),
        },
        Expr::SubgroupBallot { cond } => Expr::SubgroupBallot {
            cond: Box::new(rename_buffer_in_expr(cond, old, new)),
        },
        Expr::SubgroupShuffle { value, lane } => Expr::SubgroupShuffle {
            value: Box::new(rename_buffer_in_expr(value, old, new)),
            lane: Box::new(rename_buffer_in_expr(lane, old, new)),
        },
        Expr::SubgroupReduce { op, value } => Expr::SubgroupReduce {
            op: *op,
            value: Box::new(rename_buffer_in_expr(value, old, new)),
        },
        // Leaf expressions  -  no buffers inside.
        _ => expr.clone(),
    }
}

fn rename_buffer_in_node(node: &Node, old: &str, new: &str) -> Node {
    match node {
        Node::Let { name, value } => Node::Let {
            name: name.clone(),
            value: rename_buffer_in_expr(value, old, new),
        },
        Node::Assign { name, value } => Node::Assign {
            name: name.clone(),
            value: rename_buffer_in_expr(value, old, new),
        },
        Node::Store {
            buffer,
            index,
            value,
        } => Node::Store {
            buffer: if buffer.as_str() == old {
                new.into()
            } else {
                buffer.clone()
            },
            index: rename_buffer_in_expr(index, old, new),
            value: rename_buffer_in_expr(value, old, new),
        },
        Node::If {
            cond,
            then,
            otherwise,
        } => Node::If {
            cond: rename_buffer_in_expr(cond, old, new),
            then: then
                .iter()
                .map(|n| rename_buffer_in_node(n, old, new))
                .collect(),
            otherwise: otherwise
                .iter()
                .map(|n| rename_buffer_in_node(n, old, new))
                .collect(),
        },
        Node::Loop {
            var,
            from,
            to,
            body,
        } => Node::Loop {
            var: var.clone(),
            from: rename_buffer_in_expr(from, old, new),
            to: rename_buffer_in_expr(to, old, new),
            body: body
                .iter()
                .map(|n| rename_buffer_in_node(n, old, new))
                .collect(),
        },
        Node::Block(nodes) => Node::Block(
            nodes
                .iter()
                .map(|n| rename_buffer_in_node(n, old, new))
                .collect(),
        ),
        Node::Region {
            generator,
            source_region,
            body,
        } => Node::Region {
            generator: generator.clone(),
            source_region: source_region.clone(),
            body: Arc::new(
                body.iter()
                    .map(|n| rename_buffer_in_node(n, old, new))
                    .collect(),
            ),
        },
        Node::IndirectDispatch {
            count_buffer,
            count_offset,
        } => Node::IndirectDispatch {
            count_buffer: if count_buffer.as_str() == old {
                new.into()
            } else {
                count_buffer.clone()
            },
            count_offset: *count_offset,
        },
        Node::AsyncLoad {
            source,
            destination,
            offset,
            size,
            tag,
        } => Node::AsyncLoad {
            source: if source.as_str() == old {
                new.into()
            } else {
                source.clone()
            },
            destination: if destination.as_str() == old {
                new.into()
            } else {
                destination.clone()
            },
            offset: Box::new(rename_buffer_in_expr(offset, old, new)),
            size: Box::new(rename_buffer_in_expr(size, old, new)),
            tag: tag.clone(),
        },
        Node::AsyncStore {
            source,
            destination,
            offset,
            size,
            tag,
        } => Node::AsyncStore {
            source: if source.as_str() == old {
                new.into()
            } else {
                source.clone()
            },
            destination: if destination.as_str() == old {
                new.into()
            } else {
                destination.clone()
            },
            offset: Box::new(rename_buffer_in_expr(offset, old, new)),
            size: Box::new(rename_buffer_in_expr(size, old, new)),
            tag: tag.clone(),
        },
        Node::Trap { address, tag } => Node::Trap {
            address: Box::new(rename_buffer_in_expr(address, old, new)),
            tag: tag.clone(),
        },
        // Catch-all for extension variants this wiring pass does not rewrite.
        _ => node.clone(),
    }
}

/// Rename one buffer throughout a program, preserving program metadata.
///
/// This rebuilds through `with_rewritten_buffers` / `with_rewritten_entry`
/// rather than `Program::wrapped`. `wrapped` constructs a fresh program and
/// resets the metadata flags, so a rename silently cleared `entry_op_id` and
/// `non_composable_with_self`. Clearing the composability flag is the worst of
/// the two: the harness would then fuse two copies of a self-exclusive region
/// and only find out at validation time, reported as a composition bug rather
/// than as the pair being incompatible.
fn rename_buffer_in_program(prog: &Program, old: &str, new: &str) -> Program {
    let buffers: Vec<BufferDecl> = prog
        .buffers()
        .iter()
        .map(|buf| {
            let mut b = buf.clone();
            if b.name.as_ref() == old {
                b.name = Arc::from(new);
            }
            b
        })
        .collect();
    let entry: Vec<Node> = prog
        .entry()
        .iter()
        .map(|n| rename_buffer_in_node(n, old, new))
        .collect();
    prog.with_rewritten_buffers(buffers).with_rewritten_entry(entry)
}

/// Demote a buffer from a program result to an internal intermediate.
///
/// When op_a's output is piped into op_b's input, that buffer stops being a
/// result of the composition: it is a value handed from one stage to the next,
/// and only op_b's own output leaves the fused program. Leaving it marked as
/// an output produced a program with two output buffers, which the wire-format
/// validator rejects (V022, "program declares 2 output buffers"). The
/// compatibility check reported the pair as composable and the fused program
/// then failed to validate, which is the contradiction the pairwise tests were
/// reporting.
///
/// The demotion belongs here rather than in `fuse_programs`. That fuser also
/// serves independent batch arms, where several outputs are exactly right; it
/// is the PIPE that makes this particular buffer internal, and only the caller
/// wiring the pipe knows that.
fn demote_output_to_intermediate(prog: &Program, name: &str) -> Program {
    let buffers: Vec<BufferDecl> = prog
        .buffers()
        .iter()
        .map(|buf| {
            let mut buf = buf.clone();
            if buf.name.as_ref() == name {
                buf.is_output = false;
                buf.pipeline_live_out = false;
                buf.output_byte_range = None;
            }
            buf
        })
        .collect();
    prog.with_rewritten_buffers(buffers)
}

// ------------------------------------------------------------------
// Composition logic
// ------------------------------------------------------------------

/// A fused pair together with the wiring that produced it.
///
/// The input assembly needs to know how op_b's buffers were renamed on the way
/// into the fused program. Returning that map alongside the program keeps the
/// two halves in step. The previous shape returned the program alone and the
/// caller re-derived the wiring by zipping op_b's declaration list against its
/// witness vector, which lined up only when op_b declared no outputs and no
/// workgroup buffers before its last input.
struct Composition {
    /// The fused program, ready to validate and dispatch.
    program: Program,
    /// op_a, as built, before any renaming.
    prog_a: Program,
    /// op_b, as built, before any renaming.
    prog_b: Program,
    /// op_b buffer name to the name it carries inside `program`.
    b_renames: Vec<(String, String)>,
    /// op_a output name used as the pipe between both programs.
    wired_name: String,
}

/// Does the reference interpreter require a caller-supplied value for this
/// buffer?
///
/// Read from the interpreter rather than restated here. The obvious restatement
/// is `!buf.is_output() && buf.access() != BufferAccess::Workgroup`, which is a
/// different predicate: the interpreter keys on `is_backend_allocated_output`,
/// and where the two disagree every later input slides by one.
fn needs_input(buf: &BufferDecl) -> bool {
    vyre_reference::is_reference_input(buf)
}

fn witness_uses_legacy_abi(program: &Program, case_len: usize) -> Option<bool> {
    let logical_count = program
        .buffers()
        .iter()
        .filter(|buffer| needs_input(buffer))
        .count();
    let legacy_count = program
        .buffers()
        .iter()
        .filter(|buffer| buffer.access() != BufferAccess::Workgroup)
        .count();

    if case_len == legacy_count && case_len != logical_count {
        Some(true)
    } else if case_len == logical_count {
        Some(false)
    } else {
        None
    }
}

fn input_witness_len(program: &Program, case: &[Vec<u8>], name: &str) -> Option<usize> {
    let legacy = witness_uses_legacy_abi(program, case.len())?;
    program
        .buffers()
        .iter()
        .filter(|buffer| {
            buffer.access() != BufferAccess::Workgroup && (legacy || needs_input(buffer))
        })
        .zip(case)
        .find(|(buffer, _)| buffer.name() == name)
        .map(|(_, bytes)| bytes.len())
}

/// Attempt to compose `op_a` followed by `op_b` via shared-buffer fusion.
///
/// Returns `Ok(fused_program)` when:
/// * `op_a` has exactly one ReadWrite output buffer,
/// * `op_b` has at least one ReadOnly/Uniform input buffer,
/// * the output/input element types match,
/// * the output/input counts are compatible (both zero, or equal, or one is zero
///   and the test-input byte lengths line up).
///
/// Returns `Err(reason)` for every other pair so the adversarial test can
/// assert clean rejection.
fn try_compose(a: &UnifiedEntry, b: &UnifiedEntry) -> Result<Composition, String> {
    let prog_a = (a.build)();
    let prog_b = (b.build)();

    // ---- op_a output analysis ----
    let a_outputs: Vec<&BufferDecl> = prog_a
        .buffers()
        .iter()
        .filter(|buf| buf.access() == BufferAccess::ReadWrite)
        .collect();
    if a_outputs.is_empty() {
        return Err(format!(
            "Fix: {} has no ReadWrite output buffer; cannot wire into downstream op.",
            a.id
        ));
    }
    let explicit_outputs: Vec<&BufferDecl> = a_outputs
        .iter()
        .copied()
        .filter(|buf| buf.is_output())
        .collect();
    let a_out = match (explicit_outputs.as_slice(), a_outputs.as_slice()) {
        ([output], _) => *output,
        ([], [output]) => *output,
        ([], outputs) => {
            return Err(format!(
                "Fix: {} has {} ReadWrite buffers and no explicit output; mark the intended pipeline result with BufferDecl::output before composing.",
                a.id,
                outputs.len()
            ));
        }
        (outputs, _) => {
            return Err(format!(
                "Fix: {} has {} explicit outputs; pairwise piping requires exactly one.",
                a.id,
                outputs.len()
            ));
        }
    };

    // ---- op_b input analysis ----
    let b_inputs: Vec<&BufferDecl> = prog_b
        .buffers()
        .iter()
        .filter(|buf| matches!(buf.access(), BufferAccess::ReadOnly | BufferAccess::Uniform))
        .collect();
    if b_inputs.is_empty() {
        return Err(format!(
            "Fix: {} has no ReadOnly/Uniform input buffer; nothing can be wired from the upstream op.",
            b.id
        ));
    }
    let b_in = b_inputs[0];

    // ---- dtype check ----
    if a_out.element() != b_in.element() {
        return Err(format!(
            "Fix: dtype mismatch: {} output={:?} vs {} input={:?}. Add an explicit cast/composition adapter before fusing.",
            a.id,
            a_out.element(),
            b.id,
            b_in.element()
        ));
    }

    // ---- count / shape check ----
    let a_count = a_out.count();
    let b_count = b_in.count();
    if a_count != 0 && b_count != 0 && a_count != b_count {
        return Err(format!(
            "Fix: count mismatch: {} output count={} vs {} input count={}. Add a shape adapter before fusing.",
            a.id, a_count, b.id, b_count
        ));
    }
    if a_count != 0 && b_count == 0 {
        let element_bytes = a_out.element().size_bytes().ok_or_else(|| {
            format!(
                "Fix: {} output uses a dynamically sized element type that cannot be wired into {}'s runtime-sized input.",
                a.id, b.id
            )
        })?;
        let upstream_bytes = usize::try_from(a_count)
            .ok()
            .and_then(|count| count.checked_mul(element_bytes))
            .ok_or_else(|| {
                format!(
                    "Fix: {} output byte length overflows the host address space; split the output before composing it with {}.",
                    a.id, b.id
                )
            })?;
        if let Some(test_inputs) = b.test_inputs {
            for case in test_inputs() {
                if let Some(runtime_bytes) = input_witness_len(&prog_b, &case, b_in.name()) {
                    if runtime_bytes != upstream_bytes {
                        return Err(format!(
                            "Fix: runtime-sized input byte mismatch: {} produces {} bytes but {} witness input `{}` contains {} bytes. Add a shape adapter before fusing.",
                            a.id,
                            upstream_bytes,
                            b.id,
                            b_in.name(),
                            runtime_bytes
                        ));
                    }
                }
            }
        }
    }

    // ---- collision check & rename ----
    let a_names: HashSet<&str> = prog_a.buffers().iter().map(|buf| buf.name()).collect();
    let b_names: HashSet<&str> = prog_b.buffers().iter().map(|buf| buf.name()).collect();

    let mut prog_b_prepared = prog_b.clone();
    let mut b_renames: Vec<(String, String)> = Vec::new();

    // Rename every colliding buffer in op_b (except the wired input) so that
    // `fuse_programs` does not accidentally alias unrelated buffers.
    for colliding in b_names.intersection(&a_names) {
        if *colliding == b_in.name() {
            continue;
        }
        let new_name = format!("b_{}", colliding);
        prog_b_prepared = rename_buffer_in_program(&prog_b_prepared, colliding, &new_name);
        b_renames.push(((*colliding).to_string(), new_name));
    }

    // Wire op_b's first input to op_a's output buffer name.
    prog_b_prepared = rename_buffer_in_program(&prog_b_prepared, b_in.name(), a_out.name());
    b_renames.push((b_in.name().to_string(), a_out.name().to_string()));

    // ---- fuse ----
    let wired_name = a_out.name().to_string();
    let fused = fuse_programs(&[prog_a.clone(), prog_b_prepared])
        .map_err(|e| format!("Fix: fusion failed for {} -> {}: {}", a.id, b.id, e))?;
    let program = demote_output_to_intermediate(&fused, &wired_name);

    // Some regions are declared exclusive with themselves: they carry
    // per-instance scratch that a second copy in the same kernel would stomp.
    // Two ops embedding the same such region cannot be piped together at all,
    // so this is a compatibility verdict, not a defect in the fused program.
    // Reporting it here keeps the downstream assertion honest: a pair that
    // reaches the parity test and then fails validation really is a bug.
    let duplicates =
        vyre_foundation::algebra::composition::duplicate_self_exclusive_regions(program.entry());
    if let Some(generator) = duplicates.first() {
        return Err(format!(
            "Fix: {} -> {} would place two copies of the self-exclusive region `{generator}` in one kernel. Give each instance distinct scratch storage, or run the two stages as separate dispatches.",
            a.id, b.id
        ));
    }

    Ok(Composition {
        program,
        prog_a,
        prog_b,
        b_renames,
        wired_name,
    })
}
