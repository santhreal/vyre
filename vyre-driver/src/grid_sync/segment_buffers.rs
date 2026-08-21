//! Per-segment buffer tables for the host split: which buffers a segment
//! reads, writes, or accumulates, and the access roles that follow from that.

use std::collections::{HashMap, HashSet};

use vyre_foundation::ir::{BufferAccess, BufferDecl, Expr, Ident, MemoryKind, Node, Program};
use vyre_foundation::visit::{for_each_node, node_buffer_refs, node_operands};

use super::{
    entry_sequence, reserve_grid_sync_hash_map, reserve_grid_sync_hash_set, reserve_grid_sync_vec,
    try_split_on_grid_sync,
};
use crate::backend::BackendError;

pub(super) struct PlannedGridSyncSegment {
    pub(super) program: Program,
    pub(super) input_names: Vec<Ident>,
    pub(super) output_names: Vec<Ident>,
}

/// Diagnostics: the host-split segment **programs** (post buffer-rewrite) that
/// the host-split dispatch path (`dispatch_with_grid_sync_split*`) validates and
/// launches when the backend lacks native grid-sync. Exposed so tooling and
/// tests can inspect or validate each segment without a live backend, the
/// raw [`try_split_on_grid_sync`] output omits the per-segment buffer
/// access/role rewrite, so it is not what the backend actually sees.
///
/// # Errors
/// Propagates any [`BackendError`] from splitting or buffer rewriting.
pub fn plan_host_grid_sync_segment_programs(
    program: &Program,
) -> Result<Vec<Program>, BackendError> {
    Ok(plan_host_grid_sync_segments(program)?
        .into_iter()
        .map(|segment| segment.program)
        .collect())
}

pub(super) fn plan_host_grid_sync_segments(
    program: &Program,
) -> Result<Vec<PlannedGridSyncSegment>, BackendError> {
    let split = try_split_on_grid_sync(program)?;
    let first_writer = first_writer_segment_per_buffer(&split, program)?;
    let mut planned = Vec::new();
    reserve_grid_sync_vec(&mut planned, split.len(), "grid-sync planned host segments")?;
    for (segment_idx, segment) in split.into_iter().enumerate() {
        let rewritten =
            rewrite_segment_buffers_for_host_split(program, &segment, segment_idx, &first_writer)?;
        let input_names = segment_input_names(&rewritten)?;
        let output_names = segment_output_names(&rewritten)?;
        planned.push(PlannedGridSyncSegment {
            program: rewritten,
            input_names,
            output_names,
        });
    }
    Ok(planned)
}

/// For each buffer name, the index of the FIRST split segment that writes it.
///
/// A source-output buffer written by more than one segment is an
/// **accumulator**: each segment writes only its own slots (e.g. a fused
/// multi-rule `results_packed`, where every rule's result-store lands in a
/// different grid-sync segment). A LATER writer must therefore read+merge the
/// value forwarded from earlier segments via `current_inputs`, never overwrite
/// it with a fresh WriteOnly buffer, which would silently zero every earlier
/// segment's slots (recall=0 for every rule whose store is not in the final
/// segment). `rewrite_segment_buffers_for_host_split` uses this map to keep an
/// already-produced output buffer as a `ReadWrite` accumulator in later
/// segments instead of a write-only output.
fn first_writer_segment_per_buffer(
    split: &[Program],
    program: &Program,
) -> Result<HashMap<Ident, usize>, BackendError> {
    let mut first_writer: HashMap<Ident, usize> = HashMap::new();
    reserve_grid_sync_hash_map(
        &mut first_writer,
        program.buffers().len(),
        "grid-sync first-writer map",
    )?;
    for (segment_idx, segment) in split.iter().enumerate() {
        let mut reads = HashSet::new();
        let mut writes = HashSet::new();
        reserve_grid_sync_hash_set(
            &mut reads,
            program.buffers().len(),
            "grid-sync first-writer read scan",
        )?;
        reserve_grid_sync_hash_set(
            &mut writes,
            program.buffers().len(),
            "grid-sync first-writer write scan",
        )?;
        if collect_segment_buffer_targets(entry_sequence(segment), &mut reads, &mut writes) {
            for name in writes {
                first_writer.entry(name).or_insert(segment_idx);
            }
        } else {
            // A segment carrying an extension node names buffers this scan
            // cannot see. Treating it as the first writer of every declared
            // buffer keeps a later real writer reading the forwarded value
            // instead of overwriting it.
            for buffer in program.buffers() {
                first_writer
                    .entry(Ident::from(buffer.name()))
                    .or_insert(segment_idx);
            }
        }
    }
    Ok(first_writer)
}

fn rewrite_segment_buffers_for_host_split(
    source: &Program,
    segment: &Program,
    segment_idx: usize,
    first_writer: &HashMap<Ident, usize>,
) -> Result<Program, BackendError> {
    let mut reads = HashSet::new();
    let mut writes = HashSet::new();
    reserve_grid_sync_hash_set(
        &mut reads,
        source.buffers().len(),
        "grid-sync segment read set",
    )?;
    reserve_grid_sync_hash_set(
        &mut writes,
        source.buffers().len(),
        "grid-sync segment write set",
    )?;
    let complete = collect_segment_buffer_targets(entry_sequence(segment), &mut reads, &mut writes);
    if !complete {
        // An extension node names buffers no walk can enumerate. Dropping a
        // declaration the segment still references produces a segment program
        // that fails lowering, so keep the whole source table read-write.
        for buffer in source.buffers() {
            let name = Ident::from(buffer.name());
            reads.insert(name.clone());
            writes.insert(name);
        }
    }

    let mut buffers = Vec::new();
    reserve_grid_sync_vec(
        &mut buffers,
        source.buffers().len(),
        "grid-sync segment buffers",
    )?;
    for buffer in source.buffers() {
        let name = Ident::from(buffer.name());
        let reads_this = reads.contains(&name);
        let writes_this = writes.contains(&name);
        let readwrite_passthrough = matches!(buffer.access(), BufferAccess::ReadWrite)
            && !buffer.is_output()
            && !buffer.is_pipeline_live_out()
            && !reads_this
            && !writes_this;

        if !reads_this && !writes_this && !readwrite_passthrough {
            continue;
        }

        let mut rewritten = buffer.clone();
        if matches!(rewritten.access(), BufferAccess::Workgroup) {
            buffers.push(rewritten);
            continue;
        }

        // A source-output buffer that an EARLIER segment already wrote is an
        // accumulator across the split: this segment must read the value
        // forwarded via `current_inputs` and merge its own slots, never
        // overwrite it with a fresh WriteOnly buffer (which zeroes the earlier
        // segments' slots, the silent recall=0 mode for any fused rule whose
        // result-store does not land in the final segment).
        // `is_output()` alone missed a plain `WriteOnly` result buffer, which is
        // precisely the fresh-WriteOnly case this guard exists to prevent.
        let is_source_output = buffer.is_backend_allocated_output() || buffer.is_pipeline_live_out();
        let earlier_segment_wrote_output = is_source_output
            && first_writer
                .get(&name)
                .is_some_and(|&first| first < segment_idx);

        let access = if readwrite_passthrough {
            BufferAccess::ReadWrite
        } else if earlier_segment_wrote_output && writes_this {
            // Later writer of a multi-segment output accumulator: read the
            // accumulated prior value (uploaded as input) and merge this
            // segment's slots in place.
            BufferAccess::ReadWrite
        } else {
            match (reads_this, writes_this) {
                (true, true) => BufferAccess::ReadWrite,
                (true, false) => BufferAccess::ReadOnly,
                (false, true) => BufferAccess::WriteOnly,
                (false, false) => BufferAccess::ReadWrite,
            }
        };
        rewrite_segment_buffer_access(&mut rewritten, access);
        // Never mark a split segment's buffer as the program output: a
        // multi-segment output accumulator must CONSUME its forwarded prior
        // value as input in later segments, and `segment_buffer_consumes_input`
        // refuses any `is_output` buffer. Each writing segment still produces
        // the buffer (WriteOnly/ReadWrite both produce output), so its bytes
        // are captured into `current_inputs`; the final host-visible values are
        // reassembled by name from the SOURCE program's output set in
        // `collect_final_named_outputs`, independent of any per-segment flag.
        rewritten.is_output = false;
        rewritten.pipeline_live_out = false;
        buffers.push(rewritten);
    }

    Ok(segment.with_rewritten_buffers(buffers))
}

fn rewrite_segment_buffer_access(buffer: &mut BufferDecl, access: BufferAccess) {
    buffer.kind = match &access {
        BufferAccess::ReadOnly => MemoryKind::Readonly,
        BufferAccess::Uniform => MemoryKind::Uniform,
        BufferAccess::Workgroup => MemoryKind::Shared,
        _ => MemoryKind::Global,
    };
    buffer.access = access;
}

pub(super) fn segment_input_names(segment: &Program) -> Result<Vec<Ident>, BackendError> {
    let mut names = Vec::new();
    reserve_grid_sync_vec(
        &mut names,
        segment.buffers().len(),
        "grid-sync segment input names",
    )?;
    for buffer in segment.buffers() {
        if matches!(buffer.access(), BufferAccess::Workgroup) {
            continue;
        }
        if segment_buffer_consumes_input(buffer) {
            names.push(Ident::from(buffer.name()));
        }
    }
    Ok(names)
}

pub(super) fn segment_output_names(segment: &Program) -> Result<Vec<Ident>, BackendError> {
    let mut names = Vec::new();
    reserve_grid_sync_vec(
        &mut names,
        segment.buffers().len(),
        "grid-sync segment output names",
    )?;
    for buffer in segment.buffers() {
        if matches!(buffer.access(), BufferAccess::Workgroup) {
            continue;
        }
        if segment_buffer_produces_output(buffer) {
            names.push(Ident::from(buffer.name()));
        }
    }
    Ok(names)
}

pub(super) fn original_input_names(program: &Program) -> Result<Vec<Ident>, BackendError> {
    segment_input_names(program)
}

pub(super) fn original_output_names(program: &Program) -> Result<Vec<Ident>, BackendError> {
    segment_output_names(program)
}

/// Whether a segment reads this buffer from the dispatch inputs.
///
/// `is_backend_allocated_output` is the single cross-backend definition of a
/// buffer the backend allocates and writes rather than reads. Spelling it out
/// again here is how the interpreter and a backend end up disagreeing about
/// which buffers carry host bytes.
pub(super) fn segment_buffer_consumes_input(buffer: &BufferDecl) -> bool {
    if buffer.is_backend_allocated_output() || buffer.is_pipeline_live_out() {
        return false;
    }
    matches!(
        buffer.access(),
        BufferAccess::ReadOnly | BufferAccess::ReadWrite | BufferAccess::Uniform
    )
}

/// Whether a segment leaves a result in this buffer.
///
/// Wider than [`BufferDecl::is_backend_allocated_output`] by the `ReadWrite`
/// case: a segment that updates a buffer in place produces a value the next
/// segment reads, whether or not the whole program returns it.
pub(super) fn segment_buffer_produces_output(buffer: &BufferDecl) -> bool {
    buffer.is_backend_allocated_output()
        || buffer.is_pipeline_live_out()
        || matches!(buffer.access(), BufferAccess::ReadWrite)
}

/// The buffers `nodes` reads and writes, at any nesting depth.
///
/// Every answer comes from an owner. [`node_buffer_refs`] owns "what does this
/// statement do to a buffer BY NAME" and fails to compile when a `Node` variant
/// is added; [`node_operands`] plus [`collect_segment_expr_targets`] own the
/// buffers an operand expression reaches. The version that restated the
/// question here as a per-variant match ending in `_ => {}` named `Store` and
/// the four collectives and silently reported that an `AsyncLoad` touches
/// nothing: its source buffer then matched no segment role, its declaration was
/// dropped from the segment table, and the surviving node referenced a buffer
/// the segment no longer declared.
///
/// Returns `false` when a node refuses to name its buffers (`Node::Opaque`), so
/// the caller can keep every declaration rather than trust an incomplete set.
fn collect_segment_buffer_targets(
    nodes: &[Node],
    reads: &mut HashSet<Ident>,
    writes: &mut HashSet<Ident>,
) -> bool {
    let mut complete = true;
    for_each_node(nodes, |node| {
        let refs = node_buffer_refs(node);
        complete &= refs.complete;
        for buffer in refs.reads.into_iter().flatten() {
            reads.insert(buffer.clone());
        }
        for buffer in refs.writes.into_iter().flatten() {
            writes.insert(buffer.clone());
        }
        for operand in node_operands(node).into_iter().flatten() {
            collect_segment_expr_targets(operand, reads, writes);
        }
    });
    complete
}

fn collect_segment_expr_targets(
    expr: &Expr,
    reads: &mut HashSet<Ident>,
    writes: &mut HashSet<Ident>,
) {
    vyre_foundation::visit::visit_expr_buffer_accesses(expr, |access, buffer| {
        reads.insert(buffer.clone());
        if access == vyre_foundation::visit::ExprBufferAccess::Atomic {
            writes.insert(buffer.clone());
        }
    });
}

// Inline: covers `segment_buffer_consumes_input`, `segment_input_names`, which no integration test
// can name.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_sync::test_programs::cross_segment_store_program;

    #[test]
    fn split_keeps_multi_segment_output_as_readwrite_accumulator() {
        // An OUTPUT buffer whose slots are written by DIFFERENT grid-sync
        // segments (the fused multi-rule `results_packed` shape: each rule's
        // result-store lands in its own segment) must ACCUMULATE across the host
        // split. The first writer establishes it (WriteOnly); every LATER writer
        // must read the forwarded value and merge its own slots (ReadWrite)
        // instead of overwriting it with a fresh write-only buffer, which would
        // silently zero the earlier segments' slots (recall=0 for every rule
        // whose store is not in the final segment).
        let program = cross_segment_store_program();
        let segments =
            plan_host_grid_sync_segment_programs(&program).expect("plan host grid-sync segments");
        assert_eq!(segments.len(), 2, "one GridSync barrier -> two segments");

        let seg0_out = segments[0]
            .buffers()
            .iter()
            .find(|b| b.name() == "out")
            .expect("segment 0 must declare the output it writes");
        assert_eq!(
            seg0_out.access(),
            BufferAccess::WriteOnly,
            "the first writer establishes the accumulator as write-only"
        );
        assert!(
            !seg0_out.is_output() && !seg0_out.is_pipeline_live_out(),
            "split segment buffers must never be marked program-output; final values are reassembled by name"
        );

        let seg1_out = segments[1]
            .buffers()
            .iter()
            .find(|b| b.name() == "out")
            .expect("segment 1 must declare the output it writes");
        assert_eq!(
            seg1_out.access(),
            BufferAccess::ReadWrite,
            "a later writer of a multi-segment output must read+merge the accumulated value, not overwrite it"
        );
        assert!(
            !seg1_out.is_output() && !seg1_out.is_pipeline_live_out(),
            "the later writer must consume its forwarded prior value, which `segment_buffer_consumes_input` refuses for is_output buffers"
        );
        assert!(
            segment_input_names(&segments[1])
                .expect("segment 1 input names")
                .iter()
                .any(|n| n.as_str() == "out"),
            "the accumulated output must be forwarded as an input to the later writing segment"
        );
    }
}
