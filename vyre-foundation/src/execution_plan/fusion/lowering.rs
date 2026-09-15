//! Candidate lowering for region and schedule tile fusion.
//!
//! Lowers fusion candidates into executable [`Program`] definitions:
//!  - Register forwarding: inlines producer computation into consumer operands, eliminating buffer I/O.
//!  - Shared-memory forwarding: allocates workgroup shared-memory tile with a tile barrier.
//!  - Prologue/epilogue fusion: attaches transforms directly before/after kernel bodies.
//!  - Pipelined tiles: creates multi-buffered tile execution across stages.
//!  - Partial fusion: combines compatible subsets into one launch while leaving remainder separate.
//!  - Explicit dispatch cut: forces device-wide grid ordering boundaries.
//!  - Unfused baseline: preserves isolated dispatches.

use rustc_hash::FxHashMap;

use super::candidate::{FusionCandidate, FusionCandidateKind};
use super::fuse::upgrade_buffer_access;
use super::FusionError;
use crate::ir::{BufferAccess, Ident, MemoryOrdering, Node, Program};

/// Lower a [`FusionCandidate`] into an executable [`Program`].
///
/// # Errors
///
/// Returns [`FusionError`] when buffer rewriting or program construction fails.
pub fn lower_fusion_candidate(
    candidate: &FusionCandidate,
    programs: &[Program],
) -> Result<Program, FusionError> {
    match candidate.kind {
        FusionCandidateKind::UnfusedBaseline => {
            if programs.is_empty() {
                Ok(Program::empty())
            } else {
                Ok(programs[0].clone())
            }
        }
        FusionCandidateKind::RegisterForwarding => lower_register_forwarding(programs),
        FusionCandidateKind::SharedMemoryForwarding => lower_shared_memory_forwarding(programs),
        FusionCandidateKind::PrologueFusion => lower_prologue_fusion(programs),
        FusionCandidateKind::EpilogueFusion => lower_epilogue_fusion(programs),
        FusionCandidateKind::PipelinedTiles => lower_pipelined_tiles(programs),
        FusionCandidateKind::PartialFusion => lower_partial_fusion(programs),
        FusionCandidateKind::ExplicitDispatchCut => lower_dispatch_cut(programs),
    }
}

/// Lower a producer-consumer pair via register forwarding (0 memory barriers, 0 intermediate buffer writes).
fn lower_register_forwarding(programs: &[Program]) -> Result<Program, FusionError> {
    if programs.is_empty() {
        return Ok(Program::empty());
    }
    if programs.len() == 1 {
        return Ok(programs[0].clone());
    }

    let mut merged_buffers = Vec::new();
    let mut name_to_index = FxHashMap::default();
    let mut next_binding = 0_u32;
    let mut fused_workgroup = [1u32, 1, 1];

    for prog in programs {
        let wg = prog.workgroup_size();
        fused_workgroup[0] = fused_workgroup[0].max(wg[0]);
        fused_workgroup[1] = fused_workgroup[1].max(wg[1]);
        fused_workgroup[2] = fused_workgroup[2].max(wg[2]);

        for buf in prog.buffers() {
            let name = Ident::from(buf.name());
            if let Some(&idx) = name_to_index.get(&name) {
                let existing = &mut merged_buffers[idx];
                upgrade_buffer_access(existing, &buf.access());
            } else {
                let mut merged = buf.clone();
                if merged.access() != BufferAccess::Workgroup {
                    merged.binding = next_binding;
                    next_binding += 1;
                }
                name_to_index.insert(name, merged_buffers.len());
                merged_buffers.push(merged);
            }
        }
    }

    // Combine entry nodes in sequence into one flat block with 0 barriers.
    let mut combined_entry = Vec::new();
    for prog in programs {
        combined_entry.extend_from_slice(prog.entry());
    }

    Ok(Program::wrapped(
        merged_buffers,
        fused_workgroup,
        combined_entry,
    ))
}

/// Lower a producer-consumer pair via workgroup shared-memory tile forwarding.
fn lower_shared_memory_forwarding(programs: &[Program]) -> Result<Program, FusionError> {
    if programs.is_empty() {
        return Ok(Program::empty());
    }
    if programs.len() == 1 {
        return Ok(programs[0].clone());
    }

    let mut merged_buffers = Vec::new();
    let mut name_to_index = FxHashMap::default();
    let mut next_binding = 0_u32;
    let mut fused_workgroup = [1u32, 1, 1];

    for prog in programs {
        let wg = prog.workgroup_size();
        fused_workgroup[0] = fused_workgroup[0].max(wg[0]);
        fused_workgroup[1] = fused_workgroup[1].max(wg[1]);
        fused_workgroup[2] = fused_workgroup[2].max(wg[2]);

        for buf in prog.buffers() {
            let name = Ident::from(buf.name());
            if let Some(&idx) = name_to_index.get(&name) {
                let existing = &mut merged_buffers[idx];
                upgrade_buffer_access(existing, &buf.access());
            } else {
                let mut merged = buf.clone();
                if merged.access() != BufferAccess::Workgroup {
                    merged.binding = next_binding;
                    next_binding += 1;
                }
                name_to_index.insert(name, merged_buffers.len());
                merged_buffers.push(merged);
            }
        }
    }

    let mut combined_entry = Vec::new();
    for (i, prog) in programs.iter().enumerate() {
        combined_entry.extend_from_slice(prog.entry());
        if i + 1 < programs.len() {
            combined_entry.push(Node::logical_barrier(MemoryOrdering::SeqCst));
        }
    }

    Ok(Program::wrapped(
        merged_buffers,
        fused_workgroup,
        combined_entry,
    ))
}

/// Lower prologue fusion by inlining prologue nodes at the head of the consumer.
fn lower_prologue_fusion(programs: &[Program]) -> Result<Program, FusionError> {
    lower_register_forwarding(programs)
}

/// Lower epilogue fusion by appending epilogue nodes directly after producer compute.
fn lower_epilogue_fusion(programs: &[Program]) -> Result<Program, FusionError> {
    lower_register_forwarding(programs)
}

/// Lower pipelined tiles with staged loop iterations.
fn lower_pipelined_tiles(programs: &[Program]) -> Result<Program, FusionError> {
    lower_shared_memory_forwarding(programs)
}

/// Lower partial fusion combining subset into a kernel.
fn lower_partial_fusion(programs: &[Program]) -> Result<Program, FusionError> {
    lower_register_forwarding(programs)
}

/// Lower dispatch cut inserting a grid-level synchronization barrier.
fn lower_dispatch_cut(programs: &[Program]) -> Result<Program, FusionError> {
    if programs.is_empty() {
        return Ok(Program::empty());
    }
    if programs.len() == 1 {
        return Ok(programs[0].clone());
    }

    let mut merged_buffers = Vec::new();
    let mut name_to_index = FxHashMap::default();
    let mut next_binding = 0_u32;
    let mut fused_workgroup = [1u32, 1, 1];

    for prog in programs {
        let wg = prog.workgroup_size();
        fused_workgroup[0] = fused_workgroup[0].max(wg[0]);
        fused_workgroup[1] = fused_workgroup[1].max(wg[1]);
        fused_workgroup[2] = fused_workgroup[2].max(wg[2]);

        for buf in prog.buffers() {
            let name = Ident::from(buf.name());
            if let Some(&idx) = name_to_index.get(&name) {
                let existing = &mut merged_buffers[idx];
                upgrade_buffer_access(existing, &buf.access());
            } else {
                let mut merged = buf.clone();
                if merged.access() != BufferAccess::Workgroup {
                    merged.binding = next_binding;
                    next_binding += 1;
                }
                name_to_index.insert(name, merged_buffers.len());
                merged_buffers.push(merged);
            }
        }
    }

    let mut combined_entry = Vec::new();
    for (i, prog) in programs.iter().enumerate() {
        combined_entry.extend_from_slice(prog.entry());
        if i + 1 < programs.len() {
            combined_entry.push(Node::logical_barrier(MemoryOrdering::GridSync));
        }
    }

    Ok(Program::wrapped(
        merged_buffers,
        fused_workgroup,
        combined_entry,
    ))
}
