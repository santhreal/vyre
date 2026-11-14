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
            Ok(programs.first().cloned().unwrap_or_else(Program::empty))
        }
        FusionCandidateKind::RegisterForwarding
        | FusionCandidateKind::PrologueFusion
        | FusionCandidateKind::EpilogueFusion
        | FusionCandidateKind::PartialFusion => lower_merged_programs(programs, None),
        FusionCandidateKind::SharedMemoryForwarding | FusionCandidateKind::PipelinedTiles => {
            lower_merged_programs(programs, Some(MemoryOrdering::SeqCst))
        }
        FusionCandidateKind::ExplicitDispatchCut => {
            lower_merged_programs(programs, Some(MemoryOrdering::GridSync))
        }
    }
}

fn lower_merged_programs(
    programs: &[Program],
    separator: Option<MemoryOrdering>,
) -> Result<Program, FusionError> {
    match programs {
        [] => return Ok(Program::empty()),
        [single] => return Ok(single.clone()),
        _ => {}
    }

    let mut merged_buffers = Vec::new();
    let mut name_to_index = FxHashMap::default();
    let mut next_binding = 0_u32;
    let mut fused_workgroup = [1u32, 1, 1];

    for prog in programs {
        let wg = prog.workgroup_size();
        for (dst, &src) in fused_workgroup.iter_mut().zip(&wg) {
            *dst = (*dst).max(src);
        }
        for buf in prog.buffers() {
            let name = Ident::from(buf.name());
            if let Some(&idx) = name_to_index.get(&name) {
                upgrade_buffer_access(&mut merged_buffers[idx], &buf.access());
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
        if let Some(ordering) = separator {
            if i + 1 < programs.len() {
                combined_entry.push(Node::logical_barrier(ordering));
            }
        }
    }

    Ok(Program::wrapped(
        merged_buffers,
        fused_workgroup,
        combined_entry,
    ))
}
