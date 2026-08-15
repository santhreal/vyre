#[cfg(any(test, feature = "cpu-parity"))]
use super::validate::validate_csr_inputs;
use crate::graph::csr_closure_entry_points::define_csr_closure_entry_points;
#[cfg(any(test, feature = "cpu-parity"))]
use crate::graph::csr_closure_inputs::CsrClosureInputs;

/// CPU reference for one in-place expansion pass.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
) -> (Vec<u32>, u32) {
    let mut out = Vec::new();
    let changed = cpu_ref_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier,
        allow_mask,
        &mut out,
    );
    (out, changed)
}

/// CPU reference writing the expanded frontier into caller-owned storage.
#[cfg(any(test, feature = "cpu-parity"))]
pub(crate) fn cpu_ref_into(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
    out: &mut Vec<u32>,
) -> u32 {
    let layout = validate_csr_inputs(node_count, edge_offsets, edge_targets, edge_kind_mask)
        .unwrap_or_else(|err| {
            panic!("csr_forward_or_changed CPU oracle received malformed CSR. {err}")
        });
    let words = layout.frontier_words;
    out.clear();
    out.extend_from_slice(frontier);
    out.resize(words, 0);
    if edge_offsets.is_empty() {
        return 0;
    }
    let mut changed = 0u32;
    for src in 0..node_count as usize {
        let src_word = src / 32;
        let src_bit = 1u32 << (src % 32);
        if out[src_word] & src_bit == 0 {
            continue;
        }
        let start = edge_offsets[src] as usize;
        let end = edge_offsets[src + 1] as usize;
        for edge in start..end.min(edge_targets.len()).min(edge_kind_mask.len()) {
            if edge_kind_mask[edge] & allow_mask == 0 {
                continue;
            }
            let dst = edge_targets[edge] as usize;
            if dst >= node_count as usize {
                continue;
            }
            let word = dst / 32;
            let bit = 1u32 << (dst % 32);
            let old = out[word];
            out[word] |= bit;
            if out[word] != old {
                changed = 1;
            }
        }
    }
    changed
}

define_csr_closure_entry_points! {
    allocating: cpu_ref_closure {
        /// Iterate the internal `cpu_ref_into` step until the change flag reaches zero or
        /// `max_iters` is exhausted.
    },
    borrowing: cpu_ref_closure_into {
        /// Iterate the internal `cpu_ref_into` step using caller-owned frontier buffers.
    },
    hooked: cpu_ref_closure_into_with_step_hook,
    step_hook: |_| {},
}

/// Iterate the internal `cpu_ref_into` step with a callback after each attempted expansion.
///
/// The hook lets consumers attach observability without owning the
/// fixed-point algorithm.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref_closure_into_with_step_hook<F>(
    inputs: CsrClosureInputs<'_>,
    seed: &[u32],
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
    mut on_step: F,
) where
    F: FnMut(u32),
{
    let graph = inputs.graph;
    current.clear();
    current.extend_from_slice(seed);
    for iteration in 0..inputs.max_iters {
        on_step(iteration);
        let changed = cpu_ref_into(
            graph.node_count,
            graph.edge_offsets,
            graph.edge_targets,
            graph.edge_kind_mask,
            current,
            inputs.allow_mask,
            next,
        );
        std::mem::swap(current, next);
        if changed == 0 {
            return;
        }
    }
}
