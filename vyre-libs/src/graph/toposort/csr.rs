//! The CSR oracle: validate the adjacency, sort it, and validate an order.

use super::error::{toposort_csr_allocation, ToposortCsrError};

/// Validated dispatch layout for primitive-native CSR topological sorting.
///
/// The primitive owns these derived counts so dispatch wrappers do not fork CSR
/// offset or node scratch sizing rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ToposortCsrLayout {
    /// Number of nodes accepted by the primitive.
    pub node_count: u32,
    /// Number of words required by node-indexed scratch and output buffers.
    pub node_words: usize,
    /// Number of words required by the CSR offsets buffer.
    pub offset_words: usize,
    /// Number of words required by the CSR targets buffer.
    pub target_words: usize,
}

/// CPU reference over the primitive-native CSR adjacency shape.
///
/// `offsets` has `node_count + 1` entries and `targets` stores outgoing
/// edges from each prerequisite node to its dependent nodes. The returned
/// order is valid iff every prerequisite appears before every dependent.
///
/// # Errors
///
/// Returns [`ToposortCsrError::BadCsr`] when the CSR shape is malformed and
/// [`ToposortCsrError::BadOrder`] only if derived state violates the
/// topological-order contract after input validation.
pub fn toposort_csr(
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
) -> Result<Vec<u32>, ToposortCsrError> {
    let mut order = Vec::new();
    toposort_csr_into(node_count, offsets, targets, &mut order)?;
    Ok(order)
}

/// Caller-owned workspace for repeated CSR topological-sort CPU oracle runs.
///
/// The CPU oracle is used heavily by conformance and backend parity paths. Keeping
/// indegree and queue storage outside the call lets proof runners amortize heap
/// growth across thousands of generated graphs without changing the public
/// allocating convenience API.
#[derive(Debug, Default, Clone)]
pub struct ToposortCsrScratch {
    /// Per-node incoming-edge counts rebuilt for each run.
    pub indeg: Vec<u32>,
    /// Zero-indegree work queue consumed by Kahn traversal.
    pub queue: Vec<u32>,
}

impl ToposortCsrScratch {
    /// Create an empty reusable topological-sort workspace.
    pub fn new() -> Self {
        Self::default()
    }
}

/// CPU reference over primitive-native CSR adjacency, reusing caller storage.
///
/// # Errors
///
/// Returns [`ToposortCsrError::BadCsr`] when CSR validation fails and
/// [`ToposortCsrError::BadOrder`] when the derived order violates the
/// primitive contract.
pub fn toposort_csr_into(
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
    order: &mut Vec<u32>,
) -> Result<(), ToposortCsrError> {
    let mut scratch = ToposortCsrScratch::default();
    toposort_csr_into_with_scratch(node_count, offsets, targets, order, &mut scratch)
}

/// CPU reference over primitive-native CSR adjacency with caller-owned output
/// and scratch storage.
///
/// # Errors
///
/// Returns [`ToposortCsrError::BadCsr`] when CSR validation fails and
/// [`ToposortCsrError::BadOrder`] when the derived order violates the
/// primitive contract. Validation happens before any caller-owned storage is
/// cleared, so rejected inputs do not clobber reusable buffers.
pub fn toposort_csr_into_with_scratch(
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
    order: &mut Vec<u32>,
    scratch: &mut ToposortCsrScratch,
) -> Result<(), ToposortCsrError> {
    let layout = validate_toposort_csr_inputs(node_count, offsets, targets)?;
    order.clear();
    scratch.indeg.clear();
    scratch.queue.clear();
    if node_count == 0 {
        return Ok(());
    }

    let node_words = layout.node_words;
    crate::plumbing::host::scratch::reserve_items_with(
        &mut scratch.indeg,
        node_words,
        "toposort CSR CPU oracle",
        "toposort_csr indegree scratch",
        toposort_csr_allocation,
    )?;
    scratch.indeg.resize(node_words, 0);
    for (idx, &target) in targets.iter().enumerate() {
        scratch.indeg[target as usize] =
            scratch.indeg[target as usize]
                .checked_add(1)
                .ok_or_else(|| ToposortCsrError::BadCsr {
                    message: format!(
                    "Fix: toposort_csr target node {target} indegree overflowed at targets[{idx}]."
                ),
                })?;
    }

    crate::plumbing::host::scratch::reserve_items_with(
        &mut scratch.queue,
        node_words,
        "toposort CSR CPU oracle",
        "toposort_csr zero-indegree queue",
        toposort_csr_allocation,
    )?;
    for node in 0..node_count {
        if scratch.indeg[node as usize] == 0 {
            scratch.queue.push(node);
        }
    }
    crate::plumbing::host::scratch::reserve_items_with(
        order,
        node_words,
        "toposort CSR CPU oracle",
        "toposort_csr output order",
        toposort_csr_allocation,
    )?;
    while let Some(node) = scratch.queue.pop() {
        order.push(node);
        let start = offsets[node as usize] as usize;
        let end = offsets[node as usize + 1] as usize;
        for (edge_offset, &dependent) in targets[start..end].iter().enumerate() {
            let slot = &mut scratch.indeg[dependent as usize];
            *slot = slot
                .checked_sub(1)
                .ok_or_else(|| ToposortCsrError::BadOrder {
                    message: format!(
                    "Fix: toposort_csr indegree underflow for edge {} from {node} to {dependent}.",
                    start + edge_offset
                ),
                })?;
            if *slot == 0 {
                scratch.queue.push(dependent);
            }
        }
    }

    validate_toposort_csr_order_with_layout(&layout, offsets, targets, order)
}

/// Validate primitive-native CSR input shape.
///
/// # Errors
///
/// Returns [`ToposortCsrError::BadCsr`] when offsets are the wrong length, not
/// monotonic, inconsistent with `targets`, or when a target is out of range.
pub fn validate_toposort_csr_inputs(
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
) -> Result<ToposortCsrLayout, ToposortCsrError> {
    if node_count == 0 {
        if offsets != [0] || !targets.is_empty() {
            return Err(ToposortCsrError::BadCsr {
                message:
                    "Fix: toposort_csr zero-node graph requires offsets == [0] and empty targets."
                        .to_string(),
            });
        }
        return Ok(ToposortCsrLayout {
            node_count,
            node_words: 0,
            offset_words: 1,
            target_words: 0,
        });
    }
    let expected_offsets =
        (node_count as usize)
            .checked_add(1)
            .ok_or_else(|| ToposortCsrError::BadCsr {
                message: format!(
                    "Fix: toposort_csr node_count + 1 overflows usize for node_count={node_count}."
                ),
            })?;
    if offsets.len() != expected_offsets {
        return Err(ToposortCsrError::BadCsr {
            message: format!(
                "Fix: toposort_csr requires offsets.len() == node_count + 1, got len={}, node_count={node_count}.",
                offsets.len()
            ),
        });
    }
    if offsets[0] != 0 {
        return Err(ToposortCsrError::BadCsr {
            message: format!(
                "Fix: toposort_csr requires offsets[0] == 0, got {}.",
                offsets[0]
            ),
        });
    }
    for (idx, pair) in offsets.windows(2).enumerate() {
        if pair[0] > pair[1] {
            return Err(ToposortCsrError::BadCsr {
                message: format!(
                    "Fix: toposort_csr offsets must be monotonic, but offsets[{idx}]={} > offsets[{}]={}.",
                    pair[0],
                    idx + 1,
                    pair[1]
                ),
            });
        }
    }
    if offsets[node_count as usize] as usize != targets.len() {
        return Err(ToposortCsrError::BadCsr {
            message: format!(
                "Fix: toposort_csr offsets[node_count] must equal targets.len(), got {} vs {}.",
                offsets[node_count as usize],
                targets.len()
            ),
        });
    }
    for (idx, &target) in targets.iter().enumerate() {
        if target >= node_count {
            return Err(ToposortCsrError::BadCsr {
                message: format!(
                    "Fix: toposort_csr targets[{idx}]={target} is outside node_count={node_count}."
                ),
            });
        }
    }
    Ok(ToposortCsrLayout {
        node_count,
        node_words: node_count as usize,
        offset_words: expected_offsets,
        target_words: targets.len(),
    })
}

/// Validate that `order` is a full topological permutation for the
/// primitive-native CSR adjacency shape.
///
/// # Errors
///
/// Returns [`ToposortCsrError::BadCsr`] for malformed CSR input and
/// [`ToposortCsrError::BadOrder`] for malformed, partial, duplicate, or
/// dependency-violating orders.
pub fn validate_toposort_csr_order(
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
    order: &[u32],
) -> Result<(), ToposortCsrError> {
    let layout = validate_toposort_csr_inputs(node_count, offsets, targets)?;
    validate_toposort_csr_order_with_layout(&layout, offsets, targets, order)
}

fn validate_toposort_csr_order_with_layout(
    layout: &ToposortCsrLayout,
    offsets: &[u32],
    targets: &[u32],
    order: &[u32],
) -> Result<(), ToposortCsrError> {
    let node_count = layout.node_count;
    if order.len() != node_count as usize {
        return Err(ToposortCsrError::BadOrder {
            message: format!(
                "Fix: toposort_csr expected {} order entries, got {}.",
                node_count,
                order.len()
            ),
        });
    }
    let mut pos: Vec<usize> = Vec::new();
    crate::plumbing::host::scratch::reserve_items_with(
        &mut pos,
        layout.node_words,
        "toposort CSR CPU oracle",
        "toposort_csr order positions",
        toposort_csr_allocation,
    )?;
    pos.resize(layout.node_words, usize::MAX);
    for (idx, &node) in order.iter().enumerate() {
        if node >= node_count {
            return Err(ToposortCsrError::BadOrder {
                message: format!(
                    "Fix: toposort_csr order[{idx}]={node} is outside node_count={node_count}."
                ),
            });
        }
        let slot = &mut pos[node as usize];
        if *slot != usize::MAX {
            return Err(ToposortCsrError::BadOrder {
                message: format!(
                    "Fix: toposort_csr order contains duplicate node {node}; graph may be cyclic or backend output is malformed."
                ),
            });
        }
        *slot = idx;
    }
    if let Some((missing, _)) = pos
        .iter()
        .enumerate()
        .find(|(_, position)| **position == usize::MAX)
    {
        return Err(ToposortCsrError::BadOrder {
            message: format!(
                "Fix: toposort_csr order omitted node {missing}; graph may be cyclic."
            ),
        });
    }

    for prereq in 0..node_count {
        let start = offsets[prereq as usize] as usize;
        let end = offsets[prereq as usize + 1] as usize;
        for &dependent in &targets[start..end] {
            if pos[prereq as usize] >= pos[dependent as usize] {
                return Err(ToposortCsrError::BadOrder {
                    message: format!(
                        "Fix: toposort_csr order violates dependency edge {prereq}->{dependent}; prerequisite position {} must be before dependent position {}.",
                        pos[prereq as usize],
                        pos[dependent as usize]
                    ),
                });
            }
        }
    }
    Ok(())
}
