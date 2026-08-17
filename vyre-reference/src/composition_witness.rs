//! Independent, obviously correct sequential mathematical witnesses for composite operations.
//!
//! Per Section 183.3, reference witnesses use simple sequential mathematical algorithms
//! without Blelloch scheduling, workgroup decomposition, frontier queues, or other GPU optimizations.
//! Composed Programs continue to run through the generic reference interpreter, with independent
//! known-answer cases where interpreter parity alone would compare an implementation with itself.

use vyre_spec::Semiring;

/// Sequential mathematical witness for a generalized semiring matrix multiplication: `C = A ⊗ B`.
///
/// Shape: `A` is `m × k`, `B` is `k × n`, output `C` is `m × n`.
///
/// # Panics
/// Panics if input slice dimensions do not match `m * k` and `k * n`.
#[must_use]
pub fn semiring_gemm_witness(
    a: &[u32],
    b: &[u32],
    m: usize,
    n: usize,
    k: usize,
    semiring: Semiring,
) -> Vec<u32> {
    assert_eq!(
        a.len(),
        m * k,
        "A dimension mismatch in semiring GEMM witness"
    );
    assert_eq!(
        b.len(),
        k * n,
        "B dimension mismatch in semiring GEMM witness"
    );

    let zero = semiring.identity();
    let mut c = vec![zero; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc = zero;
            for p in 0..k {
                let a_val = a[i * k + p];
                let b_val = b[p * n + j];

                let term = match semiring {
                    Semiring::Real => a_val.wrapping_mul(b_val),
                    Semiring::MinPlus | Semiring::MaxPlus => a_val.saturating_add(b_val),
                    Semiring::MaxTimes => a_val.wrapping_mul(b_val),
                    Semiring::BoolOr => a_val & b_val,
                    Semiring::BoolAnd => a_val | b_val,
                    Semiring::Gf2 => a_val & b_val,
                    Semiring::Lineage => a_val | b_val,
                };

                acc = match semiring {
                    Semiring::Real => acc.wrapping_add(term),
                    Semiring::MinPlus => acc.min(term),
                    Semiring::MaxPlus | Semiring::MaxTimes => acc.max(term),
                    Semiring::BoolOr | Semiring::Lineage => acc | term,
                    Semiring::BoolAnd => acc & term,
                    Semiring::Gf2 => acc ^ term,
                };
            }
            c[i * n + j] = acc;
        }
    }

    c
}

/// Sequential mathematical witness for inclusive and exclusive prefix scans.
#[must_use]
pub fn prefix_scan_witness(
    input: &[u32],
    inclusive: bool,
    combine_op: impl Fn(u32, u32) -> u32,
    identity: u32,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(input.len());
    let mut acc = identity;

    for &val in input {
        if inclusive {
            acc = combine_op(acc, val);
            out.push(acc);
        } else {
            out.push(acc);
            acc = combine_op(acc, val);
        }
    }

    out
}

/// Sequential mathematical witness for CSR (Compressed Sparse Row) graph breadth-first traversal.
///
/// Computes shortest distances in unweighted graphs from `source` node to all reachable nodes.
/// Unreachable nodes receive `u32::MAX`.
#[must_use]
pub fn csr_bfs_witness(
    node_count: usize,
    row_offsets: &[u32],
    col_indices: &[u32],
    source: usize,
) -> Vec<u32> {
    assert!(
        row_offsets.len() >= node_count + 1,
        "row_offsets must have at least node_count + 1 entries"
    );

    let mut distances = vec![u32::MAX; node_count];
    if source >= node_count {
        return distances;
    }

    distances[source] = 0;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(source);

    while let Some(u) = queue.pop_front() {
        let dist_u = distances[u];
        let start = row_offsets[u] as usize;
        let end = row_offsets[u + 1] as usize;

        for &v in &col_indices[start..end] {
            let v = v as usize;
            if v < node_count && distances[v] == u32::MAX {
                distances[v] = dist_u + 1;
                queue.push_back(v);
            }
        }
    }

    distances
}
