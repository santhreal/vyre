//! Shape checks, normalization, and CSR conversion for the dense boolean
//! adjacency matrices every formulation in this module consumes.

pub(super) fn checked_dense_node_count(n: u32) -> Result<usize, String> {
    if n == 0 {
        return Err(
            "Fix: static-analysis fixpoint comparison requires at least one node.".to_string(),
        );
    }
    usize::try_from(n).map_err(|_| format!("Fix: node count {n} does not fit host indexing."))
}

pub(super) fn checked_dense_cells(n_us: usize) -> Result<usize, String> {
    n_us.checked_mul(n_us).ok_or_else(|| {
        format!("Fix: dense adjacency dimensions overflow host indexing for n={n_us}.")
    })
}

pub(super) fn normalize_bool_matrix(adj: &[u32]) -> Vec<u32> {
    adj.iter().map(|value| u32::from(*value != 0)).collect()
}

pub(super) fn dense_bool_to_csr(adj: &[u32], n_us: usize) -> Vec<Vec<usize>> {
    let mut csr = Vec::with_capacity(n_us);
    for row in 0..n_us {
        let mut targets = Vec::new();
        for col in 0..n_us {
            if adj[row * n_us + col] != 0 {
                targets.push(col);
            }
        }
        csr.push(targets);
    }
    csr
}
