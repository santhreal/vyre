//! Shared OP_MATRIX release-backend row classification.

/// Operations whose backend support is a runtime contract, not a kernel.
pub const RUNTIME_DIALECT_CONTRACT_OPS: &[&str] = &[
    "core.indirect_dispatch",
    "io.dma_from_nvme",
    "io.write_back_to_nvme",
    "mem.unmap",
    "mem.zerocopy_map",
];

/// Rows that satisfy the runtime dialect contract for their backend.
pub fn count_runtime_dialect_contract_rows(rows: &[String]) -> usize {
    rows.iter()
        .filter(|row| {
            let Some((op, backend, status)) = parse_release_backend_row(row) else {
                return false;
            };
            RUNTIME_DIALECT_CONTRACT_OPS.contains(&op)
                && ((backend == "reference" && status == "not_applicable")
                    || (matches!(backend, "cuda" | "wgpu") && status == "experimental"))
        })
        .count()
}

/// Rows claiming `supported` for an operation that is a real kernel.
pub fn count_non_runtime_supported_release_backend_rows(rows: &[String]) -> usize {
    rows.iter()
        .filter(|row| {
            let Some((op, _backend, status)) = parse_release_backend_row(row) else {
                return false;
            };
            !RUNTIME_DIALECT_CONTRACT_OPS.contains(&op) && status == "supported"
        })
        .count()
}

fn parse_release_backend_row(row: &str) -> Option<(&str, &str, &str)> {
    let (prefix, status) = row.rsplit_once(':')?;
    let (op, backend) = prefix.rsplit_once(':')?;
    Some((op, backend, status))
}
