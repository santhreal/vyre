//! WGPU-local names for shared fallible allocation reservation helpers.

pub(crate) use vyre_driver::allocation::{
    reserve_hash_map_to_capacity, reserve_smallvec_to_capacity, reserve_vec_to_capacity,
};

use vyre_driver::BackendError;

pub(crate) fn padded_wgpu_u64(
    size: u64,
    label: &'static str,
    fix: &'static str,
) -> Result<u64, BackendError> {
    let normalized = size.max(4);
    let remainder = normalized % 4;
    if remainder == 0 {
        return Ok(normalized);
    }
    normalized.checked_add(4 - remainder).ok_or_else(|| {
        BackendError::new(format!(
            "{label} overflows u64 while padding to WGPU's 4-byte buffer alignment. Fix: {fix}"
        ))
    })
}
