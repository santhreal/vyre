//! The mapping between a signature bucket's pair ordinal and the two row ids
//! that ordinal names, and the counts that mapping is derived from. Ordinals
//! enumerate the upper triangle of a bucket in row-major order, so a CUDA
//! thread can recover its row pair from its ordinal alone.

use super::{CudaEGraphKernelPlanError, CudaEGraphSignatureBucketPlan};

/// Decode a signature-bucket pair ordinal to the concrete row ids kernels must
/// compare.
///
/// Pair ordinals enumerate the upper triangle of each bucket in row-major
/// order: `(0, 1), (0, 2), ..., (1, 2), ...`. CUDA kernels can use this same
/// arithmetic to map a thread's pair ordinal to two row ids without materializing
/// all candidate pairs.
///
/// # Errors
///
/// Returns [`CudaEGraphKernelPlanError::SignaturePairOrdinalOutOfBounds`] when
/// `bucket_index` or `pair_ordinal` does not identify a planned candidate pair.
pub fn cuda_egraph_signature_pair_rows(
    plan: &CudaEGraphSignatureBucketPlan,
    bucket_index: u32,
    pair_ordinal: u64,
) -> Result<(u32, u32), CudaEGraphKernelPlanError> {
    let Some(bucket) = plan.buckets.get(bucket_index as usize) else {
        return Err(CudaEGraphKernelPlanError::SignaturePairOrdinalOutOfBounds {
            bucket_index,
            pair_ordinal,
            candidate_pair_count: 0,
        });
    };
    if pair_ordinal >= bucket.candidate_pair_count {
        return Err(CudaEGraphKernelPlanError::SignaturePairOrdinalOutOfBounds {
            bucket_index,
            pair_ordinal,
            candidate_pair_count: bucket.candidate_pair_count,
        });
    }

    let row_count = u64::from(bucket.row_count);
    let mut lo = 0_u64;
    let mut hi = row_count - 1;
    while lo < hi {
        let mid = lo + ((hi - lo) / 2);
        let next_start = signature_pairs_before_row(mid + 1, row_count)?;
        if next_start <= pair_ordinal {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let local_left = lo;
    let row_pair_base = signature_pairs_before_row(local_left, row_count)?;
    let local_right = local_left
        .checked_add(1)
        .and_then(|value| value.checked_add(pair_ordinal - row_pair_base))
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "signature pair local right row",
        })?;
    let base = bucket.first_bucket_row as usize;
    let bucket_end = base.checked_add(bucket.row_count as usize).ok_or(
        CudaEGraphKernelPlanError::CountOverflow {
            field: "signature bucket row range end",
        },
    )?;
    if bucket_end > plan.bucket_rows.len() {
        return Err(CudaEGraphKernelPlanError::SignatureBucketRowsOutOfBounds {
            bucket_index,
            first_bucket_row: base,
            row_count: bucket.row_count as usize,
            bucket_rows_len: plan.bucket_rows.len(),
        });
    }
    let left = plan.bucket_rows[base + local_left as usize];
    let right = plan.bucket_rows[base + local_right as usize];
    Ok((left, right))
}

pub(super) fn unordered_pair_count(item_count: u64) -> Result<u64, CudaEGraphKernelPlanError> {
    item_count
        .checked_mul(item_count.saturating_sub(1))
        .and_then(|count| count.checked_div(2))
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "unordered pair count",
        })
}

pub(super) fn signature_pairs_before_row(
    local_row: u64,
    row_count: u64,
) -> Result<u64, CudaEGraphKernelPlanError> {
    local_row
        .checked_mul(
            row_count
                .checked_mul(2)
                .and_then(|value| value.checked_sub(local_row))
                .and_then(|value| value.checked_sub(1))
                .ok_or(CudaEGraphKernelPlanError::CountOverflow {
                    field: "signature pair row width",
                })?,
        )
        .and_then(|value| value.checked_div(2))
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "signature pairs before row",
        })
}
