//! Memory planning for resident CSR frontier-queue batches.

use super::scratch::resident_csr_queue_scratch_bytes_per_query;

/// Memory plan for sharding resident CSR queue batches.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentCsrQueueBatchMemoryPlan {
    /// Number of input queries.
    pub query_count: usize,
    /// Largest query count used by any resident dispatch chunk in the plan.
    pub max_queries_per_dispatch: usize,
    /// Number of dispatch batches required.
    pub dispatch_batches: usize,
    /// Peak resident scratch bytes required by one query in the plan.
    pub bytes_per_query: usize,
    /// Peak resident scratch bytes for any planned dispatch batch.
    pub peak_batch_scratch_bytes: usize,
}

/// Errors produced while planning resident CSR queue batch memory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResidentCsrQueueBatchMemoryPlanError {
    /// No queries were requested.
    EmptyBatch,
    /// Queue capacity was zero.
    EmptyQueueCapacity,
    /// Arithmetic overflow occurred while computing byte requirements.
    ScratchBytesOverflow,
    /// Memory budget cannot fit even one query.
    BudgetTooSmall {
        /// Bytes required per query.
        bytes_per_query: usize,
        /// Maximum scratch bytes budgeted.
        max_scratch_bytes: usize,
    },
}

impl std::fmt::Display for ResidentCsrQueueBatchMemoryPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyBatch => {
                write!(
                    f,
                    "Fix: resident CSR queue batch must contain at least one query."
                )
            }
            Self::EmptyQueueCapacity => {
                write!(
                    f,
                    "Fix: resident CSR queue capacity must be greater than zero."
                )
            }
            Self::ScratchBytesOverflow => {
                write!(
                    f,
                    "Fix: resident CSR queue scratch byte requirements overflowed usize arithmetic."
                )
            }
            Self::BudgetTooSmall {
                bytes_per_query,
                max_scratch_bytes,
            } => {
                write!(
                    f,
                    "Fix: resident CSR queue budget ({max_scratch_bytes} bytes) cannot fit one query ({bytes_per_query} bytes)."
                )
            }
        }
    }
}

impl std::error::Error for ResidentCsrQueueBatchMemoryPlanError {}

/// Plan query sharding for resident CSR queue batch execution.
pub fn plan_resident_csr_queue_batch_memory(
    query_count: usize,
    frontier_words: usize,
    queue_capacity: u32,
    max_scratch_bytes: usize,
) -> Result<ResidentCsrQueueBatchMemoryPlan, ResidentCsrQueueBatchMemoryPlanError> {
    if query_count == 0 {
        return Err(ResidentCsrQueueBatchMemoryPlanError::EmptyBatch);
    }
    if queue_capacity == 0 {
        return Err(ResidentCsrQueueBatchMemoryPlanError::EmptyQueueCapacity);
    }

    let bytes_per_query =
        resident_csr_queue_scratch_bytes_per_query(frontier_words, queue_capacity)
            .map_err(|_| ResidentCsrQueueBatchMemoryPlanError::ScratchBytesOverflow)?;

    if bytes_per_query == 0 {
        return Err(ResidentCsrQueueBatchMemoryPlanError::ScratchBytesOverflow);
    }
    if max_scratch_bytes < bytes_per_query {
        return Err(ResidentCsrQueueBatchMemoryPlanError::BudgetTooSmall {
            bytes_per_query,
            max_scratch_bytes,
        });
    }

    let max_queries_per_dispatch = (max_scratch_bytes / bytes_per_query).max(1);
    let dispatch_batches = query_count.div_ceil(max_queries_per_dispatch);
    let peak_batch_queries = query_count.min(max_queries_per_dispatch);
    let peak_batch_scratch_bytes = peak_batch_queries
        .checked_mul(bytes_per_query)
        .ok_or(ResidentCsrQueueBatchMemoryPlanError::ScratchBytesOverflow)?;

    Ok(ResidentCsrQueueBatchMemoryPlan {
        query_count,
        max_queries_per_dispatch,
        dispatch_batches,
        bytes_per_query,
        peak_batch_scratch_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_query_batch_fits_in_exact_budget() {
        let plan = plan_resident_csr_queue_batch_memory(1, 64, 128, 1024 * 1024)
            .expect("Fix: single query should plan");
        assert_eq!(plan.query_count, 1);
        assert_eq!(plan.dispatch_batches, 1);
        assert!(plan.max_queries_per_dispatch >= 1);
        assert_eq!(plan.peak_batch_scratch_bytes, plan.bytes_per_query);
    }

    #[test]
    fn small_budget_forces_multiple_batches() {
        let one_query_plan = plan_resident_csr_queue_batch_memory(1, 64, 128, 1024 * 1024).unwrap();
        let bytes_per_query = one_query_plan.bytes_per_query;

        let budget_for_two = bytes_per_query * 2;
        let plan = plan_resident_csr_queue_batch_memory(5, 64, 128, budget_for_two)
            .expect("Fix: 5 queries in budget for 2 should plan into 3 batches");
        assert_eq!(plan.query_count, 5);
        assert_eq!(plan.max_queries_per_dispatch, 2);
        assert_eq!(plan.dispatch_batches, 3);
        assert_eq!(plan.peak_batch_scratch_bytes, budget_for_two);
    }

    #[test]
    fn reject_zero_queries() {
        let err = plan_resident_csr_queue_batch_memory(0, 64, 128, 1024 * 1024).unwrap_err();
        assert_eq!(err, ResidentCsrQueueBatchMemoryPlanError::EmptyBatch);
    }

    #[test]
    fn reject_zero_capacity() {
        let err = plan_resident_csr_queue_batch_memory(1, 64, 0, 1024 * 1024).unwrap_err();
        assert_eq!(
            err,
            ResidentCsrQueueBatchMemoryPlanError::EmptyQueueCapacity
        );
    }

    #[test]
    fn reject_budget_too_small() {
        let err = plan_resident_csr_queue_batch_memory(1, 64, 128, 10).unwrap_err();
        assert!(matches!(
            err,
            ResidentCsrQueueBatchMemoryPlanError::BudgetTooSmall { .. }
        ));
    }
}
