//! Backend-neutral pipeline-cache invalidation helpers.
//!
//! Backends provide their cache keys and lineage cells; this module owns
//! the shared causal-impact/provenance walk so the backend crates do not
//! depend on the composition modules that implement it directly.

#[cfg(feature = "libs-compositions")]
use vyre_foundation::program_dispatch::{DispatchError as ProgramDispatchError, ProgramDispatcher};
#[cfg(feature = "libs-compositions")]
use vyre_libs::encoding::scallop_provenance::provenance_closure_via_into;
#[cfg(feature = "libs-compositions")]
use vyre_libs::reasoning::do_calculus_change_impact::{
    predict_impact_via_into, project_impacted_lineage_entries_via_into, DoCalculusImpactScratch,
    ImpactedLineageProjectionScratch,
};

/// Error raised by GPU-resident cache invalidation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheInvalidationError {
    message: String,
}

impl CacheInvalidationError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for CacheInvalidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CacheInvalidationError {}

#[cfg(feature = "libs-compositions")]
impl From<ProgramDispatchError> for CacheInvalidationError {
    fn from(error: ProgramDispatchError) -> Self {
        Self::new(error.to_string())
    }
}

/// Reusable scratch for shared pipeline-cache invalidation.
#[derive(Debug, Default)]
pub struct CacheInvalidationScratch {
    #[cfg(feature = "libs-compositions")]
    impact: DoCalculusImpactScratch,
    #[cfg(feature = "libs-compositions")]
    closure: Vec<u32>,
    #[cfg(feature = "libs-compositions")]
    projection: ImpactedLineageProjectionScratch,
}

/// Compute a 0/1 impact mask for cache entries.
///
/// Production builds run the composed implementation. Builds that
/// explicitly disable `libs-compositions` fail loudly instead of running
/// a hidden reference cache-invalidation path.
pub fn impacted_entries_into(
    #[cfg(feature = "libs-compositions")] dispatcher: &dyn ProgramDispatcher,
    intervention_mask: &[u32],
    rule_adj: &[u32],
    state: &[u32],
    join_rules: &[u32],
    n: u32,
    max_iterations: u32,
    lineage_cells: &[u32],
    out: &mut Vec<u32>,
    _scratch: &mut CacheInvalidationScratch,
) -> Result<(), CacheInvalidationError> {
    #[cfg(not(feature = "libs-compositions"))]
    {
        let _ = (
            intervention_mask,
            rule_adj,
            state,
            join_rules,
            n,
            max_iterations,
            lineage_cells,
            out,
            _scratch,
        );
        Err(CacheInvalidationError::new(
            "vyre-driver cache invalidation requires the `libs-compositions` feature. Fix: enable the feature; production builds must not run the reference cache-invalidation oracle.",
        ))
    }

    #[cfg(feature = "libs-compositions")]
    {
        if lineage_cells.is_empty() {
            out.clear();
            return Ok(());
        }

        let n_us = n as usize;
        let Some(matrix_len) = n_us.checked_mul(n_us) else {
            return Err(CacheInvalidationError::new(format!(
                "Fix: cache invalidation n*n overflows usize for n={n}."
            )));
        };
        if intervention_mask.len() != n_us {
            return Err(CacheInvalidationError::new(format!(
                "Fix: cache invalidation requires intervention_mask.len() == n ({n_us}), got {}.",
                intervention_mask.len()
            )));
        }
        if rule_adj.len() != matrix_len {
            return Err(CacheInvalidationError::new(format!(
                "Fix: cache invalidation requires rule_adj.len() == n*n ({matrix_len}), got {}.",
                rule_adj.len()
            )));
        }
        if state.len() != matrix_len {
            return Err(CacheInvalidationError::new(format!(
                "Fix: cache invalidation requires state.len() == n*n ({matrix_len}), got {}.",
                state.len()
            )));
        }
        if join_rules.len() != matrix_len {
            return Err(CacheInvalidationError::new(format!(
                "Fix: cache invalidation requires join_rules.len() == n*n ({matrix_len}), got {}.",
                join_rules.len()
            )));
        }

        reserve_impact_mask(out, lineage_cells.len())?;

        predict_impact_via_into(
            dispatcher,
            rule_adj,
            intervention_mask,
            n,
            &mut _scratch.impact,
        )
        .map_err(|err| {
            out.clear();
            CacheInvalidationError::from(err)
        })?;
        provenance_closure_via_into(
            dispatcher,
            state,
            join_rules,
            n,
            max_iterations,
            &mut _scratch.closure,
        )
        .map_err(|err| {
            out.clear();
            CacheInvalidationError::from(err)
        })?;

        let impacted_rules = _scratch.impact.impact_mask();
        let closure = &_scratch.closure;
        if impacted_rules.len() != n_us || closure.len() != matrix_len {
            out.clear();
            return Err(CacheInvalidationError::new(format!(
                "Fix: cache invalidation GPU output dimensions mismatched: impact_mask={}, closure={}, required n={n_us}, matrix={matrix_len}.",
                impacted_rules.len(),
                closure.len()
            )));
        }

        project_impacted_lineage_entries_via_into(
            dispatcher,
            impacted_rules,
            closure,
            n,
            lineage_cells,
            &mut _scratch.projection,
            out,
        )
        .map_err(|err| {
            out.clear();
            CacheInvalidationError::from(err)
        })?;

        Ok(())
    }
}

/// Compute a 0/1 impact mask using temporary scratch.
#[must_use]
pub fn impacted_entries(
    #[cfg(feature = "libs-compositions")] dispatcher: &dyn ProgramDispatcher,
    intervention_mask: &[u32],
    rule_adj: &[u32],
    state: &[u32],
    join_rules: &[u32],
    n: u32,
    max_iterations: u32,
    lineage_cells: &[u32],
) -> Result<Vec<u32>, CacheInvalidationError> {
    let mut out = reserved_impact_mask(lineage_cells.len())?;
    let mut scratch = CacheInvalidationScratch::default();
    impacted_entries_into(
        #[cfg(feature = "libs-compositions")]
        dispatcher,
        intervention_mask,
        rule_adj,
        state,
        join_rules,
        n,
        max_iterations,
        lineage_cells,
        &mut out,
        &mut scratch,
    )?;
    Ok(out)
}

fn reserve_impact_mask(out: &mut Vec<u32>, len: usize) -> Result<(), CacheInvalidationError> {
    crate::allocation::try_reserve_vec_to_capacity(out, len).map_err(|error| {
        CacheInvalidationError::new(format!(
            "pipeline cache invalidation could not reserve {len} impact-mask slot(s): {error}. Fix: split lineage cells across smaller cache-invalidation shards."
        ))
    })
}

fn reserved_impact_mask(len: usize) -> Result<Vec<u32>, CacheInvalidationError> {
    let mut out = Vec::new();
    reserve_impact_mask(&mut out, len)?;
    Ok(out)
}

#[cfg(all(test, feature = "libs-compositions"))]
mod tests {
    use super::*;
    use vyre_driver_reference::ReferenceEvalDispatcher;
    #[test]
    fn impact_mask_marks_lineage_intersection() {
        let dispatcher = ReferenceEvalDispatcher;
        let n = 3;
        let mut rule_adj = vec![0u32; 9];
        rule_adj[0 * 3 + 1] = 1;
        let intervention_mask = vec![1, 0, 0];

        let mut state = vec![0u32; 9];
        state[1 * 3] = 1;
        let join_rules = vec![0u32; 9];
        let mask = impacted_entries(
            &dispatcher,
            &intervention_mask,
            &rule_adj,
            &state,
            &join_rules,
            n,
            16,
            &[1, 2],
        )
        .expect("reference dispatcher must execute GPU cache invalidation composition");
        assert_eq!(mask, vec![1, 0]);
    }

    #[test]
    fn malformed_dimensions_do_not_panic() {
        let dispatcher = ReferenceEvalDispatcher;
        let err = impacted_entries(&dispatcher, &[1], &[], &[], &[], 32, 16, &[0, 1])
            .expect_err("malformed dimensions must fail loudly");
        assert!(
            err.to_string().contains("Fix:"),
            "cache invalidation dimension errors must be actionable"
        );
    }
}
