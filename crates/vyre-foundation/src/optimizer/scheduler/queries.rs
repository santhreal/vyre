//! PassScheduler fusion-query methods + remaining constructor helpers.
//! Audit cleanup A21 (2026-04-30): split from monolithic scheduler.rs.

use rustc_hash::{FxHashMap, FxHashSet};

use super::topo::{
    reserve_hash_map_capacity, reserve_vec_capacity, schedule_pass_metadata_indices,
};
use super::{PassResearchTrace, PassScheduler, PassSchedulingError};
use crate::optimizer::{AdapterCaps, ProgramPassKind};

impl PassScheduler {
    /// Create a new `PassScheduler` from an explicit list of passes.
    ///
    /// # Panics
    ///
    /// Panics when constructor scratch cannot be reserved. The scheduler this
    /// builds is the only thing that runs a pass, so an empty one compiles the
    /// program with no optimization at all and returns it under an `Ok`. That
    /// answer is indistinguishable from a program the pipeline decided not to
    /// change, which is why allocation failure ends the compile instead of
    /// quietly emptying it. Callers that can report the failure themselves use
    /// [`Self::try_with_passes`].
    #[must_use]
    pub fn with_passes(passes: Vec<ProgramPassKind>) -> Self {
        match Self::try_with_passes(passes) {
            Ok(scheduler) => scheduler,
            Err(error) => panic!(
                "PassScheduler could not reserve constructor scratch: {error}. Fix: lower the pass count for this compile, or call try_with_passes and report the failure."
            ),
        }
    }

    /// Create a scheduler that compiles every program for `adapter`.
    ///
    /// This is the entry a backend uses once it has probed a device. Without
    /// it the scheduler falls back to [`AdapterCaps::conservative`], and an
    /// adapter-dependent pass then produces the program a device with no
    /// optional feature would want.
    #[must_use]
    pub fn for_adapter(passes: Vec<ProgramPassKind>, adapter: AdapterCaps) -> Self {
        let mut scheduler = Self::with_passes(passes);
        scheduler.adapter = adapter;
        scheduler
    }

    /// Device facts this scheduler compiles against.
    #[must_use]
    pub fn adapter(&self) -> &AdapterCaps {
        &self.adapter
    }

    /// Create a new `PassScheduler` from an explicit list of passes, surfacing
    /// allocation pressure as a structured scheduling error.
    ///
    /// Invalid dependency metadata still preserves the historical constructor
    /// behavior: the scheduler is created with `requirements_prevalidated=false`
    /// and a stable input-order fallback. Only scratch allocation failure is
    /// returned as an error.
    ///
    /// # Errors
    ///
    /// Returns [`PassSchedulingError::StorageReserveFailed`] when constructor
    /// scratch cannot be reserved.
    pub fn try_with_passes(passes: Vec<ProgramPassKind>) -> Result<Self, PassSchedulingError> {
        let mut metadata = Vec::new();
        reserve_vec_capacity(&mut metadata, passes.len(), "scheduler pass metadata")?;
        metadata.extend(passes.iter().map(ProgramPassKind::metadata));
        let scheduled = schedule_pass_metadata_indices(&metadata);
        let (requirements_prevalidated, execution_order) = match scheduled {
            Ok(order) => (true, order),
            Err(error @ PassSchedulingError::StorageReserveFailed { .. }) => return Err(error),
            Err(_) => {
                let mut fallback = Vec::new();
                reserve_vec_capacity(
                    &mut fallback,
                    passes.len(),
                    "scheduler fallback execution order",
                )?;
                fallback.extend(0..passes.len());
                (false, fallback)
            }
        };
        let mut pass_index = FxHashMap::default();
        reserve_hash_map_capacity(&mut pass_index, passes.len(), "scheduler pass index")?;
        pass_index.extend(
            passes
                .iter()
                .enumerate()
                .map(|(i, pass)| (pass.metadata().name, i)),
        );
        Ok(Self::over(
            passes,
            pass_index,
            execution_order,
            requirements_prevalidated,
        ))
    }

    /// Set the maximum number of iterations the scheduler will allow before giving up.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Returns the configured maximum iteration budget for fixpoint execution.
    #[must_use]
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Returns the recorded research trace for `pass_name`, if one was registered.
    #[must_use]
    pub fn research_trace_for(&self, pass_name: &str) -> Option<PassResearchTrace> {
        self.research_traces.get(pass_name).copied()
    }

    /// Names of every pass that may need to re-run when `pass_name` invalidates
    /// any capability they require. Computed via the substrate adjustment-set
    /// back-door analysis on the pass-dependency graph derived from
    /// `requires`/`invalidates` metadata.
    ///
    /// Replaces the hand-rolled per-call dependents traversal in
    /// `run_once` for callers that need bulk transitive
    /// invalidation queries (e.g. an external rule-update path that wants to
    /// know "if rule X changes, which pass-output capabilities go stale?").
    ///
    /// Returns an empty `Vec` when `pass_name` is not registered.
    #[must_use]
    pub fn transitive_dependents(&self, pass_name: &str) -> Vec<&'static str> {
        let n = self.passes.len();
        if n == 0 {
            return Vec::new();
        }
        let Some(&treatment_idx) = self.pass_index.get(pass_name) else {
            return Vec::new();
        };

        // Build the pass→pass adjacency: edge i→j iff pass j's `requires`
        // includes any capability listed in pass i's `invalidates`. This is
        // the "if i invalidates a cap that j requires, j must rerun after i"
        // relation. Same shape used by the scheduler's hand-rolled
        // `dependents` array but materialized as a dense bitset matrix so
        // the substrate `pass_descendants` can transitively close it.
        let adj = self.invalidation_adjacency();
        let n_u32 = u32::try_from(n).unwrap_or(u32::MAX);
        let descendants =
            crate::pass_math::adjustment_set_pass_dependency::pass_descendants(adj, n_u32);
        let row = &descendants[treatment_idx];
        row.iter()
            .filter_map(|&j| self.passes.get(j as usize).map(|pass| pass.metadata().name))
            .collect()
    }

    /// Reachability check: returns true if pass `from` can transitively
    /// invalidate any capability `to` requires. Computed via the
    /// substrate `semiring_closure::reachability_closure` with the
    /// `BoolOr` semiring over the same invalidation adjacency built by
    /// [`Self::transitive_dependents`].
    ///
    /// O(1) lookup after one sparse closure pass  -  caller
    /// can keep the closure cached across many queries by calling
    /// [`Self::invalidation_closure`] once and indexing the result.
    #[must_use]
    pub fn reaches(&self, from: &str, to: &str) -> bool {
        if self.passes.is_empty() || from == to {
            return false;
        }
        let closure = self.invalidation_closure_ref();
        closure.get(from).is_some_and(|set| set.contains(to))
    }

    /// Materialize the full pass→pass transitive invalidation closure as a
    /// row-major boolean adjacency. `closure[i*n+j] != 0` iff pass `i`
    /// transitively invalidates a capability pass `j` requires.
    ///
    /// Use this once when a caller needs to issue many reachability queries
    ///  -  keeps the closure cached so each query is O(1).
    #[must_use]
    pub fn invalidation_closure(&self) -> Vec<u32> {
        let n = self.passes.len();
        if n == 0 {
            return Vec::new();
        }
        let closure = self.invalidation_closure_ref();
        let mut dense = vec![0u32; n * n];
        for (i, from_pass) in self.passes.iter().enumerate() {
            if let Some(reachable) = closure.get(from_pass.metadata().name) {
                for (j, to_pass) in self.passes.iter().enumerate() {
                    if reachable.contains(to_pass.metadata().name) {
                        dense[i * n + j] = 1;
                    }
                }
            }
        }
        dense
    }

    /// Continuous fusion-priority indicator per pass via homotopy
    /// continuation on the relaxed pass-fusion ILP. Each entry is in
    /// `[0, 1]`; higher = more pressure to fuse this pass with
    /// neighbors. Routes through
    /// `optimizer::megakernel::schedule_oracle::schedule_via_homotopy`.
    ///
    /// `costs[i]` is the per-pass run cost (caller-provided  -
    /// scheduler doesn't track perf telemetry yet, so callers pass
    /// `[1.0; n]` for unweighted or perf-derived weights for
    /// telemetry-driven scheduling).
    #[must_use]
    pub fn fusion_pressure(&self, costs: &[f64], steps: u32, dt: f64) -> Vec<f64> {
        let n = self.passes.len();
        if n == 0 || costs.len() != n {
            return Vec::new();
        }
        let n_u32 = u32::try_from(n).unwrap_or(u32::MAX);
        crate::optimizer::megakernel::schedule_oracle::schedule_via_homotopy(
            costs, n_u32, steps, dt,
        )
    }

    /// Maximum independent set of fusable passes via matroid
    /// intersection on the invalidation adjacency. Routes through
    /// `optimizer::megakernel::matroid_subset::max_fusion_subset`.
    /// Returns a 0/1 vector indexed by pass position; 1 = pass is
    /// in the maximum-fusion subset.
    ///
    /// `seed` is the initial subset (pass empty `[0; n]` for
    /// "find max from scratch").
    #[must_use]
    pub fn fusable_subset(&self, seed: &[u32], max_iters: u32) -> Vec<u32> {
        let n = self.passes.len();
        if n == 0 || seed.len() != n {
            return Vec::new();
        }
        let adj = self.invalidation_adjacency();
        crate::optimizer::megakernel::matroid_subset::max_fusion_subset(seed, adj, n, max_iters)
    }

    /// Test whether two passes produce semantically identical capability
    /// rewrites when applied in either order. Computed via the substrate
    /// `functorial_pass_composition::passes_commute_on` on the per-pass
    /// capability column mappings.
    ///
    /// Returns false if either pass is unknown to this scheduler. Two
    /// commuting passes can be reordered freely; non-commuting passes
    /// must respect the topological order from `schedule_passes`.
    #[must_use]
    pub fn pair_commutes(&self, pass_a: &str, pass_b: &str) -> bool {
        let Some(&a_idx) = self.pass_index.get(pass_a) else {
            return false;
        };
        let Some(&b_idx) = self.pass_index.get(pass_b) else {
            return false;
        };
        if a_idx == b_idx {
            return true;
        }
        let metadata_a = self.passes[a_idx].metadata();
        let metadata_b = self.passes[b_idx].metadata();
        let a_invalidates_b_requirement = metadata_a.invalidates.iter().any(|invalidated| {
            metadata_b
                .requires
                .iter()
                .any(|required| required == invalidated)
        });
        let b_invalidates_a_requirement = metadata_b.invalidates.iter().any(|invalidated| {
            metadata_a
                .requires
                .iter()
                .any(|required| required == invalidated)
        });
        !a_invalidates_b_requirement && !b_invalidates_a_requirement
    }

    fn invalidation_adjacency(&self) -> &[u32] {
        self.invalidation_adjacency_cache.get_or_init(|| {
            let n = self.passes.len();
            let pass_metas: Vec<_> = self.passes.iter().map(ProgramPassKind::metadata).collect();
            let mut adj = vec![0u32; n * n];
            for (i, m_i) in pass_metas.iter().enumerate() {
                for (j, m_j) in pass_metas.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    let invalidates_required_by_j = m_i
                        .invalidates
                        .iter()
                        .any(|inv| m_j.requires.iter().any(|req| req == inv));
                    if invalidates_required_by_j {
                        adj[i * n + j] = 1;
                    }
                }
            }
            adj
        })
    }

    fn invalidation_closure_ref(&self) -> &FxHashMap<&'static str, FxHashSet<&'static str>> {
        self.invalidation_closure_cache.get_or_init(|| {
            let pass_metas: Vec<_> = self.passes.iter().map(ProgramPassKind::metadata).collect();

            // Sparse adjacency: pass name -> set of directly reachable pass names.
            let mut adj: FxHashMap<&'static str, FxHashSet<&'static str>> = FxHashMap::default();
            for (i, m_i) in pass_metas.iter().enumerate() {
                for (j, m_j) in pass_metas.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    if m_i
                        .invalidates
                        .iter()
                        .any(|inv| m_j.requires.iter().any(|req| req == inv))
                    {
                        adj.entry(m_i.name).or_default().insert(m_j.name);
                    }
                }
            }

            let mut closure = FxHashMap::default();
            for pass in &self.passes {
                let name = pass.metadata().name;
                let mut reachable = FxHashSet::default();
                let mut stack: Vec<&'static str> = Vec::new();

                if let Some(neighbors) = adj.get(name) {
                    for &neighbor in neighbors {
                        stack.push(neighbor);
                        reachable.insert(neighbor);
                    }
                }

                while let Some(cur) = stack.pop() {
                    if let Some(neighbors) = adj.get(cur) {
                        for &neighbor in neighbors {
                            if reachable.insert(neighbor) {
                                stack.push(neighbor);
                            }
                        }
                    }
                }

                if !reachable.is_empty() {
                    closure.insert(name, reachable);
                }
            }
            closure
        })
    }
}
