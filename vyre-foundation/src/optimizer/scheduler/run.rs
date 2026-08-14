//! PassScheduler run methods + invalidation propagation.
//! Audit cleanup A21 (2026-04-30): split from monolithic scheduler.rs.

#![allow(unused_imports)]

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BTreeSet;
use std::sync::OnceLock;

use super::{
    estimate_ir_allocations, IrAllocationEstimate, OptimizerRunReport, PassRunDecision,
    PassRunMetric, PassScheduler,
};
use crate::ir::{BufferDecl, Expr, Node};
use crate::ir_inner::model::program::Program;
use crate::optimizer::{
    fact_cache::FactCache, registered_passes, requirements_satisfied, OptimizerError, PassMetadata,
    ProgramPassKind, ProgramPassRegistration,
};
use crate::perf::PerfScope;

#[derive(Debug, Default)]
pub(crate) struct SchedulerFactState {
    facts: Option<FactCache>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SchedulerFactEvent {
    reused: bool,
    recomputed: bool,
}

impl SchedulerFactState {
    fn prepare(&mut self, program: &Program) -> SchedulerFactEvent {
        if self
            .facts
            .as_ref()
            .is_some_and(|facts| facts.is_fresh_for(program))
        {
            return SchedulerFactEvent {
                reused: true,
                recomputed: false,
            };
        }
        self.facts = Some(FactCache::derive(program));
        SchedulerFactEvent {
            reused: false,
            recomputed: true,
        }
    }

    fn invalidate(&mut self) -> bool {
        let Some(facts) = self.facts.as_mut() else {
            return false;
        };
        facts.invalidate();
        true
    }
}

fn introduces_forbidden_effects(
    before: crate::lower::effects::ProgramEffects,
    after: crate::lower::effects::ProgramEffects,
    allowed_additions: crate::lower::effects::ProgramEffects,
) -> bool {
    !after
        .introduced_since(before)
        .is_subset_of(allowed_additions)
}

fn linear_type_violations(program: &Program) -> BTreeSet<String> {
    crate::validate::linear_type::check_linear_types(program)
        .into_iter()
        .map(|error| error.message().into_owned())
        .collect()
}

fn introduces_linear_type_violations(before: &BTreeSet<String>, after: &BTreeSet<String>) -> bool {
    after.iter().any(|violation| !before.contains(violation))
}

fn shape_predicate_violations(program: &Program) -> BTreeSet<String> {
    crate::validate::shape_predicate::check_shape_predicates(program)
        .into_iter()
        .map(|error| error.message().into_owned())
        .collect()
}

fn introduces_shape_predicate_violations(
    before: &BTreeSet<String>,
    after: &BTreeSet<String>,
) -> bool {
    after.iter().any(|violation| !before.contains(violation))
}

/// Post-condition certificates for one program.
///
/// Every enabled gate is a pure function of the program, so a pass that
/// declines to rewrite leaves all of them valid, and the certificates a landed
/// rewrite is judged against are the certificates the previous accepted rewrite
/// already produced. Each field is `None` when its gate is disabled.
#[derive(Debug, Default)]
struct GateFacts {
    cost: Option<crate::optimizer::cost::CostCertificate>,
    effects: Option<crate::lower::effects::ProgramEffects>,
    linear_violations: Option<BTreeSet<String>>,
    shape_violations: Option<BTreeSet<String>>,
}

impl GateFacts {
    /// Effect-row bits, or zero when effect enforcement is disabled.
    fn effect_bits(&self) -> u32 {
        self.effects
            .map_or(0, crate::lower::effects::ProgramEffects::bits)
    }

    /// Linear-type violation count, or zero when that gate is disabled.
    fn linear_violation_count(&self) -> usize {
        self.linear_violations.as_ref().map_or(0, BTreeSet::len)
    }

    /// Shape-predicate violation count, or zero when that gate is disabled.
    fn shape_violation_count(&self) -> usize {
        self.shape_violations.as_ref().map_or(0, BTreeSet::len)
    }
}

/// Which post-condition gate rejected a rewrite.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GateRejection {
    Effects,
    LinearTypes,
    ShapePredicates,
    Cost,
}

impl GateRejection {
    fn decision(self) -> PassRunDecision {
        match self {
            Self::Effects => PassRunDecision::EffectReverted,
            Self::LinearTypes => PassRunDecision::LinearTypeReverted,
            Self::ShapePredicates => PassRunDecision::ShapePredicateReverted,
            Self::Cost => PassRunDecision::CostReverted,
        }
    }
}

/// Gate certificates carried from one pass to the next, keyed by the
/// fingerprint of the program they describe.
///
/// Deriving them costs a whole-program effect walk plus two validation walks
/// that allocate their diagnostics, and the scheduler used to pay that for
/// every running pass in every fixpoint iteration even though most passes leave
/// the program alone.
///
/// The key is `Program::fingerprint`, not the pass's own `changed` flag: a pass
/// may return a rewritten program while reporting no change, and handing that
/// program the previous program's certificates would judge the next rewrite
/// against the wrong baseline. Certificates are stored under the fingerprint of
/// the program they were derived from, and a lookup that does not match
/// re-derives, so a stale entry can be held but never served.
#[derive(Debug, Default)]
struct GateFactState {
    entry: Option<([u8; 32], GateFacts)>,
}

impl GateFactState {
    /// Carry `facts` forward as the certificates of `program`.
    fn store(&mut self, program: &Program, facts: GateFacts) {
        self.entry = Some((program.fingerprint(), facts));
    }
}

impl PassScheduler {
    /// `tag → pass names that depend on it` (their own name OR a `requires`
    /// entry equals the tag). Computed once per scheduler. Replaces the
    /// linear pass-list scan that the previous implementation ran on every
    /// invalidation event.
    fn dirty_trigger_index(&self) -> &FxHashMap<&'static str, Vec<usize>> {
        self.dirty_trigger_index_cache.get_or_init(|| {
            let mut index: FxHashMap<&'static str, Vec<usize>> = FxHashMap::default();
            index.reserve(self.passes.len() * 2);
            for (pass_index, pass) in self.passes.iter().enumerate() {
                let metadata = pass.metadata();
                index.entry(metadata.name).or_default().push(pass_index);
                for &req in metadata.requires {
                    index.entry(req).or_default().push(pass_index);
                }
            }
            index
        })
    }

    #[cfg(test)]
    pub(crate) fn mark_invalidated_passes(
        &self,
        invalidated: &[&'static str],
        next_dirty: &mut FxHashSet<&'static str>,
    ) {
        let index = self.dirty_trigger_index();
        for &tag in invalidated {
            if let Some(triggered) = index.get(tag) {
                for &pass_index in triggered {
                    if let Some(pass) = self.passes.get(pass_index) {
                        next_dirty.insert(pass.metadata().name);
                    }
                }
            }
        }
    }

    /// Execute the scheduled passes repeatedly until convergence or max iterations are reached.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError`] if pass dependencies are unsatisfied or the
    /// scheduler fails to converge within the configured iteration bound.
    pub fn run(&self, program: Program) -> Result<Program, OptimizerError> {
        let mut program = program;
        let mut last_pass = "<none>";
        let mut dirty = self.initial_dirty_flags();
        let mut next_dirty = vec![false; self.passes.len()];
        let mut gates = GateFactState::default();

        for _ in 0..self.max_iterations {
            next_dirty.fill(false);
            let (next, changed, changed_by) =
                self.run_once_flags(program, &dirty, &mut next_dirty, &mut gates)?;
            program = next;
            if let Some(name) = changed_by {
                last_pass = name;
            }
            std::mem::swap(&mut dirty, &mut next_dirty);
            if !changed {
                return Ok(program.reconcile_runnable_top_level());
            }
        }
        Err(OptimizerError::MaxIterations {
            max_iterations: self.max_iterations,
            last_pass,
        })
    }

    /// Execute the scheduled passes and return per-pass runtime/IR counters.
    ///
    /// This mirrors [`Self::run`] but retains counters that identify expensive
    /// or clone-heavy passes without requiring a profiler.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError`] if pass dependencies are unsatisfied or the
    /// scheduler fails to converge within the configured iteration bound.
    pub fn run_with_metrics(&self, program: Program) -> Result<OptimizerRunReport, OptimizerError> {
        let mut program = program;
        let mut last_pass = "<none>";
        let mut dirty = self.initial_dirty_flags();
        let mut next_dirty = vec![false; self.passes.len()];
        let mut fact_state = SchedulerFactState::default();
        let mut gates = GateFactState::default();
        let mut metrics = Vec::with_capacity(
            self.execution_order
                .len()
                .saturating_mul(self.max_iterations),
        );

        for iteration in 0..self.max_iterations {
            next_dirty.fill(false);
            let (next, changed, changed_by) = self.run_once_with_metrics(
                program,
                &dirty,
                &mut next_dirty,
                iteration,
                &mut metrics,
                &mut fact_state,
                &mut gates,
            )?;
            program = next;
            if let Some(name) = changed_by {
                last_pass = name;
            }
            std::mem::swap(&mut dirty, &mut next_dirty);
            if !changed {
                return Ok(OptimizerRunReport {
                    program: program.reconcile_runnable_top_level(),
                    passes: metrics,
                });
            }
        }
        Err(OptimizerError::MaxIterations {
            max_iterations: self.max_iterations,
            last_pass,
        })
    }

    /// True when at least one post-condition gate is enabled.
    fn enforces_post_conditions(&self) -> bool {
        self.enforce_cost_monotone
            || self.enforce_effect_handlers
            || self.enforce_linear_types
            || self.enforce_shape_predicates
    }

    /// Certificates of every enabled gate for `program`, reusing the carried
    /// ones when they describe a program with the same fingerprint.
    fn gate_facts_for(&self, state: &mut GateFactState, program: &Program) -> GateFacts {
        match state.entry.take() {
            Some((fingerprint, facts)) if fingerprint == program.fingerprint() => facts,
            _ => self.gate_facts(program),
        }
    }

    /// Derive the certificate of every enabled gate for `program`.
    fn gate_facts(&self, program: &Program) -> GateFacts {
        GateFacts {
            cost: self
                .enforce_cost_monotone
                .then(|| crate::optimizer::cost::CostCertificate::for_program(program)),
            effects: self
                .enforce_effect_handlers
                .then(|| crate::lower::effects::compute_program_effects(program)),
            linear_violations: self
                .enforce_linear_types
                .then(|| linear_type_violations(program)),
            shape_violations: self
                .enforce_shape_predicates
                .then(|| shape_predicate_violations(program)),
        }
    }

    /// Judge one landed rewrite against every enabled gate.
    ///
    /// A rewrite may discharge an existing violation but never introduce a new
    /// one. On acceptance the post-rewrite certificates are returned, because
    /// they are the pre-rewrite certificates of whatever pass runs next.
    fn judge_rewrite(
        &self,
        before: &GateFacts,
        after: &Program,
        allowed_effect_additions: crate::lower::effects::ProgramEffects,
    ) -> Result<GateFacts, GateRejection> {
        let effects = self
            .enforce_effect_handlers
            .then(|| crate::lower::effects::compute_program_effects(after));
        if let (Some(before_effects), Some(after_effects)) = (before.effects, effects) {
            if introduces_forbidden_effects(before_effects, after_effects, allowed_effect_additions)
            {
                return Err(GateRejection::Effects);
            }
        }
        let linear_violations = self
            .enforce_linear_types
            .then(|| linear_type_violations(after));
        if let (Some(before_violations), Some(after_violations)) = (
            before.linear_violations.as_ref(),
            linear_violations.as_ref(),
        ) {
            if introduces_linear_type_violations(before_violations, after_violations) {
                return Err(GateRejection::LinearTypes);
            }
        }
        let shape_violations = self
            .enforce_shape_predicates
            .then(|| shape_predicate_violations(after));
        if let (Some(before_violations), Some(after_violations)) =
            (before.shape_violations.as_ref(), shape_violations.as_ref())
        {
            if introduces_shape_predicate_violations(before_violations, after_violations) {
                return Err(GateRejection::ShapePredicates);
            }
        }
        let cost = self
            .enforce_cost_monotone
            .then(|| crate::optimizer::cost::CostCertificate::for_program(after));
        if let (Some(before_cost), Some(after_cost)) = (before.cost.as_ref(), cost.as_ref()) {
            if !after_cost.dominates_or_equal(before_cost) {
                return Err(GateRejection::Cost);
            }
        }
        Ok(GateFacts {
            cost,
            effects,
            linear_violations,
            shape_violations,
        })
    }

    fn run_once_flags(
        &self,
        mut program: Program,
        dirty: &[bool],
        next_dirty: &mut [bool],
        gates: &mut GateFactState,
    ) -> Result<(Program, bool, Option<&'static str>), OptimizerError> {
        let mut available = (!self.requirements_prevalidated).then(|| {
            let mut available = FxHashSet::default();
            available.reserve(self.execution_order.len());
            available
        });
        let mut changed = false;
        let mut changed_by = None;
        let enforce_gates = self.enforces_post_conditions();
        for &pass_index in &self.execution_order {
            let Some(pass) = self.passes.get(pass_index) else {
                continue;
            };
            let metadata = pass.metadata();
            if let Some(available) = available.as_ref() {
                if !requirements_satisfied(metadata, available) {
                    let missing = metadata
                        .requires
                        .iter()
                        .copied()
                        .find(|requirement| !available.contains(requirement))
                        .unwrap_or("<unknown>");
                    return Err(OptimizerError::UnsatisfiedRequirement {
                        pass: metadata.name,
                        missing,
                    });
                }
            }

            if dirty.get(pass_index).copied().unwrap_or(false) && pass.analyze(&program).should_run
            {
                // One snapshot serves both the gate rollback and the check for a
                // pass that reports a rewrite it did not make.
                let snapshot = program.clone();
                let (next_program, landed) = if enforce_gates {
                    let before = self.gate_facts_for(gates, &program);
                    match pass.try_batch_apply(program) {
                        Ok(result) if result.changed => {
                            match self.judge_rewrite(
                                &before,
                                &result.program,
                                pass.allowed_effect_additions(),
                            ) {
                                Ok(after) => {
                                    next_dirty.fill(true);
                                    // `changed` short-circuits the deep compare:
                                    // once an earlier pass has landed, whether
                                    // this one really rewrote anything changes
                                    // neither return value.
                                    let landed = changed || result.program != snapshot;
                                    gates.store(&result.program, after);
                                    (result.program, landed)
                                }
                                Err(_rejection) => {
                                    gates.store(&snapshot, before);
                                    (snapshot, false)
                                }
                            }
                        }
                        Ok(result) => {
                            gates.store(&snapshot, before);
                            (result.program, false)
                        }
                        Err(_refusal) => {
                            gates.store(&snapshot, before);
                            (snapshot, false)
                        }
                    }
                } else {
                    let result = pass.batch_apply(program);
                    if result.changed {
                        next_dirty.fill(true);
                    }
                    let landed = result.changed && (changed || result.program != snapshot);
                    (result.program, landed)
                };
                if landed {
                    changed = true;
                    changed_by = Some(pass.pass_id());
                }
                program = next_program;
            }
            if let Some(available) = available.as_mut() {
                available.insert(metadata.name);
            }
        }

        Ok((program, changed, changed_by))
    }

    #[expect(
        clippy::too_many_lines,
        reason = "scheduler metric collection keeps before/after counters colocated with pass execution"
    )]
    fn run_once_with_metrics(
        &self,
        mut program: Program,
        dirty: &[bool],
        next_dirty: &mut [bool],
        iteration: usize,
        metrics: &mut Vec<PassRunMetric>,
        fact_state: &mut SchedulerFactState,
        gates: &mut GateFactState,
    ) -> Result<(Program, bool, Option<&'static str>), OptimizerError> {
        let mut available = (!self.requirements_prevalidated).then(|| {
            let mut available = FxHashSet::default();
            available.reserve(self.execution_order.len());
            available
        });
        let mut changed = false;
        let mut changed_by = None;
        let mut cached_allocation_estimate: Option<IrAllocationEstimate> = None;
        let enforce_gates = self.enforces_post_conditions();

        for &pass_index in &self.execution_order {
            let Some(pass) = self.passes.get(pass_index) else {
                continue;
            };
            let metadata = pass.metadata();
            if let Some(available) = available.as_ref() {
                if !requirements_satisfied(metadata, available) {
                    let missing = metadata
                        .requires
                        .iter()
                        .copied()
                        .find(|requirement| !available.contains(requirement))
                        .unwrap_or("<unknown>");
                    return Err(OptimizerError::UnsatisfiedRequirement {
                        pass: metadata.name,
                        missing,
                    });
                }
            }

            let before_stats = *program.stats();
            let before_allocations = *cached_allocation_estimate
                .get_or_insert_with(|| estimate_ir_allocations(&program));

            let mut metric = PassRunMetric {
                iteration,
                pass: metadata.name,
                research_trace: self.research_trace_for(metadata.name),
                ran: false,
                changed: false,
                decision: PassRunDecision::CleanSkipped,
                refusal_kind: None,
                required_analyses: metadata.requires,
                declared_invalidations: metadata.invalidates,
                fact_cache_reused: false,
                fact_cache_recomputed: false,
                fact_cache_invalidated: false,
                effect_bits_before: 0,
                effect_bits_after: 0,
                linear_type_violations_before: 0,
                linear_type_violations_after: 0,
                shape_predicate_violations_before: 0,
                shape_predicate_violations_after: 0,
                runtime_ns: 0,
                nodes_before: before_stats.node_count,
                nodes_after: before_stats.node_count,
                static_storage_bytes_before: before_stats.static_storage_bytes,
                static_storage_bytes_after: before_stats.static_storage_bytes,
                instruction_count_before: before_stats.instruction_count,
                instruction_count_after: before_stats.instruction_count,
                memory_op_count_before: before_stats.memory_op_count,
                memory_op_count_after: before_stats.memory_op_count,
                atomic_op_count_before: before_stats.atomic_op_count,
                atomic_op_count_after: before_stats.atomic_op_count,
                control_flow_count_before: before_stats.control_flow_count,
                control_flow_count_after: before_stats.control_flow_count,
                register_pressure_before: before_stats.register_pressure_estimate,
                register_pressure_after: before_stats.register_pressure_estimate,
                ir_heap_allocations_before: before_allocations.allocations,
                ir_heap_allocations_after: before_allocations.allocations,
                ir_heap_bytes_before: before_allocations.bytes,
                ir_heap_bytes_after: before_allocations.bytes,
            };

            if dirty.get(pass_index).copied().unwrap_or(false) {
                let fact_event = fact_state.prepare(&program);
                metric.fact_cache_reused = fact_event.reused;
                metric.fact_cache_recomputed = fact_event.recomputed;
                if !pass.analyze(&program).should_run {
                    metric.decision = PassRunDecision::AnalysisSkipped;
                    metrics.push(metric);
                    if let Some(available) = available.as_mut() {
                        available.insert(metadata.name);
                    }
                    continue;
                }
                // One snapshot serves both the gate rollback and the check for a
                // pass that reports a rewrite it did not make.
                let snapshot = program.clone();
                metric.ran = true;
                let perf_scope = PerfScope::start("vyre-foundation", metadata.name);
                let (next_program, landed_changed) = if enforce_gates {
                    let before = self.gate_facts_for(gates, &program);
                    metric.effect_bits_before = before.effect_bits();
                    metric.linear_type_violations_before = before.linear_violation_count();
                    metric.shape_predicate_violations_before = before.shape_violation_count();
                    let result = pass.try_batch_apply(program);
                    metric.runtime_ns = u128::from(perf_scope.finish().elapsed_ns);
                    match result {
                        Ok(result) if result.changed => {
                            match self.judge_rewrite(
                                &before,
                                &result.program,
                                pass.allowed_effect_additions(),
                            ) {
                                Ok(after) => {
                                    let landed = result.program != snapshot;
                                    metric.decision = if landed {
                                        PassRunDecision::Changed
                                    } else {
                                        PassRunDecision::RanUnchanged
                                    };
                                    metric.effect_bits_after = after.effect_bits();
                                    metric.linear_type_violations_after =
                                        after.linear_violation_count();
                                    metric.shape_predicate_violations_after =
                                        after.shape_violation_count();
                                    gates.store(&result.program, after);
                                    (result.program, landed)
                                }
                                Err(rejection) => {
                                    // A tracked post-condition regressed without
                                    // an explicit refusal. Drop the rewrite and
                                    // restore the snapshot; the counters below
                                    // describe the post-revert program.
                                    metric.decision = rejection.decision();
                                    metric.effect_bits_after = before.effect_bits();
                                    metric.linear_type_violations_after =
                                        before.linear_violation_count();
                                    metric.shape_predicate_violations_after =
                                        before.shape_violation_count();
                                    gates.store(&snapshot, before);
                                    (snapshot, false)
                                }
                            }
                        }
                        Ok(result) => {
                            metric.decision = PassRunDecision::RanUnchanged;
                            metric.effect_bits_after = before.effect_bits();
                            metric.linear_type_violations_after = before.linear_violation_count();
                            metric.shape_predicate_violations_after =
                                before.shape_violation_count();
                            gates.store(&snapshot, before);
                            (result.program, false)
                        }
                        Err(refusal) => {
                            metric.decision = PassRunDecision::Refused;
                            metric.refusal_kind = Some(refusal.kind());
                            metric.effect_bits_after = before.effect_bits();
                            metric.linear_type_violations_after = before.linear_violation_count();
                            metric.shape_predicate_violations_after =
                                before.shape_violation_count();
                            gates.store(&snapshot, before);
                            (snapshot, false)
                        }
                    }
                } else {
                    let result = pass.batch_apply(program);
                    metric.runtime_ns = u128::from(perf_scope.finish().elapsed_ns);
                    let landed = result.changed && result.program != snapshot;
                    metric.decision = if landed {
                        PassRunDecision::Changed
                    } else {
                        PassRunDecision::RanUnchanged
                    };
                    (result.program, landed)
                };
                program = next_program;
                let after_stats = *program.stats();
                let after_allocations = if landed_changed {
                    estimate_ir_allocations(&program)
                } else {
                    before_allocations
                };
                cached_allocation_estimate = Some(after_allocations);
                metric.nodes_after = after_stats.node_count;
                metric.static_storage_bytes_after = after_stats.static_storage_bytes;
                metric.instruction_count_after = after_stats.instruction_count;
                metric.memory_op_count_after = after_stats.memory_op_count;
                metric.atomic_op_count_after = after_stats.atomic_op_count;
                metric.control_flow_count_after = after_stats.control_flow_count;
                metric.register_pressure_after = after_stats.register_pressure_estimate;
                metric.ir_heap_allocations_after = after_allocations.allocations;
                metric.ir_heap_bytes_after = after_allocations.bytes;
                // Reflect post-gate state. Expression-only rewrites often keep
                // the same node count; they still invalidate downstream facts.
                metric.changed = landed_changed;
                if metric.changed {
                    changed = true;
                    changed_by = Some(pass.pass_id());
                    metric.fact_cache_invalidated = fact_state.invalidate();
                    next_dirty.fill(true);
                }
            }
            metrics.push(metric);
            if let Some(available) = available.as_mut() {
                available.insert(metadata.name);
            }
        }

        Ok((program, changed, changed_by))
    }

    fn initial_dirty_flags(&self) -> Vec<bool> {
        self.initial_dirty_flags_cache
            .get_or_init(|| vec![true; self.passes.len()])
            .clone()
    }
}
