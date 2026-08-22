//! Binary-search and AST-level counterexample shrinking.
//!
//! Given a witness and a predicate that returns `true` when the witness
//! triggers a failure, shrinks toward the smallest still-failing representation.
//! Deterministic, bounded by explicit work and time limits, and preserves
//! validator acceptance on every intermediate and final step.

use std::time::{Duration, Instant};

use vyre_foundation::ir::{BufferDecl, Expr, Node, Program};
use vyre_foundation::validate::validate;
use vyre_foundation::visit::child_bodies;

/// Budget parameters for IR minimization.
#[derive(Debug, Clone, Copy)]
pub struct MinimizationBudget {
    /// Maximum candidate evaluations before returning the best found so far.
    pub max_steps: usize,
    /// Maximum wall-clock time for the minimization pass.
    pub max_duration: Option<Duration>,
}

impl Default for MinimizationBudget {
    fn default() -> Self {
        Self {
            max_steps: 500,
            max_duration: Some(Duration::from_secs(10)),
        }
    }
}

/// Statistics collected during program counterexample minimization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinimizerReport {
    /// Total node count in the starting program.
    pub initial_nodes: usize,
    /// Total node count in the minimized program.
    pub final_nodes: usize,
    /// Total buffer declarations in the starting program.
    pub initial_buffers: usize,
    /// Total buffer declarations in the minimized program.
    pub final_buffers: usize,
    /// Candidate evaluations evaluated during minimization.
    pub steps_taken: usize,
    /// Elapsed wall-clock nanoseconds.
    pub elapsed_ns: u64,
}

/// Shrinking engine.
pub struct CounterexampleMinimizer;

impl CounterexampleMinimizer {
    /// Shrink a failing u32 witness toward the smallest still-failing value.
    ///
    /// The shrinker tries progressively smaller halves; if a smaller value
    /// still satisfies the predicate it becomes the new candidate. Returns
    /// the minimum failing witness seen.
    ///
    /// # Examples
    ///
    /// ```
    /// use vyre_conform::CounterexampleMinimizer;
    /// // Predicate: "value > 100 triggers the bug".
    /// let minimized = CounterexampleMinimizer::shrink_u32(1_000, |v| v > 100);
    /// assert_eq!(minimized, 101);
    /// ```
    pub fn shrink_u32<F>(failing: u32, predicate: F) -> u32
    where
        F: Fn(u32) -> bool,
    {
        debug_assert!(
            predicate(failing),
            "caller must pass a value the predicate rejects"
        );
        let mut best = failing;
        let mut lo = 0u32;
        let mut hi = failing;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if predicate(mid) {
                best = mid;
                hi = mid;
            } else {
                if mid == u32::MAX {
                    break;
                }
                lo = mid + 1;
            }
        }
        best
    }

    /// Shrink a failing Program toward the smallest still-failing valid Program.
    #[must_use]
    pub fn shrink_program<F>(failing: &Program, predicate: F) -> Program
    where
        F: Fn(&Program) -> bool,
    {
        let (minimized, _) =
            Self::shrink_program_with_budget(failing, MinimizationBudget::default(), predicate);
        minimized
    }

    /// Shrink a failing Program with an explicit work and time budget.
    pub fn shrink_program_with_budget<F>(
        failing: &Program,
        budget: MinimizationBudget,
        predicate: F,
    ) -> (Program, MinimizerReport)
    where
        F: Fn(&Program) -> bool,
    {
        let start_time = Instant::now();
        let initial_nodes = count_nodes(failing.entry());
        let initial_buffers = failing.buffers().len();
        let mut steps = 0usize;

        debug_assert!(
            predicate(failing),
            "initial program must satisfy the failing predicate"
        );

        let mut current = failing.clone();

        loop {
            let before_nodes = count_nodes(current.entry());
            let before_buffers = current.buffers().len();

            // 1. Shrink entry nodes (try removing statements)
            current = shrink_entry_nodes(&current, &predicate, &mut steps, budget, start_time);
            if budget_exceeded(steps, budget, start_time) {
                break;
            }

            // 2. Shrink nested bodies (blocks, if branches, loops)
            current = shrink_nested_bodies(&current, &predicate, &mut steps, budget, start_time);
            if budget_exceeded(steps, budget, start_time) {
                break;
            }

            // 3. Shrink expressions (simplify compound operations)
            current = shrink_expressions(&current, &predicate, &mut steps, budget, start_time);
            if budget_exceeded(steps, budget, start_time) {
                break;
            }

            // 4. Shrink literals (shrink constant integer / float values)
            current = shrink_literals(&current, &predicate, &mut steps, budget, start_time);
            if budget_exceeded(steps, budget, start_time) {
                break;
            }

            // 5. Shrink unused buffer declarations
            current = shrink_buffers(&current, &predicate, &mut steps, budget, start_time);
            if budget_exceeded(steps, budget, start_time) {
                break;
            }

            // 6. Shrink workgroup size toward [1, 1, 1]
            current = shrink_workgroup_size(&current, &predicate, &mut steps, budget, start_time);
            if budget_exceeded(steps, budget, start_time) {
                break;
            }

            let after_nodes = count_nodes(current.entry());
            let after_buffers = current.buffers().len();

            if after_nodes == before_nodes && after_buffers == before_buffers {
                break;
            }
        }

        let elapsed_ns = start_time.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let final_nodes = count_nodes(current.entry());
        let final_buffers = current.buffers().len();

        let report = MinimizerReport {
            initial_nodes,
            final_nodes,
            initial_buffers,
            final_buffers,
            steps_taken: steps,
            elapsed_ns,
        };

        (current, report)
    }
}

fn count_nodes(nodes: &[Node]) -> usize {
    let mut total = nodes.len();
    for node in nodes {
        for body in child_bodies(node) {
            total += count_nodes(body);
        }
    }
    total
}

fn budget_exceeded(steps: usize, budget: MinimizationBudget, start_time: Instant) -> bool {
    if steps >= budget.max_steps {
        return true;
    }
    if let Some(max_duration) = budget.max_duration {
        if start_time.elapsed() >= max_duration {
            return true;
        }
    }
    false
}

fn make_program(
    buffers: &[BufferDecl],
    workgroup_size: [u32; 3],
    entry: Vec<Node>,
    entry_op_id: Option<&str>,
) -> Program {
    let mut prog = Program::from_raw_parts(buffers.to_vec(), workgroup_size, entry);
    prog.entry_op_id = entry_op_id.map(ToString::to_string);
    prog
}

fn try_candidate<F>(
    candidate: Program,
    predicate: &F,
    steps: &mut usize,
    budget: MinimizationBudget,
    start_time: Instant,
) -> Option<Program>
where
    F: Fn(&Program) -> bool,
{
    *steps += 1;
    if budget_exceeded(*steps, budget, start_time) {
        return None;
    }
    if validate(&candidate).is_empty() && predicate(&candidate) {
        Some(candidate)
    } else {
        None
    }
}

fn shrink_entry_nodes<F>(
    program: &Program,
    predicate: &F,
    steps: &mut usize,
    budget: MinimizationBudget,
    start_time: Instant,
) -> Program
where
    F: Fn(&Program) -> bool,
{
    let mut current_entry = program.entry().to_vec();
    let buffers = program.buffers();
    let workgroup_size = program.workgroup_size;
    let op_id = program.entry_op_id.as_deref();

    let mut i = 0;
    while i < current_entry.len() {
        if budget_exceeded(*steps, budget, start_time) {
            break;
        }
        let mut candidate_entry = current_entry.clone();
        candidate_entry.remove(i);

        let candidate = make_program(buffers, workgroup_size, candidate_entry, op_id);
        if let Some(accepted) = try_candidate(candidate, predicate, steps, budget, start_time) {
            current_entry = accepted.entry().to_vec();
        } else {
            i += 1;
        }
    }

    make_program(buffers, workgroup_size, current_entry, op_id)
}

fn shrink_nested_bodies<F>(
    program: &Program,
    predicate: &F,
    steps: &mut usize,
    budget: MinimizationBudget,
    start_time: Instant,
) -> Program
where
    F: Fn(&Program) -> bool,
{
    let mut current_entry = program.entry().to_vec();
    let buffers = program.buffers();
    let workgroup_size = program.workgroup_size;
    let op_id = program.entry_op_id.as_deref();

    for node_idx in 0..current_entry.len() {
        if budget_exceeded(*steps, budget, start_time) {
            break;
        }
        match current_entry[node_idx].clone() {
            Node::If {
                cond,
                then,
                otherwise,
            } => {
                // Try replacing with empty otherwise
                if !otherwise.is_empty() {
                    let mut candidate_entry = current_entry.clone();
                    candidate_entry[node_idx] = Node::If {
                        cond: cond.clone(),
                        then: then.clone(),
                        otherwise: Vec::new(),
                    };
                    let candidate = make_program(buffers, workgroup_size, candidate_entry, op_id);
                    if let Some(accepted) =
                        try_candidate(candidate, predicate, steps, budget, start_time)
                    {
                        current_entry = accepted.entry().to_vec();
                    }
                }
            }
            Node::Loop {
                var,
                from,
                to,
                body,
            } => {
                // Try removing nodes inside the loop body
                let mut current_body = body.clone();
                let mut b_idx = 0;
                while b_idx < current_body.len() {
                    if budget_exceeded(*steps, budget, start_time) {
                        break;
                    }
                    let mut candidate_body = current_body.clone();
                    candidate_body.remove(b_idx);

                    let mut candidate_entry = current_entry.clone();
                    candidate_entry[node_idx] = Node::Loop {
                        var: var.clone(),
                        from: from.clone(),
                        to: to.clone(),
                        body: candidate_body,
                    };
                    let candidate = make_program(buffers, workgroup_size, candidate_entry, op_id);
                    if let Some(accepted) =
                        try_candidate(candidate, predicate, steps, budget, start_time)
                    {
                        current_entry = accepted.entry().to_vec();
                        if let Node::Loop { body: new_b, .. } = &current_entry[node_idx] {
                            current_body = new_b.clone();
                        }
                    } else {
                        b_idx += 1;
                    }
                }
            }
            Node::Region {
                generator,
                source_region,
                body,
            } => {
                let mut current_body = body.clone();
                let mut b_idx = 0;
                while b_idx < current_body.len() {
                    if budget_exceeded(*steps, budget, start_time) {
                        break;
                    }
                    let mut candidate_body = (*current_body).clone();
                    candidate_body.remove(b_idx);

                    let mut candidate_entry = current_entry.clone();
                    candidate_entry[node_idx] = Node::Region {
                        generator: generator.clone(),
                        source_region: source_region.clone(),
                        body: std::sync::Arc::new(candidate_body),
                    };
                    let candidate = make_program(buffers, workgroup_size, candidate_entry, op_id);
                    if let Some(accepted) =
                        try_candidate(candidate, predicate, steps, budget, start_time)
                    {
                        current_entry = accepted.entry().to_vec();
                        if let Node::Region { body: new_b, .. } = &current_entry[node_idx] {
                            current_body = new_b.clone();
                        }
                    } else {
                        b_idx += 1;
                    }
                }
            }
            _ => {}
        }
    }

    make_program(buffers, workgroup_size, current_entry, op_id)
}

fn simplify_expr(expr: &Expr) -> Vec<Expr> {
    let mut candidates = Vec::new();
    match expr {
        Expr::BinOp { left, right, .. } => {
            candidates.push((**left).clone());
            candidates.push((**right).clone());
        }
        Expr::Fma { a, b, c } => {
            candidates.push((**a).clone());
            candidates.push((**b).clone());
            candidates.push((**c).clone());
        }
        Expr::Select {
            true_val,
            false_val,
            ..
        } => {
            candidates.push((**true_val).clone());
            candidates.push((**false_val).clone());
        }
        Expr::Cast { value, .. } => {
            candidates.push((**value).clone());
        }
        _ => {}
    }
    candidates
}

fn shrink_expressions<F>(
    program: &Program,
    predicate: &F,
    steps: &mut usize,
    budget: MinimizationBudget,
    start_time: Instant,
) -> Program
where
    F: Fn(&Program) -> bool,
{
    let mut current_entry = program.entry().to_vec();
    let buffers = program.buffers();
    let workgroup_size = program.workgroup_size;
    let op_id = program.entry_op_id.as_deref();

    for node_idx in 0..current_entry.len() {
        if budget_exceeded(*steps, budget, start_time) {
            break;
        }
        let node = &current_entry[node_idx];
        match node {
            Node::Let { name, value } => {
                for simplified in simplify_expr(value) {
                    let mut candidate_entry = current_entry.clone();
                    candidate_entry[node_idx] = Node::Let {
                        name: name.clone(),
                        value: simplified,
                    };
                    let candidate = make_program(buffers, workgroup_size, candidate_entry, op_id);
                    if let Some(accepted) =
                        try_candidate(candidate, predicate, steps, budget, start_time)
                    {
                        current_entry = accepted.entry().to_vec();
                        break;
                    }
                }
            }
            Node::Assign { name, value } => {
                for simplified in simplify_expr(value) {
                    let mut candidate_entry = current_entry.clone();
                    candidate_entry[node_idx] = Node::Assign {
                        name: name.clone(),
                        value: simplified,
                    };
                    let candidate = make_program(buffers, workgroup_size, candidate_entry, op_id);
                    if let Some(accepted) =
                        try_candidate(candidate, predicate, steps, budget, start_time)
                    {
                        current_entry = accepted.entry().to_vec();
                        break;
                    }
                }
            }
            Node::Store {
                buffer,
                index,
                value,
            } => {
                for simplified in simplify_expr(value) {
                    let mut candidate_entry = current_entry.clone();
                    candidate_entry[node_idx] = Node::Store {
                        buffer: buffer.clone(),
                        index: index.clone(),
                        value: simplified,
                    };
                    let candidate = make_program(buffers, workgroup_size, candidate_entry, op_id);
                    if let Some(accepted) =
                        try_candidate(candidate, predicate, steps, budget, start_time)
                    {
                        current_entry = accepted.entry().to_vec();
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    make_program(buffers, workgroup_size, current_entry, op_id)
}

fn shrink_literals<F>(
    program: &Program,
    predicate: &F,
    steps: &mut usize,
    budget: MinimizationBudget,
    start_time: Instant,
) -> Program
where
    F: Fn(&Program) -> bool,
{
    let mut current_entry = program.entry().to_vec();
    let buffers = program.buffers();
    let workgroup_size = program.workgroup_size;
    let op_id = program.entry_op_id.as_deref();

    for node_idx in 0..current_entry.len() {
        if budget_exceeded(*steps, budget, start_time) {
            break;
        }
        let node = &current_entry[node_idx];
        match node {
            Node::Let {
                name,
                value: Expr::LitU32(val),
            } if *val > 1 => {
                for &smaller in &[0, 1, val / 2] {
                    let mut candidate_entry = current_entry.clone();
                    candidate_entry[node_idx] = Node::Let {
                        name: name.clone(),
                        value: Expr::LitU32(smaller),
                    };
                    let candidate = make_program(buffers, workgroup_size, candidate_entry, op_id);
                    if let Some(accepted) =
                        try_candidate(candidate, predicate, steps, budget, start_time)
                    {
                        current_entry = accepted.entry().to_vec();
                        break;
                    }
                }
            }
            Node::Let {
                name,
                value: Expr::LitI32(val),
            } if *val != 0 && *val != 1 => {
                for &smaller in &[0, 1, val / 2] {
                    let mut candidate_entry = current_entry.clone();
                    candidate_entry[node_idx] = Node::Let {
                        name: name.clone(),
                        value: Expr::LitI32(smaller),
                    };
                    let candidate = make_program(buffers, workgroup_size, candidate_entry, op_id);
                    if let Some(accepted) =
                        try_candidate(candidate, predicate, steps, budget, start_time)
                    {
                        current_entry = accepted.entry().to_vec();
                        break;
                    }
                }
            }
            Node::Let {
                name,
                value: Expr::LitF32(val),
            } if *val != 0.0 && *val != 1.0 => {
                for &smaller in &[0.0, 1.0] {
                    let mut candidate_entry = current_entry.clone();
                    candidate_entry[node_idx] = Node::Let {
                        name: name.clone(),
                        value: Expr::LitF32(smaller),
                    };
                    let candidate = make_program(buffers, workgroup_size, candidate_entry, op_id);
                    if let Some(accepted) =
                        try_candidate(candidate, predicate, steps, budget, start_time)
                    {
                        current_entry = accepted.entry().to_vec();
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    make_program(buffers, workgroup_size, current_entry, op_id)
}

fn shrink_buffers<F>(
    program: &Program,
    predicate: &F,
    steps: &mut usize,
    budget: MinimizationBudget,
    start_time: Instant,
) -> Program
where
    F: Fn(&Program) -> bool,
{
    let mut current_buffers = program.buffers().to_vec();
    let entry = program.entry().to_vec();
    let workgroup_size = program.workgroup_size;
    let op_id = program.entry_op_id.as_deref();

    let mut i = 0;
    while i < current_buffers.len() {
        if budget_exceeded(*steps, budget, start_time) {
            break;
        }
        let mut candidate_buffers = current_buffers.clone();
        candidate_buffers.remove(i);

        let candidate = make_program(&candidate_buffers, workgroup_size, entry.clone(), op_id);
        if let Some(accepted) = try_candidate(candidate, predicate, steps, budget, start_time) {
            current_buffers = accepted.buffers().to_vec();
        } else {
            i += 1;
        }
    }

    make_program(&current_buffers, workgroup_size, entry, op_id)
}

fn shrink_workgroup_size<F>(
    program: &Program,
    predicate: &F,
    steps: &mut usize,
    budget: MinimizationBudget,
    start_time: Instant,
) -> Program
where
    F: Fn(&Program) -> bool,
{
    let buffers = program.buffers();
    let entry = program.entry().to_vec();
    let op_id = program.entry_op_id.as_deref();

    if program.workgroup_size != [1, 1, 1] {
        let candidate = make_program(buffers, [1, 1, 1], entry.clone(), op_id);
        if let Some(accepted) = try_candidate(candidate, predicate, steps, budget, start_time) {
            return accepted;
        }
    }

    program.clone()
}
