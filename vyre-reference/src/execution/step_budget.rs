//! Termination contract for the reference interpreter.
//!
//! The interpreter bounds the memory a program may ask for. Without a work
//! bound a program with a data-derived trip count runs until it finishes, so
//! the parity oracle has no termination contract and a caller waiting on it has
//! no ceiling. A caller that bounds the oracle from outside, on a thread it
//! abandons, is a workaround rather than a contract.
//!
//! One step is one statement the interpreter advances for one invocation, plus
//! one per loop iteration boundary so an empty body cannot spin for free. The
//! budget is armed once per evaluation and shared by nested evaluations, so a
//! program cannot dodge the ceiling by nesting.
//!
//! A fixed ceiling answers the wrong question for a large program that is
//! nonetheless bounded. A 256-square dense contraction declares every one of
//! its trip counts as a constant, terminates, and charges 67 million steps,
//! which a ceiling sized for the fixture corpus refuses. Refusing it says the
//! oracle cannot evaluate a program whose work it can count in advance, and
//! raising the fixed ceiling to admit it would take the refusal for an
//! unbounded program with it.
//!
//! So the ceiling has two parts.
//! [`MAX_REFERENCE_STEPS`](crate::execution::step_budget::MAX_REFERENCE_STEPS)
//! is the floor, which is what a program with a data-derived trip count is
//! refused against because nothing declares its work. Once the dispatch
//! geometry is resolved, `admit_declared_work` raises the ceiling to the work
//! the program's own constant extents declare.
//! That is the distinction the refusal already asks for: bound the trip counts
//! by a declared extent, and the declaration is what the oracle admits.

use std::cell::{Cell, RefCell};

use vyre_foundation::ir::{Expr, Node, Program};
use vyre_foundation::visit::child_bodies;

use crate::error::{ReferenceError, StepCeilingExceeded};

/// Steps the heaviest legitimate reference run in the registered operation
/// fixture corpus charges.
///
/// Measured, not chosen: 911,388 steps on `vyre-libs::math::symmetric_eigen_jacobi`.
/// `the_reference_step_ceiling_is_derived_from_the_corpus`
/// in `vyre-libs` evaluates every registered fixture case through
/// [`crate::ReferenceRequest::outputs_and_steps`] and fails when a run charges more than
/// this, printing the number to record here. Raise it only to a value that test
/// printed.
pub const MEASURED_HEAVIEST_CORPUS_STEPS: u64 = 911_388;

/// Multiple of the heaviest measured legitimate run the ceiling stands above.
///
/// Two bounds decide it. Below, a legitimate program larger than the heaviest
/// fixture must still finish, so the multiple is generous. Above, a refusal has
/// to arrive in bounded time rather than after the interpreter has burned an
/// afternoon, and the interpreter charges a few million steps a second, so the
/// ceiling stays in the tens of millions.
pub const STEP_CEILING_HEADROOM: u64 = 64;

/// Steps one reference evaluation may execute.
///
/// Derived, so the number cannot be edited without moving the measurement it
/// rests on: the `vyre-libs` corpus test asserts this product holds.
pub const MAX_REFERENCE_STEPS: u64 = MEASURED_HEAVIEST_CORPUS_STEPS * STEP_CEILING_HEADROOM;

/// Multiple of a program's own declared work the ceiling stands above.
///
/// Small, because a declared bound is already an over-estimate: a conditional
/// contributes both arms, and only one of them runs. What the multiple covers is
/// interpreter bookkeeping the statement count does not model, chiefly the
/// round-robin scheduling charge that bounds a barrier-release cycle.
pub const DECLARED_WORK_HEADROOM: u64 = 4;

thread_local! {
    /// Steps charged by the evaluation armed on this thread and the ceiling it
    /// was armed with. `None` while no evaluation is armed. A `Cell` so the
    /// charging path is one read and one write with no borrow flag.
    static BUDGET: Cell<Option<(u64, u64, bool)>> = const { Cell::new(None) };
    /// Buffer bytes the armed evaluation has allocated, and the ceilings it
    /// may reach. Separate from `BUDGET` because allocation and frame entry
    /// are cold, and widening the per-step cell would charge every step for
    /// them.
    static ALLOCATION: Cell<Limits> = const { Cell::new(Limits::UNARMED) };
    /// The armed program's name, read only when a refusal is built.
    static PROGRAM: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Memory and frame-depth ceilings of the armed evaluation.
#[derive(Clone, Copy)]
struct Limits {
    /// Buffer bytes allocated so far by this evaluation.
    allocated_bytes: usize,
    /// Buffer bytes this evaluation may allocate in total.
    max_memory_bytes: usize,
    /// Frame depth one lane may reach.
    max_recursion_depth: usize,
}

impl Limits {
    /// No evaluation armed: nothing is charged and nothing is refused.
    const UNARMED: Self = Self {
        allocated_bytes: 0,
        max_memory_bytes: usize::MAX,
        max_recursion_depth: usize::MAX,
    };
}

/// Restores the enclosing evaluation's budget when one evaluation ends.
///
/// A nested evaluation observes an armed budget and leaves it in place, so its
/// steps are charged to the outer ceiling and dropping its guard is a no-op.
pub(crate) struct BudgetGuard {
    outermost: bool,
}

impl Drop for BudgetGuard {
    fn drop(&mut self) {
        if self.outermost {
            BUDGET.with(|budget| budget.set(None));
            ALLOCATION.with(|limits| limits.set(Limits::UNARMED));
        }
    }
}

/// Arm [`ReferenceBudget::standard`](crate::ReferenceBudget::standard) for one
/// evaluation of `program`, or join the enclosing evaluation's budget.
///
/// The evaluator calls this so a program reached through a nested path is
/// bounded by the same contract as one submitted through a request.
pub(crate) fn arm(program: &Program) -> BudgetGuard {
    arm_with_budget(program, crate::ReferenceBudget::standard())
}

/// Arm one evaluation's work, memory, and frame-depth budget, or join the
/// enclosing one.
pub(crate) fn arm_with_budget(program: &Program, budget: crate::ReferenceBudget) -> BudgetGuard {
    if BUDGET.with(Cell::get).is_some() {
        return BudgetGuard { outermost: false };
    }
    let label = program_label(program);
    PROGRAM.with_borrow_mut(|armed| *armed = label);
    BUDGET.with(|cell| {
        cell.set(Some((0, budget.work_ceiling, !budget.admit_declared_work)));
    });
    ALLOCATION.with(|limits| {
        limits.set(Limits {
            allocated_bytes: 0,
            max_memory_bytes: budget.max_memory_bytes,
            max_recursion_depth: budget.max_recursion_depth,
        });
    });
    BudgetGuard { outermost: true }
}

/// Charge `bytes` of buffer allocation against the armed memory ceiling.
///
/// The ceiling bounds what one evaluation allocates in total. A run that
/// crosses it ends with a structured refusal naming the bound, which is the
/// only outcome that is neither a wrong answer nor an allocation failure that
/// ends the host process.
///
/// # Errors
/// Refuses with `BudgetExhaustion` when the armed allocation ceiling is
/// crossed.
pub(crate) fn charge_memory(bytes: usize, buffer: &str) -> Result<(), ReferenceError> {
    ALLOCATION.with(|cell| {
        let mut limits = cell.get();
        let Some(allocated) = limits.allocated_bytes.checked_add(bytes) else {
            return Err(ReferenceError::budget_exhaustion(format!(
                "allocating buffer `{buffer}` overflows the host address space. Fix: declare a \
                 buffer whose byte length this host can address."
            )));
        };
        if allocated > limits.max_memory_bytes {
            return Err(ReferenceError::budget_exhaustion(format!(
                "allocating {bytes} byte(s) for buffer `{buffer}` reaches {allocated} bytes, past \
                 the {} byte allocation ceiling this request armed. Fix: raise \
                 `ReferenceBudget::max_memory_bytes`, or submit a program whose declared buffers \
                 fit the bound.",
                limits.max_memory_bytes
            )));
        }
        limits.allocated_bytes = allocated;
        cell.set(limits);
        Ok(())
    })
}

/// Refuse a lane whose frame depth has reached the armed recursion ceiling.
///
/// `depth` is the depth the lane stands at once the frame is pushed.
///
/// # Errors
/// Refuses with `BudgetExhaustion` when the armed frame ceiling is crossed.
pub(crate) fn check_recursion_depth(depth: usize) -> Result<(), ReferenceError> {
    let max = ALLOCATION.with(|cell| cell.get().max_recursion_depth);
    if depth <= max {
        return Ok(());
    }
    Err(ReferenceError::budget_exhaustion(format!(
        "entering a nested body at frame depth {depth} passes the {max} frame ceiling this \
         request armed. Fix: raise `ReferenceBudget::max_recursion_depth`, or submit a program \
         whose block nesting fits the bound."
    )))
}

/// Raise the armed ceiling to admit the work `body` declares over
/// `invocations`, once the dispatch geometry that fixes both is resolved.
///
/// A no-op when no budget is armed, when the body declares a trip count no
/// constant bounds, or when the declared work is already inside the ceiling.
/// The ceiling only ever rises here: a program whose extents are declared is
/// admitted, and a program whose extents are not keeps
/// [`MAX_REFERENCE_STEPS`].
///
/// Raising an enclosing evaluation's ceiling is correct rather than an escape:
/// the outer run has to execute the nested program's declared work to finish,
/// and that work is declared by the same constant extents.
pub(crate) fn admit_declared_work(body: &[Node], invocations: u64) {
    let Some(per_invocation) = declared_statements(body) else {
        return;
    };
    let declared = per_invocation
        .saturating_mul(invocations)
        .saturating_add(per_invocation)
        .saturating_mul(DECLARED_WORK_HEADROOM);
    BUDGET.with(|budget| {
        if let Some((charged, ceiling, exact)) = budget.get() {
            if !exact && declared > ceiling {
                budget.set(Some((charged, declared, false)));
            }
        }
    });
}

/// Steps one invocation of `body` charges at most, or `None` when a loop's trip
/// count is not fixed by constant bounds.
///
/// An upper bound, not a prediction: a conditional contributes both arms,
/// because which one runs is what the data decides. Every statement charges
/// once when the interpreter advances to it, and every loop iteration charges
/// one boundary step on top of its body.
fn declared_statements(body: &[Node]) -> Option<u64> {
    let mut total = 0u64;
    for node in body {
        total = total.checked_add(declared_statements_of(node)?)?;
    }
    Some(total)
}

/// [`declared_statements`] for one node, including the node's own step.
///
/// Only the loop arm makes a per-variant decision, because a trip count is what
/// multiplies a body. Every other variant's children come from
/// [`child_bodies`], so a nesting variant added to `Node` is descended into here
/// without an edit rather than counted as a leaf.
fn declared_statements_of(node: &Node) -> Option<u64> {
    let nested = match node {
        Node::Loop { from, to, body, .. } => {
            let trips = u64::from(constant_u32(to)?.checked_sub(constant_u32(from)?)?);
            declared_statements(body)?
                .checked_add(1)?
                .checked_mul(trips)?
        }
        other => {
            let mut total = 0u64;
            for children in child_bodies(other) {
                total = total.checked_add(declared_statements(children)?)?;
            }
            total
        }
    };
    nested.checked_add(1)
}

/// The value `expr` states outright, or `None` when it states a computation.
///
/// Only a literal counts. A constant-folded arithmetic expression would need
/// the optimizer's evaluator, and a program that reaches the interpreter with a
/// foldable bound has already declared its extent to the pass that folds it.
fn constant_u32(expr: &Expr) -> Option<u32> {
    match expr {
        Expr::LitU32(value) => Some(*value),
        Expr::LitI32(value) => u32::try_from(*value).ok(),
        _ => None,
    }
}

/// Charge one interpreter step against the armed budget.
///
/// Charging with no budget armed is a no-op, so an evaluator driven directly by
/// a unit test does not have to arm one.
///
/// One thread-local access does the read and the write. This runs once per
/// statement of every interpreted invocation, so a second lookup here is a
/// second lookup per step of the whole corpus.
#[inline]
pub(crate) fn charge() -> Result<(), ReferenceError> {
    let exceeded = BUDGET.with(|budget| match budget.get() {
        None => None,
        Some((charged, ceiling, exact)) => {
            let charged = charged + 1;
            if charged > ceiling {
                return Some(ceiling);
            }
            budget.set(Some((charged, ceiling, exact)));
            None
        }
    });
    match exceeded {
        None => Ok(()),
        Some(ceiling) => Err(ReferenceError::step_ceiling(StepCeilingExceeded {
            program: PROGRAM.with_borrow(Clone::clone),
            ceiling,
        })),
    }
}

/// Steps charged by the armed evaluation, or `0` when none is armed.
pub(crate) fn charged() -> u64 {
    BUDGET.with(|budget| budget.get().map_or(0, |(charged, _, _)| charged))
}

/// The name a program is refused under: its entry operation id when it declares
/// one, otherwise a fingerprint prefix, because anonymous IR still has to be
/// identifiable in a refusal.
fn program_label(program: &Program) -> String {
    if let Some(id) = program.entry_op_id.as_ref() {
        return id.clone();
    }
    let fingerprint = program.fingerprint();
    let mut label = String::with_capacity(2 + 16);
    label.push_str("0x");
    for byte in fingerprint.iter().take(8) {
        use std::fmt::Write;
        let _ = write!(label, "{byte:02x}");
    }
    label
}

// Inline: `arm_with_budget`, `charge` and `charged` are crate-private, and the nesting
// rule is not observable through a public entry point.
#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node};

    fn tiny_program() -> Program {
        Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
        )
    }

    #[test]
    fn a_charge_with_no_budget_armed_is_a_no_op() {
        assert_eq!(charged(), 0);
        charge().expect("Fix: an unarmed interpreter must charge nothing");
        assert_eq!(
            charged(),
            0,
            "Fix: charging without an armed budget must not accumulate"
        );
    }

    #[test]
    fn the_ceiling_refuses_the_step_that_exceeds_it_and_not_the_one_that_reaches_it() {
        let program = tiny_program();
        let guard = arm_with_budget(&program, crate::ReferenceBudget::bounded(2));
        charge().expect("Fix: the first step is within a ceiling of two");
        charge().expect("Fix: the second step reaches the ceiling and is admitted");
        let error = charge().expect_err("Fix: the third step exceeds a ceiling of two");
        let source = error
            .step_ceiling_source()
            .expect("Fix: the refusal carries the ceiling it exceeded");
        assert_eq!(source.ceiling, 2);
        assert_eq!(
            charged(),
            2,
            "Fix: a refused step must not be charged, so the count states what actually ran"
        );
        drop(guard);
        assert_eq!(charged(), 0, "Fix: the outermost guard disarms the budget");
    }

    #[test]
    fn a_nested_evaluation_is_charged_to_the_enclosing_ceiling() {
        let program = tiny_program();
        let outer = arm_with_budget(&program, crate::ReferenceBudget::bounded(10));
        charge().expect("Fix: the outer evaluation charges its own step");
        {
            let inner = arm_with_budget(&program, crate::ReferenceBudget::bounded(u64::MAX));
            charge().expect("Fix: a nested evaluation charges the enclosing budget");
            assert_eq!(
                charged(),
                2,
                "Fix: a nested evaluation must not reset the enclosing count"
            );
            drop(inner);
            assert_eq!(
                charged(),
                2,
                "Fix: dropping a nested guard must not disarm the enclosing budget"
            );
        }
        for _ in 0..8 {
            charge().expect("Fix: the enclosing ceiling of ten admits ten steps in total");
        }
        charge().expect_err("Fix: a nested evaluation cannot raise the enclosing ceiling");
        drop(outer);
    }

    #[test]
    fn an_unnamed_program_is_labelled_by_its_fingerprint_prefix() {
        let program = tiny_program();
        assert!(program.entry_op_id.is_none());
        let label = program_label(&program);
        assert!(
            label.starts_with("0x") && label.len() == 18,
            "Fix: an unnamed program is labelled by a fingerprint prefix, got {label}"
        );
        assert_eq!(
            label,
            program_label(&tiny_program()),
            "Fix: the label is a function of the program, so two equal programs share it"
        );
    }

    /// WHY: a constant loop bound is what the refusal tells a caller to write,
    /// so the walker has to read one and refuse to guess at anything else. A
    /// walker that folded, or that treated an unknown bound as one iteration,
    /// would report a bound smaller than the run and reintroduce the refusal it
    /// exists to prevent.
    #[test]
    fn declared_work_is_counted_only_where_every_trip_count_is_a_literal() {
        let leaf = vec![Node::store("out", Expr::u32(0), Expr::u32(7))];
        assert_eq!(
            declared_statements(&leaf),
            Some(1),
            "Fix: one statement charges one step"
        );

        let constant_loop = vec![Node::loop_("i", Expr::u32(0), Expr::u32(10), leaf.clone())];
        assert_eq!(
            declared_statements(&constant_loop),
            Some(1 + 10 * (1 + 1)),
            "Fix: a loop charges its own step plus one boundary and one body step per iteration"
        );

        let offset_loop = vec![Node::loop_("i", Expr::u32(4), Expr::u32(10), leaf.clone())];
        assert_eq!(
            declared_statements(&offset_loop),
            Some(1 + 6 * (1 + 1)),
            "Fix: the trip count is the difference of the declared bounds, not the upper bound"
        );

        let both_arms = vec![Node::if_then_else(
            Expr::bool(true),
            leaf.clone(),
            vec![
                Node::store("out", Expr::u32(0), Expr::u32(1)),
                Node::store("out", Expr::u32(1), Expr::u32(2)),
            ],
        )];
        assert_eq!(
            declared_statements(&both_arms),
            Some(1 + 1 + 2),
            "Fix: a bound over a conditional counts both arms, because the data picks the arm"
        );

        let data_derived = vec![Node::loop_(
            "i",
            Expr::u32(0),
            Expr::load("bound", Expr::u32(0)),
            leaf.clone(),
        )];
        assert_eq!(
            declared_statements(&data_derived),
            None,
            "Fix: a trip count read from a buffer declares no work, so the floor must stand"
        );

        let mixed = vec![
            Node::loop_("i", Expr::u32(0), Expr::u32(10), leaf.clone()),
            Node::loop_("j", Expr::u32(0), Expr::load("bound", Expr::u32(0)), leaf),
        ];
        assert_eq!(
            declared_statements(&mixed),
            None,
            "Fix: one undeclared trip count leaves the whole body's work undeclared"
        );
    }

    /// WHY: the ceiling used to be one fixed number, and a 256-square dense
    /// contraction that declares every extent as a constant was refused by it
    /// after 95 seconds of legitimate work. Admission raises the ceiling only
    /// for work the program declares, and this pins both halves: a declared
    /// bound above the floor raises it, and an undeclared one does not.
    #[test]
    fn admission_raises_the_ceiling_for_declared_work_and_leaves_it_for_data_derived_work() {
        let program = tiny_program();
        let leaf = vec![Node::store("out", Expr::u32(0), Expr::u32(7))];

        let guard = arm(&program);
        admit_declared_work(&leaf, 1);
        let (_, ceiling, _) = BUDGET
            .with(Cell::get)
            .expect("Fix: the budget stays armed through admission");
        assert_eq!(
            ceiling, MAX_REFERENCE_STEPS,
            "Fix: declared work below the floor must not lower the ceiling"
        );

        // One store per invocation over more invocations than the floor admits:
        // the smallest shape whose declared work exceeds it.
        let invocations = MAX_REFERENCE_STEPS + 1;
        admit_declared_work(&leaf, invocations);
        let (_, raised, _) = BUDGET
            .with(Cell::get)
            .expect("Fix: the budget stays armed through admission");
        assert_eq!(
            raised,
            (invocations + 1) * DECLARED_WORK_HEADROOM,
            "Fix: admission raises the ceiling to the declared work plus its headroom"
        );

        let data_derived = vec![Node::loop_(
            "i",
            Expr::u32(0),
            Expr::load("bound", Expr::u32(0)),
            leaf,
        )];
        admit_declared_work(&data_derived, u64::MAX);
        let (_, unchanged, _) = BUDGET
            .with(Cell::get)
            .expect("Fix: the budget stays armed through admission");
        assert_eq!(
            unchanged, raised,
            "Fix: a body with a data-derived trip count declares nothing, so it must not raise the ceiling"
        );
        drop(guard);
    }

    /// WHY: admission is a no-op without an armed budget, so an evaluator driven
    /// directly by a unit test cannot arm one by accident.
    #[test]
    fn admission_without_an_armed_budget_arms_nothing() {
        admit_declared_work(&[Node::store("out", Expr::u32(0), Expr::u32(7))], u64::MAX);
        assert!(
            BUDGET.with(Cell::get).is_none(),
            "Fix: admission must not arm a budget nobody armed"
        );
    }
}
