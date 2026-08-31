//! Conformance of a declared numeric contract against a measured result.
//!
//! WHY: BACKLOG row 65 requires conformance to cover adversarial magnitudes,
//! cancellation, long recurrences and reductions, special values, deterministic
//! and intentionally nondeterministic modes, and to fail each schedule variant
//! whose measured error exceeds the declared complete-graph contract. A
//! contract nothing measures is a comment; these cases measure one.
//!
//! Each case builds the same reduction or recurrence in several combine orders,
//! evaluates every order through the parity oracle, and compares the distance
//! from a binary64 oracle against the contract
//! `vyre_foundation::numeric` derives for that order. The orders are the ones a
//! schedule may select: the stated sequential chain, a balanced tree, and a
//! chunked chain. A variant measured outside its declared budget is a failure,
//! and one case proves the comparison can produce that failure.
//!
//! What these cases do not prove: what a device computes. The oracle runs the
//! reference interpreter, so a divergence between a backend and this bound is
//! `prove`'s finding. What is proven here is that the declared bound is one a
//! correct implementation of the stated order can meet, and that a reordered
//! schedule priced against the tighter budget is rejected rather than absorbed.

#![forbid(unsafe_code)]

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::numeric::{
    AtomicOrderSensitivity, ContractRefusal, Determinism, ErrorMeasure, NumericContract,
    Reassociation, ScalarFormat,
};
use vyre_reference::value::Value;
use vyre_spec::SubnormalBehavior;

/// Region identity of the fixture programs.
const OP_ID: &str = "vyre-conform::numeric_contract_conformance";

/// Terms in every reduction case.
const TERMS: u32 = 1024;

/// Steps in the recurrence case.
const STEPS: u32 = 256;

/// Damping factor of the recurrence, chosen so the state stays bounded.
const DAMPING: f32 = 0.5;

/// An order a schedule may combine a reduction in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    /// The chain the program states, left to right.
    Stated,
    /// A balanced binary tree over the same terms.
    Tree,
    /// Sequential within chunks of the given width, then over the chunk sums.
    Chunked(u32),
}

impl Order {
    /// How the contract reads this order.
    fn reassociation(self) -> Reassociation {
        match self {
            Self::Stated => Reassociation::Forbidden,
            Self::Tree | Self::Chunked(_) => Reassociation::WithinBudget,
        }
    }

    /// A name a failure can name the variant by.
    fn label(self) -> String {
        match self {
            Self::Stated => "stated".to_owned(),
            Self::Tree => "tree".to_owned(),
            Self::Chunked(width) => format!("chunked-{width}"),
        }
    }
}

/// Every order a schedule may select for a reduction of [`TERMS`] terms.
fn orders() -> Vec<Order> {
    vec![Order::Stated, Order::Tree, Order::Chunked(32)]
}

/// The buffers every fixture program declares.
fn buffers(terms: u32) -> Vec<BufferDecl> {
    vec![
        BufferDecl::storage("values", 0, BufferAccess::ReadOnly, DataType::F32).with_count(terms),
        BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::F32).with_count(1),
    ]
}

/// Nodes that sum `terms` values into `acc` in the stated left-to-right chain.
fn stated_chain(terms: u32) -> Vec<Node> {
    let mut nodes = vec![Node::let_bind("acc", Expr::load("values", Expr::u32(0)))];
    for index in 1..terms {
        nodes.push(Node::assign(
            "acc",
            Expr::add(Expr::var("acc"), Expr::load("values", Expr::u32(index))),
        ));
    }
    nodes
}

/// Nodes that sum `terms` values into `acc` as a balanced binary tree.
///
/// Every level binds its partial sums to their own names, so the expression
/// depth stays at one add however long the reduction is.
fn tree_chain(terms: u32) -> Vec<Node> {
    let mut nodes = Vec::new();
    let mut level: Vec<Expr> = (0..terms)
        .map(|index| Expr::load("values", Expr::u32(index)))
        .collect();
    let mut generation = 0u32;
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for (position, pair) in level.chunks(2).enumerate() {
            match pair {
                [left, right] => {
                    let name = format!("t{generation}_{position}");
                    nodes.push(Node::let_bind(
                        name.as_str(),
                        Expr::add(left.clone(), right.clone()),
                    ));
                    next.push(Expr::var(name.as_str()));
                }
                [single] => next.push(single.clone()),
                _ => unreachable!("chunks(2) yields one or two elements"),
            }
        }
        level = next;
        generation += 1;
    }
    nodes.push(Node::let_bind("acc", level.remove(0)));
    nodes
}

/// Nodes that sum `terms` values into `acc` chunk by chunk.
fn chunked_chain(terms: u32, width: u32) -> Vec<Node> {
    let mut nodes = Vec::new();
    let mut chunk_sums = Vec::new();
    let mut start = 0;
    while start < terms {
        let end = (start + width).min(terms);
        let name = format!("c{start}");
        nodes.push(Node::let_bind(
            name.as_str(),
            Expr::load("values", Expr::u32(start)),
        ));
        for index in start + 1..end {
            nodes.push(Node::assign(
                name.as_str(),
                Expr::add(
                    Expr::var(name.as_str()),
                    Expr::load("values", Expr::u32(index)),
                ),
            ));
        }
        chunk_sums.push(name);
        start = end;
    }
    let mut sums = chunk_sums.into_iter();
    let first = sums.next().expect("a reduction has at least one chunk");
    nodes.push(Node::let_bind("acc", Expr::var(first.as_str())));
    for name in sums {
        nodes.push(Node::assign(
            "acc",
            Expr::add(Expr::var("acc"), Expr::var(name.as_str())),
        ));
    }
    nodes
}

/// The program that reduces `terms` binary32 values in `order`.
fn reduction_program(order: Order, terms: u32) -> Program {
    let mut body = match order {
        Order::Stated => stated_chain(terms),
        Order::Tree => tree_chain(terms),
        Order::Chunked(width) => chunked_chain(terms, width),
    };
    body.push(Node::store("out", Expr::u32(0), Expr::var("acc")));
    Program::wrapped(
        buffers(terms),
        [1, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    )
}

/// The program that advances `state = state * DAMPING + values[i]` `steps`
/// times.
///
/// A recurrence has one chain and no order to select, which is why it is
/// measured against a compounded contract rather than a reduction one.
fn recurrence_program(steps: u32) -> Program {
    let mut body = vec![Node::let_bind("acc", Expr::load("values", Expr::u32(0)))];
    for index in 1..steps {
        body.push(Node::assign(
            "acc",
            Expr::add(
                Expr::mul(Expr::var("acc"), Expr::f32(DAMPING)),
                Expr::load("values", Expr::u32(index)),
            ),
        ));
    }
    body.push(Node::store("out", Expr::u32(0), Expr::var("acc")));
    Program::wrapped(
        buffers(steps),
        [1, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    )
}

/// Run `program` over `values` through the parity oracle.
fn evaluate(program: &Program, values: &[f32]) -> f32 {
    let payload = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<u8>>();
    let outputs =
        vyre_reference::reference_eval(program, &[Value::from(payload), Value::from(vec![0u8; 4])])
            .expect("Fix: the reference oracle must execute the numeric fixture program");
    let scalar = outputs
        .iter()
        .map(vyre_reference::value::Value::to_bytes)
        .find(|bytes| bytes.len() == 4)
        .expect("Fix: the fixture declares one single-element output buffer");
    f32::from_le_bytes([scalar[0], scalar[1], scalar[2], scalar[3]])
}

/// The binary64 value the reduction states, computed in the same order.
///
/// Binary64 holds 29 more mantissa bits than binary32, so over these term
/// counts its own rounding is below the last bit of the binary32 answer and the
/// distance measured against it is the binary32 error.
fn oracle_sum(values: &[f32]) -> f64 {
    values.iter().map(|value| f64::from(*value)).sum()
}

/// The binary64 value the recurrence states.
fn oracle_recurrence(values: &[f32]) -> f64 {
    let mut state = f64::from(values[0]);
    for value in &values[1..] {
        state = state * f64::from(DAMPING) + f64::from(*value);
    }
    state
}

/// The relative distance between a measured result and the exact one.
fn relative_distance(measured: f32, exact: f64) -> f64 {
    ((f64::from(measured) - exact) / exact).abs()
}

/// The contract a reduction of `terms` terms in `order` declares.
///
/// One unit in the last place per combine is the rounding a correctly rounded
/// binary32 add produces, doubled from the half unit it actually costs so the
/// bound is one an implementation can meet rather than one it must be lucky to.
fn declared_reduction(order: Order, terms: u32) -> NumericContract {
    NumericContract::of(ScalarFormat::F32)
        .within_ulp(1)
        .reassociating(order.reassociation())
        .over_reduction(terms)
        .expect("Fix: a binary32 reduction states a readable bound")
}

/// The contract a recurrence of `steps` steps declares.
///
/// The stated contract admits reordering, which a reduction of the same length
/// would be entitled to. Advancing it over a chain is what takes that back: one
/// chain has one order.
fn declared_recurrence(steps: u32) -> NumericContract {
    NumericContract::of(ScalarFormat::F32)
        .within_ulp(1)
        .reassociating(Reassociation::WithinBudget)
        .over_recurrence(steps)
        .expect("Fix: a binary32 recurrence states a readable bound")
}

/// Whether the order measured against `declared` stays inside it.
///
/// This is the judgement the row requires: a schedule variant whose measured
/// error exceeds the declared complete-graph contract is refused, whatever the
/// variant was selected for.
fn verdict(declared: &NumericContract, measured: f64) -> Result<(), ContractRefusal> {
    declared.admits(&ErrorMeasure::relative(measured))
}

/// The adversarial inputs every order is measured over.
///
/// Each set targets one way a floating reduction loses digits: terms spanning
/// many binades, a large term that cancels and leaves the small ones carrying
/// the answer, a long run of equal terms whose sum outgrows one term's last
/// bit, and a descending run where a sequential chain adds terms below its own
/// accumulator's resolution.
fn adversarial_reductions() -> Vec<(&'static str, Vec<f32>)> {
    let count = TERMS as usize;
    let mixed_magnitudes = (0..count)
        .map(|index| {
            let exponent = i32::try_from(index % 40).unwrap_or(0) - 20;
            (2.0f32).powi(exponent) * if index % 3 == 0 { -1.0 } else { 1.0 }
        })
        .collect();
    let mut cancellation = vec![1.0e7f32, -1.0e7f32];
    cancellation.extend((2..count).map(|index| if index % 2 == 0 { 1.5e-3 } else { -0.5e-3 }));
    let equal_terms = vec![0.1f32; count];
    let mut descending = vec![1.0f32];
    descending.extend((1..count).map(|index| 1.0e-7f32 / (index as f32)));
    vec![
        ("mixed magnitudes", mixed_magnitudes),
        ("cancellation then small terms", cancellation),
        ("long run of equal terms", equal_terms),
        ("descending magnitudes", descending),
    ]
}

/// WHY: the bound a contract declares is worthless if no implementation of the
/// stated order can meet it, and dangerous if it is so wide that a wrong result
/// fits inside. Measuring every admitted order over adversarial inputs is what
/// separates a declared budget from a guess.
#[test]
fn every_admitted_order_stays_inside_the_budget_its_contract_declares() {
    for (name, values) in adversarial_reductions() {
        let exact = oracle_sum(&values);
        assert!(
            exact.abs() > 0.0,
            "Fix: `{name}` must have a nonzero exact sum to be read as a relative error"
        );
        for order in orders() {
            let measured =
                relative_distance(evaluate(&reduction_program(order, TERMS), &values), exact);
            let declared = declared_reduction(order, TERMS);
            verdict(&declared, measured).unwrap_or_else(|refusal| {
                panic!(
                    "Fix: the `{}` order over `{name}` measured {measured:e}, outside the contract it declares: {refusal}",
                    order.label()
                )
            });
        }
    }
}

/// WHY: the comparison must be able to fail. A conformance case that admits
/// every variant certifies nothing, and the reordering budget row 65 opens is
/// only safe if pricing a reordered schedule against the tighter budget refuses
/// it. The stated chain over descending magnitudes is the classic loss, and the
/// tree budget is the one a reordering schedule would inherit if the pricing
/// were skipped.
#[test]
fn a_variant_measured_outside_the_declared_budget_is_refused() {
    let values = adversarial_reductions()
        .into_iter()
        .find(|(name, _)| *name == "long run of equal terms")
        .map(|(_, values)| values)
        .expect("Fix: the adversarial set must carry the equal-terms case");
    let exact = oracle_sum(&values);
    let measured = relative_distance(
        evaluate(&reduction_program(Order::Stated, TERMS), &values),
        exact,
    );
    let tree_budget = declared_reduction(Order::Tree, TERMS);
    let refusal = verdict(&tree_budget, measured)
        .expect_err("Fix: a sequential chain does not fit a tree budget over 1024 equal terms");
    assert!(
        matches!(refusal, ContractRefusal::BudgetExceeded { .. }),
        "Fix: an over-budget measurement is a budget refusal, not {refusal}"
    );
    verdict(&declared_reduction(Order::Stated, TERMS), measured)
        .expect("Fix: the same measurement fits the budget the stated order declares");
}

/// WHY: a recurrence compounds where a reduction adds, so pricing one as the
/// other understates the error of every long chain. The case measures a real
/// chain against the compounded contract and proves the compounded bound is the
/// wider of the two over the same length.
#[test]
fn a_long_recurrence_stays_inside_its_compounded_contract() {
    let values = (0..STEPS as usize)
        .map(|index| if index % 2 == 0 { 1.0e3 } else { -0.75e3 })
        .collect::<Vec<f32>>();
    let exact = oracle_recurrence(&values);
    let measured = relative_distance(evaluate(&recurrence_program(STEPS), &values), exact);
    let declared = declared_recurrence(STEPS);
    verdict(&declared, measured).unwrap_or_else(|refusal| {
        panic!("Fix: the recurrence measured {measured:e}, outside its contract: {refusal}")
    });
    assert_eq!(
        declared.reassociation,
        Reassociation::Forbidden,
        "Fix: one chain has one order, whatever the contract stated"
    );
    let reduction = declared_reduction(Order::Stated, STEPS)
        .relative_error()
        .expect("Fix: a binary32 reduction bound is readable");
    assert!(
        declared
            .relative_error()
            .expect("Fix: a compounded bound is readable")
            > reduction,
        "Fix: a compounded chain is priced above a reduction of the same length"
    );
}

/// WHY: cancellation leaves an exact answer near zero, where a relative reading
/// divides by nothing. The contract states that as an absolute bound, and a
/// case that read it relatively would report a correct result as an infinite
/// error.
#[test]
fn a_sum_that_cancels_to_zero_is_held_to_an_absolute_bound() {
    let mut values = Vec::with_capacity(TERMS as usize);
    for index in 0..TERMS as usize / 2 {
        let magnitude = 1.0e6f32 * (1.0 + index as f32);
        values.push(magnitude);
        values.push(-magnitude);
    }
    assert_eq!(oracle_sum(&values), 0.0, "the fixture cancels exactly");

    let largest = values
        .iter()
        .fold(0.0f32, |widest, value| widest.max(value.abs()));
    let step = ScalarFormat::F32
        .ulp_fraction()
        .expect("Fix: binary32 has a uniform unit in the last place");
    let declared = NumericContract::of(ScalarFormat::F32)
        .with_measure(ErrorMeasure::absolute(
            f64::from(largest) * step * f64::from(TERMS),
        ))
        .reassociating(Reassociation::WithinBudget);
    assert!(
        matches!(
            declared.relative_error(),
            Err(ContractRefusal::UnboundedMagnitude { .. })
        ),
        "Fix: an absolute bound has no relative reading without a magnitude proof"
    );

    for order in orders() {
        let measured = f64::from(evaluate(&reduction_program(order, TERMS), &values)).abs();
        assert!(
            measured <= declared.measure.magnitude(),
            "Fix: the `{}` order left {measured:e} of a sum that cancels to zero, above the \
             absolute bound {:e} the contract declares",
            order.label(),
            declared.measure.magnitude()
        );
    }
}

/// The contract the parity oracle path states.
///
/// The reference interpreter canonicalizes every binary32 operand and result:
/// subnormals become signed zero so a backend that flushes and one that does
/// not converge on the same bits. A contract stated over this path says so,
/// because the alternative is a bound measured against a value the path never
/// produces.
fn oracle_contract() -> NumericContract {
    NumericContract::of(ScalarFormat::F32).flushing_subnormals()
}

/// The result the declared subnormal behavior states for a sum of `count`
/// negative subnormal terms.
///
/// The match is exhaustive with no catch-all arm: a new behavior turns this
/// red until someone states what the oracle does with it.
fn expected_subnormal_sum(behavior: SubnormalBehavior, count: usize) -> f32 {
    match behavior {
        SubnormalBehavior::FlushedToSignedZero => -0.0,
        SubnormalBehavior::PreservedIEEE => -f32::from_bits(1) * count as f32,
        SubnormalBehavior::Unsupported => {
            panic!("Fix: binary32 has subnormals, so the contract cannot state it has none")
        }
    }
}

/// WHY: a contract states what happens to a NaN, an infinity and a subnormal,
/// and a reordered schedule that changed any of the three would compute a
/// different program. The propagation is the same for every order because the
/// contract is the same for every order, and the subnormal case is read from
/// the declared behavior rather than assumed, so a path that starts flushing or
/// stops flushing is a contract change and not a silent one.
#[test]
fn special_values_propagate_the_way_the_contract_states() {
    let count = TERMS as usize;
    let with_nan = {
        let mut values = vec![1.0f32; count];
        values[count / 2] = f32::NAN;
        values
    };
    let with_infinity = {
        let mut values = vec![1.0f32; count];
        values[7] = f32::INFINITY;
        values
    };
    let opposing_infinities = {
        let mut values = vec![1.0f32; count];
        values[7] = f32::INFINITY;
        values[9] = f32::NEG_INFINITY;
        values
    };
    let subnormals = vec![-f32::from_bits(1); count];
    let expected = expected_subnormal_sum(oracle_contract().subnormal, count);

    for order in orders() {
        let program = reduction_program(order, TERMS);
        let label = order.label();
        assert!(
            evaluate(&program, &with_nan).is_nan(),
            "Fix: the `{label}` order must carry a NaN to the output"
        );
        assert_eq!(
            evaluate(&program, &with_infinity),
            f32::INFINITY,
            "Fix: the `{label}` order must carry an infinity to the output"
        );
        assert!(
            evaluate(&program, &opposing_infinities).is_nan(),
            "Fix: the `{label}` order must produce a NaN from opposing infinities"
        );
        assert_eq!(
            evaluate(&program, &subnormals).to_bits(),
            expected.to_bits(),
            "Fix: the `{label}` order must handle subnormals the way the contract states, \
             including the sign of the zero it flushes to"
        );
    }
}

/// WHY: a deterministic contract promises the same bits on every run and gets
/// them by refusing every order but the stated one. A contract that promised
/// determinism and admitted reordering would promise something the schedule is
/// free to break.
#[test]
fn a_deterministic_contract_refuses_reordering_and_repeats_its_bits() {
    let values = adversarial_reductions()
        .into_iter()
        .next()
        .map(|(_, values)| values)
        .expect("Fix: the adversarial set is not empty");
    let program = reduction_program(Order::Stated, TERMS);
    let first = evaluate(&program, &values);
    let second = evaluate(&program, &values);
    assert_eq!(
        first.to_bits(),
        second.to_bits(),
        "Fix: one order over one input produces one result"
    );

    let deterministic = NumericContract::ieee_f32(1);
    assert_eq!(deterministic.determinism, Determinism::Deterministic);
    assert!(
        matches!(
            deterministic.permits_reassociation(),
            Err(ContractRefusal::ReassociationRefused { .. })
        ),
        "Fix: a deterministic binary32 contract states the order it was measured under"
    );
}

/// WHY: a contract that admits run-to-run variation is not a licence for any
/// answer. It bounds the spread between the orders the device may select, and
/// that bound is the widest admitted order, never the best one: the stated
/// chain stays admissible when reordering is allowed, so a contract priced at
/// the tree it hopes for understates what it promises. Both halves are
/// asserted, because a spread of zero would make the bound unmeasured.
#[test]
fn a_run_to_run_variable_contract_bounds_the_spread_between_admitted_orders() {
    let values = vec![0.1f32; TERMS as usize];
    let declared = NumericContract::of(ScalarFormat::F32)
        .within_ulp(1)
        .over_reduction(TERMS)
        .expect("Fix: a binary32 reduction states a readable bound")
        .reassociating(Reassociation::WithinBudget)
        .under(Determinism::RunToRunVariable)
        .sensitive_to(AtomicOrderSensitivity::Sensitive);

    let results = orders()
        .into_iter()
        .map(|order| {
            (
                order,
                f64::from(evaluate(&reduction_program(order, TERMS), &values)),
            )
        })
        .collect::<Vec<_>>();
    let widest = results
        .iter()
        .map(|(_, value)| *value)
        .fold(f64::NEG_INFINITY, f64::max);
    let narrowest = results
        .iter()
        .map(|(_, value)| *value)
        .fold(f64::INFINITY, f64::min);
    assert!(
        widest > narrowest,
        "Fix: the admitted orders must actually differ, or the bound is unmeasured"
    );
    let spread = (widest - narrowest) / oracle_sum(&values);
    verdict(&declared, spread).unwrap_or_else(|refusal| {
        panic!("Fix: the admitted orders spread {spread:e}, outside the declared bound: {refusal}")
    });

    let priced_at_its_best_order = declared_reduction(Order::Tree, TERMS);
    assert!(
        matches!(
            verdict(&priced_at_its_best_order, spread),
            Err(ContractRefusal::BudgetExceeded { .. })
        ),
        "Fix: a run-to-run variable contract priced at the order it hopes for must be refused, \
         or the pricing admits a spread it cannot meet"
    );

    let deterministic = declared.under(Determinism::Deterministic);
    assert_ne!(
        deterministic, declared,
        "Fix: determinism is part of what a contract states, not a comment on it"
    );
}
