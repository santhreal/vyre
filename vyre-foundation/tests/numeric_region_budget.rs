//! The numeric contract a graph region states, and the budget its outputs carry.
//!
//! WHY: an error budget stated per operation answers nothing about a graph. A
//! caller asks whether the value it reads back is inside a bound, and that is
//! the composition of every region that fed it: a rounding pointwise stage, a
//! reduction whose step count depends on its extent, a recurrence whose error
//! compounds, and a narrowing conversion that rounds a second time. Each case
//! here fixes one of those contributions, so a derivation that prices a region
//! at zero fails instead of certifying a budget nothing proves.
//!
//! What these cases do not prove: what a device measures. The proof is carried
//! out on the IR before lowering; conformance measures the result.

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ValueLifetime,
};
use vyre_foundation::logical::LogicalProgramGraph;
use vyre_foundation::numeric::{
    budget_admits, graph_budget, region_contract, reordering_admitted, Approximation,
    ContractRefusal, Determinism, ErrorMeasure, NumericContract, Reassociation, RegionArithmetic,
    RegionNumericFacts, ScalarFormat,
};

use vyre_test_support::graph_values::typed_vector as vector;

/// One region that adds a constant to every `f32` point it reads.
fn pointwise_f32(count: u32) -> ProgramGraph {
    let input = vector(
        count,
        DataType::F32,
        BufferAccess::ReadOnly,
        ValueLifetime::Invocation,
    );
    let output = vector(
        count,
        DataType::F32,
        BufferAccess::WriteOnly,
        ValueLifetime::Output,
    );
    let mut graph = ProgramGraph::new();
    let source = graph
        .add_external_value("input", input.clone())
        .expect("fixture external value must be valid");
    graph
        .add_node(
            "scale",
            Program::wrapped(
                vec![
                    BufferDecl::read("input", 0, DataType::F32).with_count(count),
                    BufferDecl::output("output", 1, DataType::F32).with_count(count),
                ],
                [64, 1, 1],
                vec![Node::store(
                    "output",
                    Expr::gid_x(),
                    Expr::mul(Expr::load("input", Expr::gid_x()), Expr::f32(2.0)),
                )],
            ),
            vec![GraphInput {
                buffer: "input".into(),
                value: source,
                contract: input,
            }],
            vec![GraphOutput {
                buffer: "output".into(),
                name: "output".into(),
                contract: output,
                retained_successor_of: None,
            }],
        )
        .expect("fixture pointwise node must be valid");
    graph
}

fn logical_budget(graph: &ProgramGraph) -> NumericContract {
    let logical = LogicalProgramGraph::validate(graph, &BTreeMap::new())
        .expect("fixture logical stage must validate");
    let budgets = logical
        .output_budgets()
        .expect("outputs must state a budget");
    assert_eq!(
        budgets.len(),
        1,
        "the fixture states one caller-visible output"
    );
    budgets[0].1
}

#[test]
fn a_rounding_region_states_one_unit_in_the_last_place() {
    let contract = region_contract(&RegionNumericFacts {
        input: Some(ScalarFormat::F32),
        output: Some(ScalarFormat::F32),
        arithmetic: RegionArithmetic::Pointwise,
        atomics: false,
        reorderable: false,
    })
    .expect("a pointwise f32 region prices");
    assert_eq!(contract.measure, ErrorMeasure::Ulp { count: 1 });
    assert_eq!(contract.reassociation, Reassociation::Forbidden);
}

#[test]
fn an_exact_region_states_the_exact_contract() {
    let contract = region_contract(&RegionNumericFacts {
        input: Some(ScalarFormat::U32),
        output: Some(ScalarFormat::U32),
        arithmetic: RegionArithmetic::Pointwise,
        atomics: true,
        reorderable: true,
    })
    .expect("a pointwise u32 region prices");
    assert_eq!(contract.measure, ErrorMeasure::Exact);
    assert_eq!(contract.reassociation, Reassociation::Exact);
    assert_eq!(
        contract.determinism,
        Determinism::Deterministic,
        "an exact atomic combine cannot disagree between runs"
    );
}

/// WHY: a narrowing conversion rounds the result a second time. Pricing it at
/// one step states the error of the operation and drops the error of storing it.
#[test]
fn a_narrowing_region_prices_the_conversion_it_performs() {
    let narrowing = region_contract(&RegionNumericFacts {
        input: Some(ScalarFormat::F32),
        output: Some(ScalarFormat::F16),
        arithmetic: RegionArithmetic::Pointwise,
        atomics: false,
        reorderable: false,
    })
    .expect("a narrowing region prices");
    assert_eq!(narrowing.measure, ErrorMeasure::Ulp { count: 2 });
    assert_eq!(narrowing.storage, ScalarFormat::F16);
    assert_eq!(narrowing.intermediate, ScalarFormat::F32);
}

/// WHY: a rounding atomic combine lands in whatever order the device produces,
/// so two runs of the same schedule on the same input disagree. A schedule that
/// promises bit-reproducible output over one is promising what it cannot hold.
#[test]
fn a_rounding_atomic_region_is_run_to_run_variable() {
    let contract = region_contract(&RegionNumericFacts {
        input: Some(ScalarFormat::F32),
        output: Some(ScalarFormat::F32),
        arithmetic: RegionArithmetic::Pointwise,
        atomics: true,
        reorderable: true,
    })
    .expect("an atomic f32 region prices");
    assert_eq!(contract.determinism, Determinism::RunToRunVariable);
}

#[test]
fn a_longer_reduction_states_a_wider_bound() {
    let reduce = |terms| {
        region_contract(&RegionNumericFacts {
            input: Some(ScalarFormat::F32),
            output: Some(ScalarFormat::F32),
            arithmetic: RegionArithmetic::Reduction { terms },
            atomics: false,
            reorderable: false,
        })
        .expect("an f32 reduction prices")
        .measure
    };
    assert_eq!(reduce(2), ErrorMeasure::Ulp { count: 1 });
    assert_eq!(reduce(1024), ErrorMeasure::Ulp { count: 1023 });
    assert!(
        reduce(4096).magnitude() > reduce(1024).magnitude(),
        "a reduction over four times as many terms rounds more often"
    );
}

/// WHY: a recurrence feeds its own output back in, so the error of one step is
/// the input of the next. Pricing it like a reduction would understate a long
/// chain by the difference between a sum and a product.
#[test]
fn a_recurrence_compounds_where_a_reduction_adds() {
    let steps = 4096;
    let recurrence = region_contract(&RegionNumericFacts {
        input: Some(ScalarFormat::F16),
        output: Some(ScalarFormat::F16),
        arithmetic: RegionArithmetic::Recurrence { steps },
        atomics: false,
        reorderable: false,
    })
    .expect("an f16 recurrence prices");
    let reduction = region_contract(&RegionNumericFacts {
        input: Some(ScalarFormat::F16),
        output: Some(ScalarFormat::F16),
        arithmetic: RegionArithmetic::Reduction { terms: steps },
        atomics: false,
        reorderable: false,
    })
    .expect("an f16 reduction prices");
    let compounded = recurrence
        .relative_error()
        .expect("a relative bound reads as a fraction");
    let added = reduction
        .relative_error()
        .expect("a ulp bound over f16 reads as a fraction");
    assert!(
        compounded > added,
        "compounding {compounded} must exceed adding {added}"
    );
    assert_eq!(
        recurrence.reassociation,
        Reassociation::Forbidden,
        "a chain has one order, so there is no other order to choose"
    );
}

/// WHY: a region that holds no number still appears in the chain. Refusing to
/// price it would make a graph carrying bytes unschedulable; pricing it at an
/// error would make every budget downstream of a copy wrong.
#[test]
fn a_region_holding_no_number_states_the_exact_contract() {
    let contract = region_contract(&RegionNumericFacts::opaque()).expect("an opaque region prices");
    assert_eq!(contract.measure, ErrorMeasure::Exact);
}

#[test]
fn a_graph_budget_composes_its_regions_in_order() {
    let stage = region_contract(&RegionNumericFacts {
        input: Some(ScalarFormat::F32),
        output: Some(ScalarFormat::F32),
        arithmetic: RegionArithmetic::Pointwise,
        atomics: false,
        reorderable: false,
    })
    .expect("a pointwise f32 region prices");
    let composed = graph_budget([&stage, &stage, &stage]).expect("three f32 stages compose");
    assert_eq!(
        composed.measure,
        ErrorMeasure::Ulp { count: 3 },
        "three rounding stages round three times"
    );
    assert_eq!(
        graph_budget(std::iter::empty()).expect("an empty graph composes"),
        NumericContract::EXACT
    );
}

/// WHY: composition is what makes a whole-graph budget mean anything, and it is
/// only defined where the second region reads what the first produced. A graph
/// that silently composed across a format it never converted would state a ULP
/// count in a format nothing holds.
#[test]
fn a_graph_budget_refuses_a_format_the_previous_region_does_not_produce() {
    let produces_f32 = NumericContract::of(ScalarFormat::F32).within_ulp(1);
    let reads_f16 = NumericContract::of(ScalarFormat::F16).within_ulp(1);
    assert_eq!(
        graph_budget([&produces_f32, &reads_f16]),
        Err(ContractRefusal::FormatMismatch {
            first: ScalarFormat::F32,
            second: ScalarFormat::F16,
        })
    );
}

#[test]
fn a_validated_graph_states_the_budget_its_output_carries() {
    let graph = pointwise_f32(256);
    let budget = logical_budget(&graph);
    assert_eq!(budget.storage, ScalarFormat::F32);
    assert_eq!(budget.measure, ErrorMeasure::Ulp { count: 1 });

    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("fixture logical stage must validate");
    assert_eq!(
        logical.regions()[0].numeric,
        budget,
        "one region is its own budget"
    );
}

/// WHY: a value the caller supplied has not been computed here, so nothing in
/// this graph has rounded it. Stating a rounding bound over an input would make
/// every budget that reads one wider than what the program does.
#[test]
fn an_unproduced_value_carries_the_exact_contract() {
    let graph = pointwise_f32(64);
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("fixture logical stage must validate");
    let external = graph
        .values()
        .iter()
        .find(|value| value.producer.is_none())
        .expect("the fixture supplies one external value");
    assert_eq!(
        logical
            .value_budget(external.id)
            .expect("an external value states a budget"),
        NumericContract::EXACT
    );
}

/// WHY: a declared budget that admits whatever a graph happens to prove is a
/// comment. Every stage that consumes a budget reads its answer through this
/// one, so a refusal that never fires would make an over-budget schedule legal
/// everywhere at once.
#[test]
fn a_declared_budget_refuses_a_graph_that_proves_more_error_than_it_states() {
    let declared = NumericContract::of(ScalarFormat::F32).within_ulp(4);
    let inside = NumericContract::of(ScalarFormat::F32).within_ulp(4);
    let outside = NumericContract::of(ScalarFormat::F32).within_ulp(5);
    assert_eq!(budget_admits(&declared, &inside), Ok(()));
    assert_eq!(
        budget_admits(&declared, &outside),
        Err(ContractRefusal::BudgetExceeded {
            declared: ErrorMeasure::Ulp { count: 4 },
            composed: ErrorMeasure::Ulp { count: 5 },
        })
    );
    assert_eq!(
        budget_admits(
            &declared,
            &NumericContract::of(ScalarFormat::F16).within_ulp(1)
        ),
        Err(ContractRefusal::FormatMismatch {
            first: ScalarFormat::F32,
            second: ScalarFormat::F16,
        }),
        "a count of units in one format is not a count in another"
    );
    let approximating = NumericContract::of(ScalarFormat::F32)
        .within_ulp(1)
        .approximating(Approximation::Native {
            measure: ErrorMeasure::Ulp { count: 9 },
        });
    assert!(
        budget_admits(&declared, &approximating).is_err(),
        "an approximate instruction contributes its own error to what the graph proves"
    );
}

/// WHY: whether a region's combines may be reordered is the fact that decides
/// which schedules reach it, and it is priced differently in each answer: a
/// stated order rounds once per term, a reordered one once per level. A region
/// contract that ignored the fact would price every reduction as the order it
/// was not selected under.
#[test]
fn a_reorderable_region_states_the_order_it_permits_and_is_priced_for_it() {
    let reduce = |reorderable| {
        region_contract(&RegionNumericFacts {
            input: Some(ScalarFormat::F32),
            output: Some(ScalarFormat::F32),
            arithmetic: RegionArithmetic::Reduction { terms: 1024 },
            atomics: false,
            reorderable,
        })
        .expect("an f32 reduction prices")
    };
    let stated = reduce(false);
    let reorderable = reduce(true);
    assert_eq!(stated.reassociation, Reassociation::Forbidden);
    assert_eq!(reorderable.reassociation, Reassociation::WithinBudget);
    assert_eq!(stated.measure, ErrorMeasure::Ulp { count: 1023 });
    assert_eq!(
        reorderable.measure,
        ErrorMeasure::Ulp { count: 10 },
        "a tree over 1024 terms rounds once per level"
    );
}

/// WHY: the reordering answer every stage reads is one function, so the search
/// and the route selection cannot drift apart. A budget wide enough for the new
/// order admits it, a budget narrower than one rounding step does not, and an
/// exact region needs no budget at all.
#[test]
fn a_reordering_is_admitted_only_where_the_stated_budget_covers_it() {
    let region = NumericContract::of(ScalarFormat::F32).within_ulp(1);
    assert!(reordering_admitted(
        &NumericContract::of(ScalarFormat::F32).within_ulp(16),
        &region,
        1024
    ));
    assert!(
        !reordering_admitted(
            &NumericContract::of(ScalarFormat::F32).within_ulp(4),
            &region,
            1024
        ),
        "ten levels of rounding do not fit four units in the last place"
    );
    assert!(
        !reordering_admitted(&NumericContract::of(ScalarFormat::F32), &region, 1024),
        "a budget of zero admits no new order"
    );
    assert!(
        reordering_admitted(&NumericContract::EXACT, &NumericContract::EXACT, u32::MAX),
        "an exact combine is the same value in every order"
    );
}
