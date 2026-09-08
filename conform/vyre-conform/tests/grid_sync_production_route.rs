//! Whole-grid fences on the hardware-free production route.
//!
//! WHY: a fenced program takes a compile path no unfenced program takes. When
//! the target reports no cooperative launch, the compiler cuts the fenced node
//! into ordered segments joined by a retained carrier, and the artifact then
//! carries resources the source graph never declared. Two defects lived in that
//! path and neither was reachable without a fence.
//!
//! The first is identity. `GraphValueId` and `ArtifactValueId` are separate
//! numberings, and the cut inserts a carrier ahead of the node's own outputs, so
//! every value after the insertion shifts by one. The executor read the graph
//! number as the artifact number and returned the carrier's bytes under the
//! output's name, which is a wrong answer rather than a rejection.
//!
//! The second is capability. The reference interpreter runs the whole grid
//! through one inter-fence segment before the next over one shared memory, which
//! is what a cooperative launch buys, and it reported otherwise. The cut then
//! handed segment state across a boundary that a one-shot host submission cannot
//! carry, and the later segment read a zeroed carrier.
//!
//! Neither is caught by an unfenced case, so this runs the fenced registered ops
//! through the same `ProductionSession` a release conformance run uses and
//! compares against the reference oracle. What it does not catch: a fence cut on
//! a target that does report cooperative launch, because such a target takes the
//! uncut path, and any divergence a device introduces after lowering.

use vyre::ir::Program;
use vyre_conform::production::ProductionSession;
use vyre_conform::witness_plan::{plan_witness_inputs_into, WitnessInputPlan};
use vyre_reference::value::Value;
use vyre_registry_link::backend::live_backend_registry;
use vyre_registry_link::operation::live_operation_registry;

/// The oracle backend, which executes a fence inside one dispatch.
const ORACLE_BACKEND_ID: &str = "cpu-ref";

/// One registered operation whose program carries a whole-grid fence, with the
/// fixture inputs its registration declares.
struct FencedCase {
    id: &'static str,
    program: Program,
    inputs: Vec<Vec<u8>>,
}

/// Every registered operation that builds a fenced program and declares fixture
/// inputs for it.
///
/// The roster is read off the registry at run time rather than listed here, so
/// an operation that grows a fence joins this suite without an edit and an
/// operation that loses one leaves it.
fn fenced_cases() -> Vec<FencedCase> {
    let mut cases = Vec::new();
    for entry in live_operation_registry().iter() {
        let (Some(build), Some(fixtures)) = (entry.build, entry.test_inputs) else {
            continue;
        };
        let program = build();
        if !vyre_megakernel::grid_sync::requires_grid_sync(&program) {
            continue;
        }
        let Some(inputs) = fixtures().into_iter().next() else {
            continue;
        };
        cases.push(FencedCase {
            id: entry.id,
            program,
            inputs,
        });
    }
    cases
}

/// Reference-oracle outputs for one fenced program under its planned inputs.
fn oracle_outputs(case: &FencedCase, planned: &[&[u8]]) -> Vec<Vec<u8>> {
    let values: Vec<Value> = planned.iter().map(|bytes| Value::from(*bytes)).collect();
    vyre_reference::reference_eval(&case.program, &values)
        .unwrap_or_else(|error| {
            panic!(
                "Fix: `{}` is registered with fixture inputs its own reference oracle refuses: {error}",
                case.id
            )
        })
        .iter()
        .map(Value::to_bytes)
        .collect()
}

/// WHY: the production route compiles, admits and submits an artifact, so it is
/// the only path that exercises the fence cut, the carrier the cut mints, and
/// the value renumbering the carrier causes. Comparing it against the oracle on
/// the same bytes is what separates a wrong answer from a rejection: both
/// defects this closes returned a completion the caller had no way to doubt.
#[test]
fn every_fenced_operation_matches_the_oracle_through_the_production_route() {
    let cases = fenced_cases();
    assert!(
        !cases.is_empty(),
        "Fix: no registered operation builds a whole-grid-fenced program with fixture inputs, so the fence cut and the value renumbering it causes are unproven. Restore a fenced operation or delete this suite and record what became unproven."
    );
    let registration = live_backend_registry()
        .expect("valid backend registry")
        .iter()
        .find(|registration| registration.id == ORACLE_BACKEND_ID)
        .unwrap_or_else(|| {
            panic!("Fix: the `{ORACLE_BACKEND_ID}` oracle backend must be linked into this test.")
        });

    for case in &cases {
        let plan = WitnessInputPlan::for_program(&case.program).unwrap_or_else(|error| {
            panic!("Fix: `{}` has no witness input plan: {error}", case.id)
        });
        let mut planned: Vec<&[u8]> = Vec::new();
        plan_witness_inputs_into(&case.inputs, &plan, &mut planned).unwrap_or_else(|error| {
            panic!(
                "Fix: `{}` fixture inputs do not satisfy its own witness plan: {error}",
                case.id
            )
        });
        let expected = oracle_outputs(case, &planned);
        let produced = ProductionSession::from_registration(&case.program, registration)
            .unwrap_or_else(|error| {
                panic!(
                    "Fix: `{}` must compile for the oracle backend: {error}",
                    case.id
                )
            })
            .submit(&planned)
            .unwrap_or_else(|error| {
                panic!(
                    "Fix: `{}` must execute through canonical artifact submission: {error}",
                    case.id
                )
            });
        assert_eq!(
            produced.outputs, expected,
            "Fix: `{}` answered differently through the production artifact route than through the reference oracle. A fenced program whose carrier is mis-identified or never carried returns the wrong buffer rather than failing.",
            case.id
        );
    }
}

/// WHY: the cut is chosen by one capability report, so a backend that satisfies
/// a fence inside one dispatch and reports otherwise sends every fenced program
/// down a path it does not need and cannot complete. The oracle is that backend:
/// `vyre_reference` partitions the body at each top-level fence and runs the
/// whole grid through one segment before the next, over one memory.
#[test]
fn the_oracle_backend_reports_the_whole_grid_fence_it_satisfies() {
    let registration = live_backend_registry()
        .expect("valid backend registry")
        .iter()
        .find(|registration| registration.id == ORACLE_BACKEND_ID)
        .unwrap_or_else(|| {
            panic!("Fix: the `{ORACLE_BACKEND_ID}` oracle backend must be linked into this test.")
        });
    let backend = (registration.factory)().unwrap_or_else(|error| {
        panic!("Fix: the `{ORACLE_BACKEND_ID}` backend must construct: {error}")
    });
    assert!(
        backend.supports_grid_sync(),
        "Fix: the reference interpreter executes a whole-grid fence inside one dispatch, so `{ORACLE_BACKEND_ID}` must report it. Reporting otherwise cuts every fenced program into segments that a one-shot host submission cannot join."
    );
}
