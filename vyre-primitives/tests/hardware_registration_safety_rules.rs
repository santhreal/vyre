//! Every Category C hardware registration is subject to the registry safety rules.
//!
//! `registry_oob_clean` runs those rules over the whole registry, so a hardware
//! intrinsic is covered there only for as long as it carries a fixture: its loops skip
//! an unfixtured op, and a Category C op registered without one arrives with a witness
//! program nothing ever executes. `subgroup_ballot` and `subgroup_shuffle` shipped that
//! way in the opposite direction, fixtured but never compiled by the default feature
//! set, and each performed hundreds of out-of-bounds loads on its own declared-valid
//! fixture because the collective's operand was an unguarded buffer load, which every
//! lane of the subgroup evaluates rather than only the lanes a store guard admits.
//!
//! This gate enumerates the hardware registrations FROM THE REGISTRY at run time and
//! is therefore fail-by-default on a new member: a registration added without fixtures
//! turns it red at the coverage assertion, and one added with fixtures gets every rule
//! run over it. It adds the lane-order axis the subgroup defect hid behind: forward,
//! reversed AND rotated step orders, compared against each other and against the
//! registered `expected_output` oracle. Reversal alone is a symmetric permutation, so
//! an implementation that confuses lane identity with step position can be reversal
//! symmetric and still wrong; rotation separates the two.
#![cfg(feature = "hardware")]

mod common;

use common::overfire_grid;
use vyre_foundation::operation::SemanticOperation;
use vyre_reference::value::Value;

/// Rotations exercised on top of forward and reversed order. Offsets that land inside a
/// 32-lane subgroup, on its last lane, and past a 64-lane workgroup, so a lane window
/// resolved by step position lands on the wrong lanes in every direction.
const ROTATIONS: [u32; 4] = [1, 7, 31, 33];

fn fixture_cases(fixtures: vyre_foundation::operation::OperationFixtures) -> Vec<Vec<Value>> {
    fixtures()
        .into_iter()
        .map(|case| case.into_iter().map(Value::from).collect())
        .collect()
}

fn bytes(values: &[Value]) -> Vec<Vec<u8>> {
    values.iter().map(Value::to_bytes).collect()
}

/// Every hardware registration the registry reports, in registration order.
fn hardware_entries() -> Vec<SemanticOperation> {
    vyre_primitives::hardware::all_entries().collect()
}

/// A hardware registration without fixtures is a witness program nothing executes.
///
/// The safety rules below are driven by `test_inputs` / `expected_output`, so a
/// fixtureless registration would be enumerated and then silently skipped: exactly the
/// hole that lets a new intrinsic land with an unchecked witness program. Requiring the
/// fixtures makes the omission red instead of invisible.
#[test]
fn every_hardware_registration_carries_the_fixtures_the_safety_rules_need() {
    let entries = hardware_entries();
    let mut missing = Vec::new();
    for entry in &entries {
        let mut gaps = Vec::new();
        if entry.build.is_none() {
            gaps.push("build");
        }
        if entry.test_inputs.is_none() {
            gaps.push("test_inputs");
        }
        if entry.expected_output.is_none() {
            gaps.push("expected_output");
        }
        // A Category C intrinsic maps to one hardware instruction, so its oracle is a
        // byte identity. An ULP tolerance here would make the oracle comparison in the
        // rules gate approximate, which is how a wrong lane model with a near-miss
        // value survives.
        if entry.tolerance.f32_ulp != 0 {
            gaps.push("exact tolerance");
        }
        if !gaps.is_empty() {
            missing.push(format!("{} (missing {})", entry.id, gaps.join(", ")));
        }
    }

    assert!(
        !entries.is_empty(),
        "Fix: the hardware registry reported no Category C registrations. Enable the \
         `hardware` feature so the intrinsics submit their registrations; a gate that \
         enumerates an empty registry proves nothing."
    );
    assert!(
        missing.is_empty(),
        "Fix: {} Category C hardware registration(s) do not carry what every registry \
         safety rule is driven by, so their witness program is never executed by any \
         gate and can index out of bounds, race, or depend on lane step order \
         undetected. Give each one `build`, `test_inputs`, `expected_output` and an \
         exact tolerance. Offenders:\n{}",
        missing.len(),
        missing.join("\n")
    );
}

/// The registry safety rules, run over every hardware registration under the feature
/// set that compiles them, plus the lane-order axis.
///
/// Rules applied per fixture case:
///   1. zero out-of-bounds accesses at the natural (buffer-inferred) grid;
///   2. zero out-of-bounds accesses when the dispatch is over-fired by one workgroup;
///   3. output byte-identical under over-fire;
///   4. output byte-identical under reversed and rotated lane step order;
///   5. output byte-identical to the registered `expected_output` oracle in every one
///      of those lane orders.
///
/// Rules 1 and 2 are what `subgroup_ballot` (112 OOB loads) and `subgroup_shuffle`
/// (224) failed. Rules 4 and 5 are what a collective that resolves its peer lanes by
/// step position rather than by lane index fails.
#[test]
fn every_hardware_registration_passes_every_registry_safety_rule() {
    let entries = hardware_entries();
    let mut offenders = Vec::new();
    let mut ruled_ops = 0usize;
    let mut checked_cases = 0usize;

    for entry in &entries {
        let Some(inputs_fn) = entry.test_inputs else {
            // Reported by the coverage gate above; nothing to run here.
            continue;
        };
        let program = entry
            .program()
            .expect("Fix: a hardware registration must provide a neutral builder");
        let overfire = overfire_grid(&program);
        let expected_cases = entry.expected_output.map(fixture_cases);
        ruled_ops += 1;

        for (case_idx, values) in fixture_cases(inputs_fn).into_iter().enumerate() {
            let mut fail = |rule: &str, detail: String| {
                offenders.push(format!("{} (case {case_idx}) {rule}: {detail}", entry.id));
            };

            let (forward, report) =
                match vyre_reference::reference_eval_oob_report(&program, &values) {
                    Ok(pair) => pair,
                    Err(error) => {
                        fail("evaluates its own fixture", error.to_string());
                        continue;
                    }
                };
            checked_cases += 1;
            if report.total() > 0 {
                fail(
                    "is out-of-bounds clean",
                    format!(
                        "{} load(s), {} store(s), {} atomic(s) past a buffer on valid input",
                        report.oob_loads, report.oob_stores, report.oob_atomics
                    ),
                );
            }

            match vyre_reference::reference_eval_with_dispatch_oob_report(
                &program, &values, overfire,
            ) {
                Ok((overfired, over_report)) => {
                    if over_report.total() > 0 {
                        fail(
                            "is out-of-bounds clean under grid over-fire",
                            format!(
                                "grid>={overfire}: {} load(s), {} store(s), {} atomic(s)",
                                over_report.oob_loads,
                                over_report.oob_stores,
                                over_report.oob_atomics
                            ),
                        );
                    }
                    if bytes(&overfired) != bytes(&forward) {
                        fail(
                            "keeps its output under grid over-fire",
                            format!("grid>={overfire} changed the output bytes"),
                        );
                    }
                }
                Err(error) => fail("evaluates under grid over-fire", error.to_string()),
            }

            let mut orders: Vec<(String, Vec<Value>)> = Vec::new();
            match vyre_reference::reference_eval_lane_reversed(&program, &values) {
                Ok(reversed) => orders.push(("reversed lane order".to_string(), reversed)),
                Err(error) => fail("evaluates in reversed lane order", error.to_string()),
            }
            for by in ROTATIONS {
                match vyre_reference::reference_eval_lane_rotated(&program, &values, by) {
                    Ok(rotated) => orders.push((format!("lane order rotated by {by}"), rotated)),
                    Err(error) => fail(
                        &format!("evaluates in lane order rotated by {by}"),
                        error.to_string(),
                    ),
                }
            }
            for (label, permuted) in &orders {
                if bytes(permuted) != bytes(&forward) {
                    fail(
                        "keeps its output under a permuted lane step order",
                        format!(
                            "{label} disagrees with forward order; a collective that resolves \
                             its peers by step position instead of lane index, or a non-atomic \
                             cross-lane write-write race"
                        ),
                    );
                }
            }

            // Rule 5: agreement with the registered CPU oracle, in every lane order. An
            // interpreter and an oracle can drift together into the same wrong lane
            // model, so pin both against the fixture, not only against each other.
            let Some(expected) = expected_cases
                .as_ref()
                .and_then(|cases| cases.get(case_idx))
            else {
                continue;
            };
            let expected_bytes: Vec<Vec<u8>> = expected.iter().map(Value::to_bytes).collect();
            for (label, produced) in
                std::iter::once(&("forward lane order".to_string(), forward)).chain(orders.iter())
            {
                let produced_bytes = bytes(produced);
                if produced_bytes.len() != expected_bytes.len()
                    || produced_bytes
                        .iter()
                        .zip(expected_bytes.iter())
                        .any(|(got, want)| got != want)
                {
                    fail(
                        "matches its registered oracle",
                        format!("{label} disagrees with expected_output"),
                    );
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "Fix: {} registry safety rule violation(s) across {} hardware registration(s), \
         {checked_cases} fixture case(s) checked. A Category C witness program must \
         index only inside the buffers it declares, on its natural grid and one \
         workgroup past it, and must produce the same bytes in every lane step order: a \
         subgroup collective is defined over lane identity, so an operand or a peer \
         lookup resolved by physical step position is a defect in the operation, not a \
         scheduling artifact. Violations:\n{}",
        offenders.len(),
        entries.len(),
        offenders.join("\n")
    );
    assert_eq!(
        ruled_ops,
        entries.len(),
        "Fix: {} of {} hardware registration(s) were enumerated but never subjected to \
         the safety rules, so a Category C intrinsic can land with an unchecked witness \
         program. Give every registration fixtures.",
        entries.len() - ruled_ops,
        entries.len()
    );
    assert!(
        checked_cases > 0,
        "Fix: no hardware fixture case was executed, so this gate passed vacuously."
    );
}
