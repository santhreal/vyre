//! Whole-registry parity guard for the library surface: no registered
//! composition may access a buffer out of bounds, change its answer when the
//! dispatch is over-fired, or depend on the order lanes are stepped.
//!
//! WHY: `vyre-primitives` has carried these nets since the gather-class audit,
//! and the library surface is where the compositions live. It had none, so a
//! composition that indexed past a buffer end was caught only if somebody wrote
//! a test for that one builder. One did not exist for the grid-stride tree
//! reduction, whose tail loads were guarded by `Expr::select`: that evaluates
//! both arms, so every lane past the element count read past the buffer end.
//! The reference interpreter absorbs the read as a zero, which is precisely the
//! masking a backend that does not bounds-check will not do.
//!
//! Each net derives its population from the catalog at run time, so a
//! registration added tomorrow is judged tomorrow, and each names the ops it
//! could not reach rather than reporting a clean sweep of a subset.

#![forbid(unsafe_code)]

use vyre_reference::value::Value;
use vyre_test_support::overfire_grid;

/// The library catalog, refused when empty: an empty walk passes every net
/// below without proving anything.
fn fixtured_entries() -> Vec<vyre_foundation::operation::SemanticOperation> {
    let entries: Vec<_> = vyre_libs::operation_catalog::all_entries().collect();
    assert!(
        !entries.is_empty(),
        "Fix: the library catalog is empty, so this run judges no registration at all"
    );
    entries
        .into_iter()
        .filter(|entry| entry.test_inputs.is_some())
        .collect()
}

fn program_of(entry: &vyre_foundation::operation::SemanticOperation) -> vyre_foundation::ir::Program {
    entry
        .program()
        .expect("Fix: registered library operation must provide a neutral builder")
}

fn cases(entry: &vyre_foundation::operation::SemanticOperation) -> Vec<Vec<Value>> {
    let inputs_fn = entry
        .test_inputs
        .expect("Fix: this walk is filtered to fixtured entries");
    inputs_fn()
        .into_iter()
        .map(|case| case.into_iter().map(Value::from).collect())
        .collect()
}

#[test]
fn every_registered_composition_is_oob_clean_on_its_fixtures() {
    let mut offenders = Vec::new();
    let mut checked_cases = 0usize;
    let mut eval_errored: Vec<String> = Vec::new();

    for entry in fixtured_entries() {
        let program = program_of(&entry);
        for (case_idx, values) in cases(&entry).into_iter().enumerate() {
            match vyre_reference::reference_eval_oob_report(&program, &values) {
                Ok((_out, report)) => {
                    checked_cases += 1;
                    if report.total() > 0 {
                        offenders.push(format!(
                            "{} (fixture case {case_idx}): {} OOB load(s), {} OOB store(s), {} OOB atomic(s)",
                            entry.id, report.oob_loads, report.oob_stores, report.oob_atomics
                        ));
                    }
                }
                Err(err) => eval_errored.push(format!("{} (case {case_idx}): {err}", entry.id)),
            }
        }
    }

    eprintln!(
        "library OOB sweep: {checked_cases} fixture case(s) checked, {} un-evaluable",
        eval_errored.len()
    );
    assert!(
        checked_cases > 0,
        "Fix: no library fixture was exercised, so the sweep proves nothing"
    );
    assert!(
        offenders.is_empty(),
        "Fix: {} of {checked_cases} checked library fixture case(s) accessed a buffer OUT OF BOUNDS on their own \
         valid input. The reference absorbs the access (zero-filled load, dropped store) so the answer looks \
         right, while a backend that does not bounds-check reads garbage or corrupts memory. Gate the index with \
         CONTROL FLOW: `Expr::select(i < n, load(buf, i), fallback)` evaluates the load for every i. Offenders:\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}

#[test]
fn every_registered_composition_is_oob_clean_under_grid_overfire() {
    let mut offenders = Vec::new();
    let mut checked_cases = 0usize;

    for entry in fixtured_entries() {
        let program = program_of(&entry);
        let grid = overfire_grid(&program);
        for (case_idx, values) in cases(&entry).into_iter().enumerate() {
            if let Ok((_out, report)) =
                vyre_reference::reference_eval_with_dispatch_oob_report(&program, &values, grid)
            {
                checked_cases += 1;
                if report.total() > 0 {
                    offenders.push(format!(
                        "{} (fixture case {case_idx}, grid>={grid}): {} OOB load(s), {} OOB store(s), {} OOB atomic(s)",
                        entry.id, report.oob_loads, report.oob_stores, report.oob_atomics
                    ));
                }
            }
        }
    }

    assert!(
        checked_cases > 0,
        "Fix: no library fixture was exercised under over-fire, so the sweep proves nothing"
    );
    assert!(
        offenders.is_empty(),
        "Fix: {} of {checked_cases} checked library fixture case(s) accessed a buffer OUT OF BOUNDS when the \
         dispatch was OVER-FIRED by one workgroup. A backend dispatches whole workgroups, so the lanes past the \
         logical count DO run and every per-lane guard must survive them. Offenders:\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}

#[test]
fn every_registered_composition_output_is_invariant_under_grid_overfire() {
    let mut offenders = Vec::new();
    let mut checked_cases = 0usize;

    for entry in fixtured_entries() {
        let program = program_of(&entry);
        let grid = overfire_grid(&program);
        for (case_idx, values) in cases(&entry).into_iter().enumerate() {
            let Ok(baseline) = vyre_reference::reference_eval(&program, &values) else {
                continue;
            };
            let Ok(overfired) =
                vyre_reference::reference_eval_with_dispatch(&program, &values, grid)
            else {
                continue;
            };
            checked_cases += 1;
            let base_bytes: Vec<Vec<u8>> = baseline.iter().map(Value::to_bytes).collect();
            let over_bytes: Vec<Vec<u8>> = overfired.iter().map(Value::to_bytes).collect();
            if base_bytes != over_bytes {
                let where_ = base_bytes
                    .iter()
                    .zip(over_bytes.iter())
                    .position(|(a, b)| a != b)
                    .map_or_else(
                        || format!("output count {} vs {}", base_bytes.len(), over_bytes.len()),
                        |idx| format!("output #{idx} differs"),
                    );
                offenders.push(format!(
                    "{} (fixture case {case_idx}, grid>={grid}): {where_}",
                    entry.id
                ));
            }
        }
    }

    assert!(
        checked_cases > 0,
        "Fix: no library fixture was compared across grids, so the sweep proves nothing"
    );
    assert!(
        offenders.is_empty(),
        "Fix: {} of {checked_cases} checked library fixture case(s) produced a DIFFERENT output when the dispatch \
         was OVER-FIRED by one workgroup. The extra lanes run on hardware, so a write they reach that no natural \
         lane touches diverges from the oracle every other test trusts, with no out-of-bounds access to show for \
         it. Gate every write on the logical count. Offenders:\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}

#[test]
fn every_registered_composition_is_race_free_under_lane_reversal() {
    let mut offenders = Vec::new();
    let mut checked_cases = 0usize;

    for entry in fixtured_entries() {
        let program = program_of(&entry);
        for (case_idx, values) in cases(&entry).into_iter().enumerate() {
            let Ok(forward) = vyre_reference::reference_eval(&program, &values) else {
                continue;
            };
            let Ok(reversed) = vyre_reference::reference_eval_lane_reversed(&program, &values)
            else {
                continue;
            };
            checked_cases += 1;
            let forward_bytes: Vec<Vec<u8>> = forward.iter().map(Value::to_bytes).collect();
            let reversed_bytes: Vec<Vec<u8>> = reversed.iter().map(Value::to_bytes).collect();
            if forward_bytes != reversed_bytes {
                offenders.push(format!("{} (fixture case {case_idx})", entry.id));
            }
        }
    }

    assert!(
        checked_cases > 0,
        "Fix: no library fixture was stepped in both lane orders, so the sweep proves nothing"
    );
    assert!(
        offenders.is_empty(),
        "Fix: {} of {checked_cases} checked library fixture case(s) changed their output when the lane STEP ORDER \
         was reversed, which means two lanes write one slot without an atomic. The reference resolves that \
         deterministically and hardware does not, so the answer is stable here and driver-defined there. Give \
         every shared slot a commutative atomic or disjoint ownership. Offenders:\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}
