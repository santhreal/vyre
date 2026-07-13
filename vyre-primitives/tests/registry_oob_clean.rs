//! Whole-registry parity guard: no registered Tier-2.5 primitive may access a
//! buffer out of bounds while running its own VALID fixture inputs.
//!
//! The reference interpreter silently absorbs OOB accesses (see
//! `vyre-reference/src/oob.rs`: zero-fill loads, dropped stores) so its output
//! stays deterministic, but that masking HIDES the gather-class GPU/CPU parity
//! hazard: an IR program with an ungated data-derived index "works" on the
//! reference yet a real GPU (CUDA does no bounds-checking) reads garbage or
//! corrupts memory. `reference_eval_oob_report` surfaces every absorbed access.
//!
//! This gate runs EVERY registered primitive against its own fixtures and asserts
//! zero OOB accesses, so a newly-added primitive that indexes past a buffer on
//! valid input (an off-by-one, or a data-derived index that is only in-bounds by
//! luck of the fixture) is caught the moment it lands, the standing net that
//! generalizes the manual gather-class audit (ziftsieve/base64/sketch/simplicial).
#![cfg(feature = "inventory-registry")]

use vyre_reference::value::Value;

/// The one-workgroup over-fire dispatch floor shared by the over-fire gates: the
/// largest declared buffer element count plus one whole workgroup of lanes, the
/// realistic worst case a whole-workgroup GPU dispatch produces past the logical
/// element count. ONE home so the two over-fire gates cannot drift.
fn overfire_grid(program: &vyre_foundation::ir::Program) -> u32 {
    let workgroup_lanes = program.workgroup_size()[0].max(1);
    let max_count = program
        .buffers()
        .iter()
        .map(vyre_foundation::ir::BufferDecl::count)
        .max()
        .unwrap_or(0);
    max_count.saturating_add(workgroup_lanes)
}

#[test]
fn every_registered_primitive_is_oob_clean_on_its_fixtures() {
    let mut offenders = Vec::new();
    let mut checked_cases = 0usize;
    let mut total_ops = 0usize;
    let mut fixtured_ops = 0usize;
    let mut eval_errored: Vec<String> = Vec::new();
    // Distinct from a fixture-shape error: a registered program that FAILS IR VALIDATION
    // (e.g. a duplicate-binding shadow) is a real defect the no-shadowing validator and the
    // CUDA backend both reject, it must FAIL this gate, not be skipped. (This caught the
    // union_find `uf_root_b` shadow that only string-matching inline tests had missed.)
    let mut invalid_programs: Vec<String> = Vec::new();

    for entry in vyre_primitives::harness::all_entries() {
        total_ops += 1;
        let Some(inputs_fn) = entry.test_inputs else {
            // No fixture ⇒ not exercised by this gate. Counted (not silently
            // dropped) so the coverage summary below is honest about what went
            // unchecked.
            continue;
        };
        fixtured_ops += 1;
        let program = (entry.build)();
        for (case_idx, case) in inputs_fn().into_iter().enumerate() {
            let values: Vec<Value> = case.into_iter().map(Value::from).collect();
            // A malformed fixture (wrong buffer count / under-length input) makes
            // reference_eval itself error; that is a fixture bug, not an OOB access,
            // and is surfaced by the dedicated conformance harness, count it so the
            // coverage summary stays honest, then skip the OOB check for that case.
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
                // NAME the erroring op (not just count it): a registered primitive
                // whose OWN fixture fails to reference-evaluate is a latent defect, a
                // malformed fixture that tests nothing, or a real interpreter-rejecting
                // program (and must be actionable, not an anonymous tally).
                Err(err) => {
                    let msg = format!("{} (case {case_idx}): {err}", entry.id);
                    if format!("{err}").contains("failed IR validation") {
                        invalid_programs.push(msg);
                    } else {
                        eval_errored.push(msg);
                    }
                }
            }
        }
    }

    // Coverage is surfaced, not silently partial: visible with `--nocapture` and in
    // the assertion messages. `total_ops - fixtured_ops` ops carry no fixture and are
    // OUT of this gate's reach (a candidate follow-up (add fixtures), not a false green).
    eprintln!(
        "registry OOB gate coverage: {fixtured_ops}/{total_ops} ops fixtured, \
         {checked_cases} case(s) checked, {} case(s) eval-errored (skipped), \
         {} unfixtured op(s) out of reach",
        eval_errored.len(),
        total_ops - fixtured_ops
    );
    if !eval_errored.is_empty() {
        eprintln!(
            "  eval-errored (named for follow-up):\n    {}",
            eval_errored.join("\n    ")
        );
    }

    assert!(
        invalid_programs.is_empty(),
        "Fix: {} registered primitive(s) build an IR-INVALID Program that fails validation on its own fixture \
         (e.g. a duplicate-binding shadow, which the no-shadowing IR validator AND the CUDA backend both reject). \
         The op cannot run on the reference OR the GPU, structural inline tests that only string-match the IR \
         dump do NOT catch this. A registered primitive MUST emit valid IR. Invalid programs:\n{}",
        invalid_programs.len(),
        invalid_programs.join("\n")
    );
    assert!(
        checked_cases > 0,
        "Fix: no registered primitive fixtures were exercised ({total_ops} ops seen, {fixtured_ops} fixtured). \
         enable the domain features (e.g. --features inventory-registry,all-lego) so ops register."
    );
    assert!(
        offenders.is_empty(),
        "Fix: {} of {checked_cases} checked registered-primitive fixture case(s) accessed a buffer OUT OF BOUNDS \
         on VALID input. The reference silently masks this (zero-fill/drop) but a real GPU would fault or corrupt \
         memory: it means an ungated data-derived index (the gather-class bug). Gate the index against its buffer \
         length (see ziftsieve/sketch/simplicial/ssa_dominance_scan/line_splice_classify fixes). Offenders:\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}

/// Every registered primitive MUST emit IR that passes validation (fixtured OR NOT).
/// The fixtured OOB gate above only validates ops that carry a fixture (its loop skips
/// unfixtured ops entirely), so an unfixtured registered op with an IR defect, e.g. a
/// duplicate-binding shadow, which the no-shadowing validator AND the CUDA backend both
/// reject, would land undetected. Validation runs BEFORE input binding in the
/// interpreter, so an IR-invalid program reports "failed IR validation" no matter what
/// inputs are supplied; a valid program on empty inputs reports a benign "missing input"
/// (ignored here). This closes the gap the union_find shadow bug exposed.
#[test]
fn every_registered_primitive_program_is_ir_valid() {
    let mut invalid = Vec::new();
    let mut total = 0usize;
    for entry in vyre_primitives::harness::all_entries() {
        total += 1;
        let program = (entry.build)();
        // Prefer the op's own (valid) fixture so a correct program eval-succeeds; fall back
        // to empty inputs for unfixtured ops (validation still runs first either way).
        let values: Vec<Value> = match entry.test_inputs {
            Some(inputs_fn) => inputs_fn()
                .into_iter()
                .next()
                .unwrap_or_default()
                .into_iter()
                .map(Value::from)
                .collect(),
            None => Vec::new(),
        };
        if let Err(err) = vyre_reference::reference_eval(&program, &values) {
            if format!("{err}").contains("failed IR validation") {
                invalid.push(format!("{}: {err}", entry.id));
            }
        }
    }
    assert!(
        total > 0,
        "Fix: no registered primitives seen (enable --features inventory-registry,all-lego)."
    );
    assert!(
        invalid.is_empty(),
        "Fix: {} registered primitive(s) emit IR that FAILS validation (the no-shadowing validator + the \
         CUDA backend both reject it, the op cannot run on the reference OR the GPU). A registered op MUST \
         emit valid IR whether or not it carries a fixture. Invalid:\n{}",
        invalid.len(),
        invalid.join("\n")
    );
}

/// Second standing net: the same registry sweep, but deliberately OVER-FIRES the
/// dispatch by one extra workgroup. Real GPUs dispatch whole workgroups, so lanes
/// beyond the logical element count DO run and must be guarded. A guard written as
/// `Expr::and(t < n, load(buf, t))` fails, the data-flow AND evaluates the load
/// for `t >= n`, an OOB read (the ssa_dominance_scan bug this gate generalizes).
/// Asserting zero OOB under over-fire catches that whole class registry-wide.
#[test]
fn every_registered_primitive_is_oob_clean_under_grid_overfire() {
    let mut offenders = Vec::new();
    let mut checked_cases = 0usize;

    for entry in vyre_primitives::harness::all_entries() {
        let Some(inputs_fn) = entry.test_inputs else {
            continue;
        };
        let program = (entry.build)();
        let overfire_grid = overfire_grid(&program);
        for (case_idx, case) in inputs_fn().into_iter().enumerate() {
            let values: Vec<Value> = case.into_iter().map(Value::from).collect();
            match vyre_reference::reference_eval_with_dispatch_oob_report(
                &program,
                &values,
                overfire_grid,
            ) {
                Ok((_out, report)) => {
                    checked_cases += 1;
                    if report.total() > 0 {
                        offenders.push(format!(
                            "{} (fixture case {case_idx}, grid>={overfire_grid}): {} OOB load(s), {} OOB store(s), {} OOB atomic(s)",
                            entry.id, report.oob_loads, report.oob_stores, report.oob_atomics
                        ));
                    }
                }
                Err(_) => {}
            }
        }
    }

    assert!(
        checked_cases > 0,
        "Fix: no registered primitive fixtures were exercised (enable domain features (e.g. --features inventory-registry,all-lego))."
    );
    assert!(
        offenders.is_empty(),
        "Fix: {} of {checked_cases} checked registered-primitive fixture case(s) accessed a buffer OUT OF BOUNDS \
         when the dispatch was OVER-FIRED by one workgroup. Whole-workgroup GPU dispatch runs lanes past the \
         logical count, so a per-lane guard must survive them, a guard using `Expr::and(t < n, load(buf, t))` \
         evaluates the load for t >= n (the ssa_dominance_scan bug). Control-flow-nest the guard so the load only \
         runs when in range. Offenders:\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}

/// Third standing net, STRICTLY STRONGER than the OOB gate: a primitive's OUTPUT
/// must be INVARIANT to grid over-fire, not merely OOB-clean under it.
///
/// The OOB gate catches an over-fired lane that reads/writes PAST a buffer. But an
/// over-fired lane can also corrupt the result WITHOUT any OOB, by writing a
/// wrong-but-IN-BOUNDS slot (e.g. `output[garbage_index & mask] = lane`, or an
/// unconditional `output[lane] = 0` zero-fill that reaches a valid slot no natural
/// lane touched). Real GPUs dispatch WHOLE workgroups, so those extra lanes DO run;
/// if they change any output byte, the GPU result diverges from the reference oracle
/// every other test trusts. This gate runs each fixtured primitive at its natural
/// (buffer-inferred) grid AND over-fired by one workgroup and asserts the returned
/// outputs are byte-identical, so any grid-sensitive write is caught registry-wide,
/// generalizing the byte_histogram over-fire zero-fill fix to the whole surface.
#[test]
fn every_registered_primitive_output_is_invariant_under_grid_overfire() {
    let mut offenders = Vec::new();
    let mut checked_cases = 0usize;

    for entry in vyre_primitives::harness::all_entries() {
        let Some(inputs_fn) = entry.test_inputs else {
            continue;
        };
        let program = (entry.build)();
        let overfire_grid = overfire_grid(&program);
        for (case_idx, case) in inputs_fn().into_iter().enumerate() {
            let values: Vec<Value> = case.into_iter().map(Value::from).collect();
            // Baseline = the canonical reference output every other test trusts.
            let Ok(baseline) = vyre_reference::reference_eval(&program, &values) else {
                // A malformed fixture that the interpreter rejects is surfaced by the
                // conformance harness, not here (skip it (matches the OOB gates)).
                continue;
            };
            let Ok(overfired) =
                vyre_reference::reference_eval_with_dispatch(&program, &values, overfire_grid)
            else {
                continue;
            };
            checked_cases += 1;
            // Compare every returned output buffer byte-for-byte.
            let base_bytes: Vec<Vec<u8>> = baseline.iter().map(Value::to_bytes).collect();
            let over_bytes: Vec<Vec<u8>> = overfired.iter().map(Value::to_bytes).collect();
            if base_bytes != over_bytes {
                // Locate the first diverging output for a precise message.
                let where_ = base_bytes
                    .iter()
                    .zip(over_bytes.iter())
                    .position(|(a, b)| a != b)
                    .map_or_else(
                        || format!("output count {} vs {}", base_bytes.len(), over_bytes.len()),
                        |idx| format!("output #{idx} differs"),
                    );
                offenders.push(format!(
                    "{} (fixture case {case_idx}, grid>={overfire_grid}): {where_}",
                    entry.id
                ));
            }
        }
    }

    assert!(
        checked_cases > 0,
        "Fix: no registered primitive fixtures were exercised (enable domain features (e.g. --features inventory-registry,all-lego))."
    );
    assert!(
        offenders.is_empty(),
        "Fix: {} of {checked_cases} checked registered-primitive fixture case(s) produced a DIFFERENT output when \
         the dispatch was OVER-FIRED by one workgroup. Whole-workgroup GPU dispatch runs the extra lanes, so an \
         over-fired lane that writes a wrong-but-in-bounds slot (an ungated `output[lane]=…` / zero-fill / \
         scatter) makes the real GPU diverge from the reference oracle without any OOB. Gate every write so \
         over-fired lanes are no-ops (guard `lane < logical_count` / `lane < buf_len` via control flow). \
         Offenders:\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}

/// Fourth standing net, orthogonal to the three grid nets: a primitive's output
/// must be INVARIANT to the order lanes are stepped (i.e. it must be RACE-FREE).
///
/// The GPU makes NO ordering guarantee for NON-atomic stores: if two lanes plain-
/// `store` the same slot, the winner is driver-defined and varies run to run. The
/// single-threaded reference resolves that race DETERMINISTICALLY (last stepped lane
/// wins), so the output looks stable here while it is nondeterministic on hardware 
/// a hazard the three grid gates CANNOT see (they compare same-order runs). This gate
/// runs each fixture once forward and once with `reference_eval_lane_reversed` (step
/// order reversed) and asserts identical output: a race-free primitive (disjoint
/// output slots, or commutative atomics for any shared slot) is order-invariant; a
/// non-atomic cross-lane write-write conflict diverges, exactly as it would across GPU
/// runs. The positive/negative controls below prove the detector actually fires.
#[test]
fn every_registered_primitive_is_race_free_under_lane_reversal() {
    let mut offenders = Vec::new();
    let mut checked_cases = 0usize;

    for entry in vyre_primitives::harness::all_entries() {
        let Some(inputs_fn) = entry.test_inputs else {
            continue;
        };
        let program = (entry.build)();
        for (case_idx, case) in inputs_fn().into_iter().enumerate() {
            let values: Vec<Value> = case.into_iter().map(Value::from).collect();
            let Ok(forward) = vyre_reference::reference_eval(&program, &values) else {
                continue;
            };
            let Ok(reversed) = vyre_reference::reference_eval_lane_reversed(&program, &values)
            else {
                continue;
            };
            checked_cases += 1;
            let fwd: Vec<Vec<u8>> = forward.iter().map(Value::to_bytes).collect();
            let rev: Vec<Vec<u8>> = reversed.iter().map(Value::to_bytes).collect();
            if fwd != rev {
                let where_ = fwd
                    .iter()
                    .zip(rev.iter())
                    .position(|(a, b)| a != b)
                    .map_or_else(
                        || "output count".to_string(),
                        |idx| format!("output #{idx}"),
                    );
                offenders.push(format!(
                    "{} (fixture case {case_idx}): {where_} differs",
                    entry.id
                ));
            }
        }
    }

    assert!(
        checked_cases > 0,
        "Fix: no registered primitive fixtures were exercised (enable domain features (e.g. --features inventory-registry,all-lego))."
    );
    assert!(
        offenders.is_empty(),
        "Fix: {} of {checked_cases} checked registered-primitive fixture case(s) produced a DIFFERENT output when \
         the lane STEP ORDER was reversed, a non-atomic cross-lane write-write RACE. The GPU leaves the winner of \
         two lanes plain-storing the same slot driver-defined (nondeterministic run to run); the reference hides it \
         by resolving last-stepped-wins. Use an atomic (atomic_or/atomic_add/…) for any slot >1 lane may write, or \
         give each lane a disjoint slot. Offenders:\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}

/// Positive control: a program with a deliberate non-atomic write-write race MUST be
/// caught by the forward-vs-reversed comparison, otherwise the gate above would pass
/// vacuously (a detector that always reports "equal" proves nothing).
#[test]
fn lane_reversal_detects_a_deliberate_write_write_race() {
    use std::sync::Arc;
    use vyre_foundation::ir::model::expr::Ident;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

    // Four lanes each unconditionally `store(out[0], local_x)`: a textbook race on
    // out[0]. Forward steps lanes 0..3 so lane 3 writes last (out[0]==3); reversed
    // steps 3..0 so lane 0 writes last (out[0]==0). A real GPU would return either
    // (or another) nondeterministically.
    let body = vec![Node::Region {
        generator: Ident::from("test::race_positive_control"),
        source_region: None,
        body: Arc::new(vec![Node::store("out", Expr::u32(0), Expr::local_x())]),
    }];
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [4, 1, 1],
        body,
    );
    let inputs = vec![Value::from(vec![0u8; 4])];
    let forward = vyre_reference::reference_eval(&program, &inputs).expect("forward eval");
    let reversed =
        vyre_reference::reference_eval_lane_reversed(&program, &inputs).expect("reversed eval");
    assert_eq!(
        forward[0].to_bytes(),
        3u32.to_le_bytes().to_vec(),
        "forward step order: the last lane (3) wins the race on out[0]"
    );
    assert_eq!(
        reversed[0].to_bytes(),
        0u32.to_le_bytes().to_vec(),
        "reversed step order: the first lane (0) wins, proving the detector distinguishes step order"
    );
    assert_ne!(
        forward[0].to_bytes(),
        reversed[0].to_bytes(),
        "the race MUST make forward and reversed diverge, or the gate is blind"
    );
}

/// Negative control: a race-FREE program (each lane writes its own disjoint slot) MUST
/// be order-invariant, so the gate does not false-positive on correct scatter.
#[test]
fn lane_reversal_is_invariant_for_a_race_free_program() {
    use std::sync::Arc;
    use vyre_foundation::ir::model::expr::Ident;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

    // Each lane writes out[local_x] = local_x (disjoint slots, no shared write).
    let body = vec![Node::Region {
        generator: Ident::from("test::race_negative_control"),
        source_region: None,
        body: Arc::new(vec![Node::store("out", Expr::local_x(), Expr::local_x())]),
    }];
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(4)],
        [4, 1, 1],
        body,
    );
    let inputs = vec![Value::from(vec![0u8; 16])];
    let forward = vyre_reference::reference_eval(&program, &inputs).expect("forward eval");
    let reversed =
        vyre_reference::reference_eval_lane_reversed(&program, &inputs).expect("reversed eval");
    assert_eq!(
        forward.iter().map(Value::to_bytes).collect::<Vec<_>>(),
        reversed.iter().map(Value::to_bytes).collect::<Vec<_>>(),
        "disjoint-slot scatter must be identical regardless of lane step order"
    );
    // And the actual value is the expected [0,1,2,3].
    assert_eq!(
        forward[0].to_bytes(),
        vyre_primitives::wire::pack_u32_slice(&[0, 1, 2, 3]),
        "each lane wrote its own slot"
    );
}
