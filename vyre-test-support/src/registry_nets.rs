//! The four registry parity nets, run against whatever population a crate
//! registers.
//!
//! Every crate that registers programs owes the same four contracts: a
//! registered program reads no buffer out of bounds on its own fixture, it
//! stays out of bounds when the dispatch is over-fired by a whole workgroup, it
//! returns the same bytes at both grids, and it returns the same bytes when the
//! lane step order is reversed. The reference interpreter absorbs an
//! out-of-bounds access (`vyre-reference/src/oob.rs`: zero-fill loads, dropped
//! stores) and resolves a store race deterministically, so a program can pass
//! every other test while a backend that bounds-checks nothing reads garbage.
//!
//! The nets were written twice, once per crate, and the second copy drifted:
//! one refused a case it could not evaluate and the other skipped it, so the
//! same population produced a clean sweep of a subset on one surface and a
//! failure on the other. The driver lives here and the caller supplies only its
//! population, so both surfaces are judged by the same rule.
//!
//! A case the net cannot evaluate is a failure, not a skip. An un-evaluable
//! fixture leaves its composition unjudged, and reporting the remainder as
//! clean is the exact failure these nets exist to remove.

use vyre_foundation::ir::Program;
use vyre_foundation::operation::SemanticOperation;
use vyre_reference::value::Value;

use crate::pass_programs::overfire_grid;

/// One registered entry's program paired with one of its fixture cases.
pub struct SweepCase {
    /// How a finding names this case, normally `<op-id> (case N)`.
    label: String,
    /// The program the entry builds.
    program: Program,
    /// The input values the fixture supplies.
    inputs: Vec<Value>,
}

impl SweepCase {
    /// One case of one registered entry.
    #[must_use]
    pub fn new(label: String, program: Program, inputs: Vec<Value>) -> Self {
        Self {
            label,
            program,
            inputs,
        }
    }
}

/// Every fixture case one registered surface publishes, and the four nets it
/// must pass.
pub struct RegistrySweep {
    /// The surface a finding names, normally the crate's registry.
    surface: &'static str,
    /// Every case the caller derived from that registry.
    cases: Vec<SweepCase>,
}

impl RegistrySweep {
    /// The population one surface registers.
    #[must_use]
    pub fn new(surface: &'static str, cases: Vec<SweepCase>) -> Self {
        Self { surface, cases }
    }

    /// Every fixture case a catalog of registered operations publishes.
    ///
    /// The catalog walk is the same on every surface: refuse an empty one,
    /// carry an unfixtured entry as out of reach rather than dropping it in
    /// silence, and pair each fixture case with the program its entry builds.
    /// Writing it per crate is what let one surface skip a case the other
    /// refused.
    ///
    /// # Panics
    /// Panics when the catalog is empty, which passes every net without
    /// proving anything, or when a registered entry builds no program.
    #[must_use]
    pub fn from_catalog(
        surface: &'static str,
        entries: impl Iterator<Item = SemanticOperation>,
    ) -> Self {
        let mut cases = Vec::new();
        let mut total = 0usize;
        let mut fixtured = 0usize;

        for entry in entries {
            total += 1;
            let Some(inputs_fn) = entry.test_inputs else {
                continue;
            };
            fixtured += 1;
            let program = entry.program().unwrap_or_else(|| {
                panic!(
                    "Fix: registered operation `{}` carries a fixture but builds no program",
                    entry.id
                )
            });
            for (index, case) in inputs_fn().into_iter().enumerate() {
                cases.push(SweepCase::new(
                    format!("{} (fixture case {index})", entry.id),
                    program.clone(),
                    case.into_iter().map(Value::from).collect(),
                ));
            }
        }

        assert!(
            total > 0,
            "Fix: {surface} registered nothing on this run, so every net below judges an empty \
             population. Select the domain features so the catalog populates."
        );
        eprintln!(
            "{surface}: {fixtured}/{total} entries fixtured, {} case(s), {} entry(s) out of reach",
            cases.len(),
            total - fixtured
        );
        Self::new(surface, cases)
    }

    /// No case reads or writes out of bounds on its own fixture inputs.
    ///
    /// # Panics
    /// Panics when a case accesses a buffer out of bounds, when a case cannot
    /// be evaluated, or when the population is empty.
    pub fn assert_oob_clean(&self) {
        let mut offenders = Vec::new();
        let mut skipped = Vec::new();
        let mut checked = 0usize;

        for case in &self.cases {
            match vyre_reference::reference_eval_oob_report(&case.program, &case.inputs) {
                Ok((_out, report)) => {
                    checked += 1;
                    if report.total() > 0 {
                        offenders.push(format!(
                            "{}: {} OOB load(s), {} OOB store(s), {} OOB atomic(s)",
                            case.label, report.oob_loads, report.oob_stores, report.oob_atomics
                        ));
                    }
                }
                Err(err) => skipped.push(format!("{}: {err}", case.label)),
            }
        }

        self.refuse_skips("the fixture out-of-bounds sweep", checked, &skipped);
        assert!(
            offenders.is_empty(),
            "Fix: {} of {checked} checked {} fixture case(s) accessed a buffer OUT OF BOUNDS on VALID input. \
             The reference masks that access with a zero-fill load or a dropped store; a backend that \
             bounds-checks nothing reads whatever is there. Fold the index into range at the load, or gate the \
             access with control flow. Offenders:\n{}",
            offenders.len(),
            self.surface,
            offenders.join("\n")
        );
    }

    /// No case reads or writes out of bounds when the dispatch over-fires by one
    /// whole workgroup.
    ///
    /// # Panics
    /// Panics when an over-fired lane accesses a buffer out of bounds, when a
    /// case cannot be evaluated, or when the population is empty.
    pub fn assert_oob_clean_under_overfire(&self) {
        let mut offenders = Vec::new();
        let mut skipped = Vec::new();
        let mut checked = 0usize;

        for case in &self.cases {
            let grid = overfire_grid(&case.program);
            match vyre_reference::reference_eval_with_dispatch_oob_report(
                &case.program,
                &case.inputs,
                grid,
            ) {
                Ok((_out, report)) => {
                    checked += 1;
                    if report.total() > 0 {
                        offenders.push(format!(
                            "{} (grid>={grid}): {} OOB load(s), {} OOB store(s), {} OOB atomic(s)",
                            case.label, report.oob_loads, report.oob_stores, report.oob_atomics
                        ));
                    }
                }
                Err(err) => skipped.push(format!("{} (over-fired): {err}", case.label)),
            }
        }

        self.refuse_skips("the over-fired out-of-bounds sweep", checked, &skipped);
        assert!(
            offenders.is_empty(),
            "Fix: {} of {checked} checked {} fixture case(s) accessed a buffer OUT OF BOUNDS when the dispatch \
             was OVER-FIRED by one workgroup. A device dispatches whole workgroups, so the extra lanes run. \
             `Expr::and` and `Expr::select` evaluate both sides, so neither is a bounds guard: nest the access \
             in control flow, or fold the index into range. Offenders:\n{}",
            offenders.len(),
            self.surface,
            offenders.join("\n")
        );
    }

    /// Every case returns the same bytes at its natural grid and over-fired.
    ///
    /// Stronger than [`Self::assert_oob_clean_under_overfire`]: an over-fired
    /// lane can write a wrong slot that is still in bounds, which no
    /// out-of-bounds report sees.
    ///
    /// # Panics
    /// Panics when an over-fired grid changes any output byte, when a case
    /// cannot be evaluated, or when the population is empty.
    pub fn assert_output_invariant_under_overfire(&self) {
        let mut offenders = Vec::new();
        let mut skipped = Vec::new();
        let mut checked = 0usize;

        for case in &self.cases {
            let grid = overfire_grid(&case.program);
            let baseline = match vyre_reference::reference_eval(&case.program, &case.inputs) {
                Ok(baseline) => baseline,
                Err(err) => {
                    skipped.push(format!("{}: {err}", case.label));
                    continue;
                }
            };
            let overfired = match vyre_reference::reference_eval_with_dispatch(
                &case.program,
                &case.inputs,
                grid,
            ) {
                Ok(overfired) => overfired,
                Err(err) => {
                    skipped.push(format!("{} (over-fired): {err}", case.label));
                    continue;
                }
            };
            checked += 1;
            let base_bytes: Vec<Vec<u8>> = baseline.iter().map(Value::to_bytes).collect();
            let over_bytes: Vec<Vec<u8>> = overfired.iter().map(Value::to_bytes).collect();
            if base_bytes != over_bytes {
                let where_ = base_bytes
                    .iter()
                    .zip(over_bytes.iter())
                    .position(|(left, right)| left != right)
                    .map_or_else(
                        || format!("output count {} vs {}", base_bytes.len(), over_bytes.len()),
                        |index| format!("output #{index} differs"),
                    );
                offenders.push(format!("{} (grid>={grid}): {where_}", case.label));
            }
        }

        self.refuse_skips("the over-fire invariance sweep", checked, &skipped);
        assert!(
            offenders.is_empty(),
            "Fix: {} of {checked} checked {} fixture case(s) produced a DIFFERENT output when the dispatch was \
             OVER-FIRED by one workgroup. The extra lanes run on a device, so a write one of them reaches that \
             no natural lane touches diverges from the oracle every other test trusts, with no out-of-bounds \
             access to show for it. Gate every write on the logical count. Offenders:\n{}",
            offenders.len(),
            self.surface,
            offenders.join("\n")
        );
    }

    /// Every case returns the same bytes when the lane step order is reversed.
    ///
    /// # Panics
    /// Panics when reversing the step order changes any output byte, when a
    /// case cannot be evaluated, or when the population is empty.
    pub fn assert_race_free_under_lane_reversal(&self) {
        let mut offenders = Vec::new();
        let mut skipped = Vec::new();
        let mut checked = 0usize;

        for case in &self.cases {
            let forward = match vyre_reference::reference_eval(&case.program, &case.inputs) {
                Ok(forward) => forward,
                Err(err) => {
                    skipped.push(format!("{}: {err}", case.label));
                    continue;
                }
            };
            let reversed =
                match vyre_reference::reference_eval_lane_reversed(&case.program, &case.inputs) {
                    Ok(reversed) => reversed,
                    Err(err) => {
                        skipped.push(format!("{} (reversed): {err}", case.label));
                        continue;
                    }
                };
            checked += 1;
            let forward_bytes: Vec<Vec<u8>> = forward.iter().map(Value::to_bytes).collect();
            let reversed_bytes: Vec<Vec<u8>> = reversed.iter().map(Value::to_bytes).collect();
            if forward_bytes != reversed_bytes {
                offenders.push(case.label.clone());
            }
        }

        self.refuse_skips("the lane-reversal sweep", checked, &skipped);
        assert!(
            offenders.is_empty(),
            "Fix: {} of {checked} checked {} fixture case(s) changed their output when the lane STEP ORDER was \
             reversed, which means two lanes write one slot without an atomic. The reference resolves that \
             deterministically and a device does not, so the answer is stable here and driver-defined there. \
             Give every shared slot a commutative atomic or disjoint ownership. Offenders:\n{}",
            offenders.len(),
            self.surface,
            offenders.join("\n")
        );
    }

    /// Refuse a case the net could not evaluate, and an empty population.
    ///
    /// A net that skips what it cannot evaluate reports a clean sweep of a
    /// subset, and a net over an empty population reports a clean sweep of
    /// nothing.
    fn refuse_skips(&self, net: &str, checked: usize, skipped: &[String]) {
        assert!(
            skipped.is_empty(),
            "Fix: {} of the {} fixture case(s) could not be evaluated by {net}, so {} composition(s) went \
             unjudged while the remainder reported clean. Repair the fixture, or the program it feeds. \
             Un-evaluable cases:\n{}",
            skipped.len(),
            self.surface,
            skipped.len(),
            skipped.join("\n")
        );
        assert!(
            checked > 0,
            "Fix: {net} evaluated no case at all from {}. An empty walk passes every net without proving \
             anything, so the registry, the feature selection, or the fixture derivation is the defect.",
            self.surface
        );
    }
}
