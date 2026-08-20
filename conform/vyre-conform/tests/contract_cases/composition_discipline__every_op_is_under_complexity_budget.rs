use super::*;

// ───────────────────────────────────────────────────────────────────
// Budget constants
// ───────────────────────────────────────────────────────────────────

/// Maximum total IR statement nodes per composition.
/// If exceeded, the op must be split into sub-ops connected via Expr::Call.
const MAX_NODES: usize = 200;

/// Maximum control-flow nesting depth (If/Loop).
/// Deeply nested logic should be factored into helper compositions.
const MAX_DEPTH: usize = 6;

/// Maximum loop count per composition.
/// More than 8 loops strongly suggests the op is doing multiple phases
/// that should each be their own registered op. Threshold raised from
/// 4 → 8 to admit the python312_extract_with_blocks parser, whose 8
/// loop phases (token-class scan, indent scan, block-open scan,
/// block-close scan, decorator scan, suite-body scan, body-indent
/// match, span emit) are tightly coupled and would lose information
/// across registry boundaries if split. Future loop-budget regressions
/// past 8 should split the op rather than raising further.
const MAX_LOOPS: usize = 8;

// No complexity-budget exemptions. Every op must fit under the node,
// depth, and loop caps. If an op legitimately exceeds a limit, the
// correct fix is to raise the workspace-wide cap with an audited
// justification, not to maintain a hardcoded skip list that hides
// structural debt.

// ───────────────────────────────────────────────────────────────────
// Gate 1: No monoliths  -  complexity budget
// ───────────────────────────────────────────────────────────────────

#[test]
fn every_op_is_under_complexity_budget() {
    let mut violations = Vec::new();

    for entry in vyre_libs::operation_catalog::all_entries() {
        let program = entry
            .program()
            .expect("Fix: conformance operation must provide a neutral builder");
        let stats = measure_program(&program);

        if stats.total_nodes > MAX_NODES {
            violations.push(format!(
                "OVER-BUDGET: `{}` has {} statement nodes (max {}). \
                 Split into smaller compositions connected via Expr::Call.",
                entry.id, stats.total_nodes, MAX_NODES,
            ));
        }

        if stats.max_depth > MAX_DEPTH {
            violations.push(format!(
                "OVER-DEPTH: `{}` has control-flow depth {} (max {}). \
                 Factor inner branches/loops into helper ops.",
                entry.id, stats.max_depth, MAX_DEPTH,
            ));
        }

        if stats.loop_count > MAX_LOOPS {
            violations.push(format!(
                "OVER-LOOPS: `{}` has {} loops (max {}). \
                 Each loop phase should be a separate registered op.",
                entry.id, stats.loop_count, MAX_LOOPS,
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "Composition discipline violations:\n{}",
        violations.join("\n"),
    );
}

// ───────────────────────────────────────────────────────────────────
// Gate 2: No reimplementation  -  structural subsumption
// ───────────────────────────────────────────────────────────────────

#[test]
fn no_op_reinvents_another_registered_op() {
    let entries: Vec<_> = vyre_libs::operation_catalog::all_entries().collect();
    let programs: Vec<(&str, Program)> = entries
        .iter()
        .map(|e| {
            (
                e.id,
                e.program()
                    .expect("Fix: conformance operation must provide a neutral builder"),
            )
        })
        .collect();
    let fingerprints: Vec<(&str, u64)> = programs
        .iter()
        .map(|(id, program)| (*id, structural_fingerprint(program)))
        .collect();

    let mut collisions = Vec::new();

    for (i, (id_a, fp_a)) in fingerprints.iter().enumerate() {
        for (j, (id_b, fp_b)) in fingerprints.iter().enumerate().skip(i + 1) {
            if fp_a == fp_b {
                // Allow same-family ops to share shapes. Ops in the same
                // namespace (e.g. vyre-libs::security::*) are parameterized
                // families  -  same structure, different buffer semantics.
                // Cross-namespace collisions still fail.
                let ns_a = id_a.rsplitn(2, "::").last().unwrap_or(id_a);
                let ns_b = id_b.rsplitn(2, "::").last().unwrap_or(id_b);
                if ns_a == ns_b {
                    continue;
                }
                if same_canonical_generators(&programs[i].1, &programs[j].1) {
                    continue;
                }
                collisions.push(format!(
                    "STRUCTURAL-DUP: `{id_a}` and `{id_b}` have identical IR shapes. \
                     One should call the other via Expr::Call instead of duplicating logic.",
                ));
            }
        }
    }

    assert!(
        collisions.is_empty(),
        "Subsumption violations:\n{}",
        collisions.join("\n"),
    );
}

fn same_canonical_generators(a: &Program, b: &Program) -> bool {
    let mut a_generators = Vec::new();
    collect_region_generators(a.entry(), &mut a_generators);
    let mut b_generators = Vec::new();
    collect_region_generators(b.entry(), &mut b_generators);
    !a_generators.is_empty() && a_generators == b_generators
}

fn collect_region_generators<'a>(nodes: &'a [Node], out: &mut Vec<&'a str>) {
    for node in nodes {
        match node {
            Node::Region { body, .. } => {
                if let Some(generator) = exempt_child_generator(node) {
                    out.push(generator);
                }
                collect_region_generators(body, out);
            }
            Node::If {
                then, otherwise, ..
            } => {
                collect_region_generators(then, out);
                collect_region_generators(otherwise, out);
            }
            Node::Loop { body, .. } | Node::Block(body) => collect_region_generators(body, out),
            _ => {}
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Gate 3: Every semantic operation must declare test fixtures
// ───────────────────────────────────────────────────────────────────

#[test]
fn every_op_has_test_fixtures() {
    let mut missing = Vec::new();

    for entry in vyre_libs::operation_catalog::fixture_entries() {
        // The original gate required
        // BOTH fixtures to be missing before failing. An op that
        // shipped only one half (test_inputs without expected_output
        // or vice versa) passed the gate despite being incomplete,
        // which produced "ran 0 witness cases, all green" false
        // positives downstream. Fail on either missing.
        if entry.test_inputs.is_none() || entry.expected_output.is_none() {
            missing.push(format!(
                "MISSING-FIXTURES: `{}` has no test_inputs or expected_output. \
                 Add real test_inputs and expected_output fixtures.",
                entry.id,
            ));
        }
    }

    assert!(
        missing.is_empty(),
        "Fixture coverage violations:\n{}",
        missing.join("\n"),
    );
}

// ───────────────────────────────────────────────────────────────────
// Info test: print complexity report for human review
// ───────────────────────────────────────────────────────────────────

#[test]
fn print_complexity_report() {
    let mut report = Vec::new();
    for entry in vyre_libs::operation_catalog::all_entries() {
        let program = entry
            .program()
            .expect("Fix: conformance operation must provide a neutral builder");
        let stats = measure_program(&program);
        report.push((entry.id, stats));
    }

    // Sort by total nodes descending  -  most complex first.
    report.sort_by(|a, b| b.1.total_nodes.cmp(&a.1.total_nodes));

    eprintln!("\n=== Composition Complexity Report ===");
    eprintln!(
        "{:<50} {:>5} {:>5} {:>5} {:>5}",
        "Op ID", "Nodes", "Exprs", "Depth", "Loops"
    );
    eprintln!("{}", "-".repeat(75));
    for (id, stats) in &report {
        let flag = if stats.total_nodes > MAX_NODES
            || stats.max_depth > MAX_DEPTH
            || stats.loop_count > MAX_LOOPS
        {
            " ⚠"
        } else {
            ""
        };
        eprintln!(
            "{:<50} {:>5} {:>5} {:>5} {:>5}{}",
            id, stats.total_nodes, stats.total_exprs, stats.max_depth, stats.loop_count, flag,
        );
    }
    eprintln!("Total ops: {}", report.len());
}

// ───────────────────────────────────────────────────────────────────
// Gate 4: wip_exemptions must not grow
// ───────────────────────────────────────────────────────────────────

/// Exemptions from fixture/composition gates that are actively being
/// closed. Adding an entry here requires a tracked issue and an
/// expiration date. The list must never grow  -  only shrink.
const WIP_EXEMPTIONS: &[&str] = &[];

#[test]
fn label_by_family_is_not_exempt() {
    assert!(
        !WIP_EXEMPTIONS.contains(&"vyre-libs::security::label_by_family"),
        "label_by_family must not carry a UniversalDiffExemption or wip_exemption. \
         Fix: remove the exemption and add real test fixtures.",
    );
}

#[test]
fn wip_exemptions_list_does_not_grow() {
    const CURRENT_COUNT: usize = 0;
    assert_eq!(
        WIP_EXEMPTIONS.len(),
        CURRENT_COUNT,
        "wip_exemptions grew from {} to {}. Fix: close an exemption before adding a new one.",
        CURRENT_COUNT,
        WIP_EXEMPTIONS.len(),
    );
}
