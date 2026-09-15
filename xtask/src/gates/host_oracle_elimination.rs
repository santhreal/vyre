//! `cargo xtask host-oracle-elimination` - zero production CPU oracles in
//! shipping crates.
//!
//! A crate that registers semantic operations must not carry a host routine
//! that computes what one of its operations declares a device produces. Such a
//! routine is a witness for a test, and a reader who finds one beside the
//! operation cannot tell which side an answer came from.
//!
//! Every rule here names a syntactic shape:
//!
//! - A function whose name declares it a reference implementation
//!   (`cpu_ref`, `cpu_reference`, a `vyre_reference` simulator) in production
//!   scope.
//! - A direct call to `vyre_reference` outside test scope.
//! - An `OperationRegistration` expected-output producer that is anything
//!   other than exact byte constants, including a helper call, a wire codec
//!   such as `pack_u32_slice`, a loop, or arithmetic. A `test_inputs`
//!   generator may still use a codec: an input is not the answer.
//! - A dispatch error or fallback path (`Err(_)`, `unwrap_or_else`,
//!   `or_else`, `is_err()`) that runs a host candidate, which is the silent
//!   fallback shape.
//! - A post-dispatch host reduction over device output (`any`, `all`, `sum`,
//!   `count`, `fold`, `reduce`, a loop), which must be dispatched instead.
//!
//! Classification is source-derived. Caller identity is an exact definition
//! index, so two same-named methods in different impl blocks do not collapse.
//! Macro bodies (`inventory::submit!`, `vec![]`) are parsed as AST rather than
//! text. Test scope covers parent module graphs, `#[cfg(test)] impl`, and
//! `#[cfg(test)] trait`.
//!
//! # What this no longer proves
//!
//! A sixth rule convicted any host data-processing function that the call
//! graph could not reach from a production root. Its premise was that a host
//! routine nothing production reaches is a test witness left in shipped code.
//! Measured against the crates it names rather than the three directories it
//! was tuned on, it reported 615 findings, 473 of them in the IR crate and 129
//! in drivers: tiling arithmetic, workgroup selection, rank validation, a
//! linker anchor. Every one is a host-side compile-time computation, which
//! this compiler is made of, and the rule had no way to separate that from
//! evaluating a user program. It reported zero only because its scan had
//! narrowed to three paths that no longer described the workspace.
//!
//! So a host semantic twin that no production path reaches, and that is not
//! named as a reference, is no longer detected here. What still catches it is
//! the registration rule when the twin produces an expected output, the
//! fallback rule when a dispatch failure calls it, and the dependency closure
//! below when it lives behind `vyre-reference`. A twin reached by none of
//! those is unproven, and closing that needs a rule that reads what a function
//! computes rather than who calls it.

use crate::gate::{GateCtx, GateError, Report};
use crate::gates::scan::Tree;

use super::host_oracle_elimination_eval::analyze_sources;
use crate::gates::scan::test_module_files;

/// Zero-baseline gate that eliminates host CPU oracles and semantic twins from production library code.
pub struct HostOracleElimination;

impl crate::gate::GateBehavior for HostOracleElimination {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let roots = operation_bearing_roots(&tree, &mut report)?;
        if roots.is_empty() {
            return Err(GateError::new(
                "no shipped crate registers a semantic operation, so there is nothing this rule describes".to_string(),
                "give every operation-registering crate a `[[crate]]` row in docs/CRATE_OWNERSHIP.toml",
            ));
        }
        let borrowed: Vec<&str> = roots.iter().map(String::as_str).collect();
        let sources = tree.rust(&borrowed)?;
        report.cover_complete("production library sources", sources.len());

        let test_scoped_files = test_module_files(&tree, &sources)?;
        let findings = analyze_sources(&tree, &sources, &test_scoped_files)?;
        for finding in findings {
            report.find(finding);
        }

        // Containing no host oracle is not the same as linking none: a shipped
        // crate that names the interpreter as a dependency carries it whether
        // or not a line calls it.
        let closure = super::host_oracle_closure::findings(&tree, &mut report)?;
        for finding in closure {
            report.find(finding);
        }

        report.note(format!(
            "{} production library source file(s) across {} shipped crate(s) analyzed, and every shipped crate's production dependency closure checked for a host evaluator",
            sources.len(),
            roots.len()
        ));
        Ok(report)
    }
}

/// Calling a registration constructor puts a crate in scope.
const REGISTRATION_CALL: &str = "OperationRegistration::";

/// Defining the type does not.
const REGISTRATION_DEFINITION: &str = "impl OperationRegistration";

/// The `src` directory of every shipped crate that registers an operation.
///
/// This is the shape a host semantic twin hides in: an op whose declared
/// meaning is a device program, sitting beside a host routine that computes
/// the same answer, so a reader cannot tell which one the result came from. A
/// crate that registers no operation has no such pairing, and the reachability
/// model behind the twin rule does not describe it: an IR crate builds no
/// program roots and a driver's entry points are trait implementations the
/// runtime calls, so scanning either convicts hundreds of ordinary helpers for
/// being unreachable inside a set that never contained their callers.
///
/// The set is read from the tree. Three literal paths stood here, and when the
/// library and driver crates were split apart the scan narrowed to a fraction
/// of what it claimed while still reporting a clean verdict over the whole
/// workspace. A crate that starts registering operations now enrols itself,
/// and one whose sources move with it stays scanned.
///
/// The crate that declares `OperationRegistration` is excluded by the shape of
/// its own source rather than by name, so moving the type does not silently
/// drop its new home out of scope or pull the old one back in.
pub fn operation_bearing_roots(tree: &Tree, report: &mut Report) -> Result<Vec<String>, GateError> {
    let mut roots = Vec::new();
    for root in super::host_oracle_closure::shipped_source_roots(tree, report)? {
        let mut calls = false;
        let mut defines = false;
        for path in &tree.rust(&[root.as_str()])? {
            let text = tree.read(path)?;
            calls |= text.contains(REGISTRATION_CALL);
            defines |= text.contains(REGISTRATION_DEFINITION);
        }
        if calls && !defines {
            roots.push(root);
        }
    }
    Ok(roots)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::host_oracle_elimination_tests_part1::analyze_files;

    #[test]
    fn mutation_oracle_detection_catches_production_cpu_ref_fn() {
        let code = r#"
pub fn popcount(input: &str, out: &str, n: u32) -> Program {
    let p = Program::new();
    p
}

fn cpu_ref(input: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &x in input {
        out.push((x.count_ones() & 0xFF) as u8);
    }
    out
}
"#;
        let findings = analyze_files(&[("vyre-libs/src/nn/cpu_ref_test.rs", code)]);
        assert!(
            !findings.is_empty(),
            "production cpu_ref helper must be convicted"
        );
        assert!(findings[0].message.contains("`cpu_ref`"));
        assert_eq!(findings[0].line, Some(7));
    }

    /// Every written form of a call into the oracle crate is convicted, and a
    /// name that merely ends in those bytes is not.
    ///
    /// WHY: the rule compared the recorded callee with the bare crate name,
    /// while a callee is recorded as the path the source writes. No Rust
    /// source spells `vyre_reference(..)`, so the rule could not fire and a
    /// production call into the interpreter was reported by the dependency
    /// closure alone, which sees a manifest edge rather than a call.
    ///
    /// What it does not catch: a call through an item imported by `use`, or
    /// through a manifest rename. Those reach the interpreter under a path
    /// that does not name the crate, and the dependency closure is what
    /// convicts them.
    #[test]
    fn a_production_path_into_the_oracle_crate_is_convicted_in_every_written_form() {
        for call in [
            "vyre_reference::ReferenceRequest::standard(program, inputs)",
            "::vyre_reference::output_index(program, \"out\")",
            "vyre_reference::value::Value::from(bytes)",
        ] {
            let code = format!("pub fn probe() {{ let _ = {call}; }}\n");
            let findings = analyze_files(&[("vyre-libs/src/nn/probe.rs", code.as_str())]);
            assert!(
                findings
                    .iter()
                    .any(|finding| finding.message.contains("`vyre_reference`")),
                "Fix: a production call written as `{call}` must be convicted"
            );
        }

        let benign = "pub fn probe() { let _ = local_vyre_reference::helper(); }\n";
        let findings = analyze_files(&[("vyre-libs/src/nn/probe.rs", benign)]);
        assert!(
            !findings
                .iter()
                .any(|finding| finding.message.contains("`vyre_reference`")),
            "Fix: a path whose first segment only ends in the crate name is not a call into it: {:?}",
            findings
                .iter()
                .map(|finding| finding.message.clone())
                .collect::<Vec<_>>()
        );
    }
}
