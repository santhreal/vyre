//! `cargo xtask host-oracle-elimination`  -  zero production CPU oracles in shipping crates.
//!
//! A shipping library (`vyre-libs`, `vyre-primitives`) must not compile or execute
//! host mathematical oracles, reference simulations, or unisolated data-processing semantic
//! twins in production code. CPU reference implementations (`cpu_ref`, `cpu_reference`,
//! `vyre_reference` simulators, and generic-named host algorithms that only serve tests)
//! exist exclusively to provide independent semantic witnesses for test verification; they
//! must never be linked into production binaries or invoked at registration time for dynamic
//! expected-output evaluation.
//!
//! The classification is 100% source-derived and structural:
//! - Candidate detection is role-independent: body AST visitor (`BodyFeatureVisitor`) inspects
//!   `ExprBinary` arithmetic/bitwise/shifts, `ExprUnary` numeric/not, numeric methods (min/max/clamp/abs/sqrt/etc.),
//!   branch-on-data classifiers (ExprIf/ExprMatch), loops, iterators, and search/sort algorithms.
//! - Roles establish reachability through structural types, effects, and call graphs, never by
//!   erasing candidate status.
//! - Trusted roots require exact canonical qualified type provenance derived from actual workspace
//!   declarations and imports (`vyre_foundation::ir::*`, `vyre_foundation::operation::OperationRegistration`,
//!   `vyre_foundation::program_dispatch::*`); bare names, glob imports, `crate::bogus::*`,
//!   sibling-module imports, and local dummy traits/structs fail closed.
//! - IR builder roots strictly require returning AST/IR owner types (`Program`, `Node`, `Expr`,
//!   `OperationRegistration`), optionally wrapped in `Result<T, _>`, `Option<T>`,
//!   `Arc<T>`, `Box<T>`, `Vec<T>`, or homogenous AST owner tuples; metadata types (`DataType`, etc.) and
//!   mixed data-output tuples (`(Vec<u32>, DataType)`) or data results (`Result<Vec<u32>, FusionError>`)
//!   do NOT establish builder roots.
//! - Dispatch roots strictly require an exact canonical dispatcher capability parameter
//!   (`ProgramDispatcher`) AND device dispatch execution in the body, derived dynamically from grounded
//!   trait signatures taking `Program` or `ResidentDispatchStep` plan types and producing dispatch/readback
//!   effects (capability, metadata, allocation, upload-only, and free methods do not establish execution).
//!   Passing dispatcher to non-dispatching helpers does not root; helpers that execute dispatch establish execution
//!   transitively.
//! - Dispatch error / fallback paths (`Err(_)`, `unwrap_or_else`, `or_else`, `*.is_err()`, etc.) are
//!   forbidden from executing host candidates or inline semantic operations; fallback calls do NOT
//!   receive reachability edges and are convicted.
//! - Post-dispatch host reductions / aggregations (`.any()`, `.all()`, `.sum()`, `.count()`, `.fold()`,
//!   `.reduce()`, loops over output) are forbidden; reductions must be dispatched on GPU.
//!   Post-dispatch phase is expression-granular: nested match expressions (`match dispatcher.dispatch(..) { Ok(out) => ... }`),
//!   chained method calls (`dispatcher.dispatch(..).map(|out| ...)`), and conditional expressions flip into post-dispatch
//!   phase for their success continuations.
//! - OperationRegistration expected-output fixture producer contexts require exact byte literal constants
//!   (or allocations from constant byte arrays via `.to_vec()` / `vec![]`). Any dynamic helper function call,
//!   wire codec invocation (`pack_u32_slice`), local helper closure alias, loop, or arithmetic in `expected_output`
//!   convicts the registration; `test_inputs` generators and codecs remain permitted.
//! - Caller identity is tracked by exact definition index to prevent collapsing same-named methods
//!   across different impl blocks or traits.
//! - Macro contents (`ItemMacro`, `ExprMacro`, `StmtMacro`) such as `inventory::submit!` and `vec![]`
//!   are recursively parsed into AST nodes without double-counting traversals.
//! - Test scoping covers parent module graphs, `#[cfg(test)] impl`, and `#[cfg(test)] trait` items.
//! - Dynamic `expected_output` evaluations and computed static/const fixture initializers (resolved via
//!   path references regardless of token naming) are strictly forbidden from executing semantic candidates.

use crate::gate::{GateCtx, GateError, Report};
use crate::gates::scan::Tree;

use super::host_oracle_elimination_eval::analyze_sources;
use super::host_oracle_elimination_records::TARGET_ROOTS;
use crate::gates::scan::test_module_files;

/// Zero-baseline gate that eliminates host CPU oracles and semantic twins from production library code.
pub struct HostOracleElimination;

impl crate::gate::GateBehavior for HostOracleElimination {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let sources = tree.rust(TARGET_ROOTS)?;
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
            "{} production library source file(s) analyzed, and every shipped crate's production dependency closure checked for a host evaluator",
            sources.len()
        ));
        Ok(report)
    }
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
}
