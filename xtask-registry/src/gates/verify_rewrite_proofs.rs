//! `verify-rewrite-proofs` - discharge every shipped optimizer rewrite obligation.
//!
//! Walks `vyre_foundation::optimizer::rewrite_proof_registry::shipped_obligations`,
//! emits each obligation as SMT-LIB v2 under `target/rewrite-proofs/<rewrite>.smt2`,
//! and runs `z3 -smt2` on the script. An obligation is discharged only when z3
//! answers `unsat`. Every other verdict is a finding. A missing `z3` binary is a
//! gate error rather than a pass, because a gate that cannot decide has not
//! judged the tree.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use xtask::gate::{Finding, Gate, GateCtx, GateError, Report};

/// Verifies every optimizer rewrite proof obligation with an SMT solver.
pub struct VerifyRewriteProofs;

impl Gate for VerifyRewriteProofs {
    fn name(&self) -> &'static str {
        "verify-rewrite-proofs"
    }

    fn help(&self) -> &'static str {
        "Verify every optimizer rewrite proof fixture"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let obligations = vyre_foundation::optimizer::rewrite_proof_registry::shipped_obligations();
        let out_dir = ctx.root.join("target").join("rewrite-proofs");
        fs::create_dir_all(&out_dir).map_err(|error| {
            GateError::new(
                format!("failed to create {}: {error}", out_dir.display()),
                "make the target directory writable, then run the gate again",
            )
        })?;
        let solver = solver_path().ok_or_else(|| {
            GateError::new(
                "z3 is not on PATH, so no rewrite obligation can be discharged",
                "install z3 and run the gate again; an undecided obligation is not a proven one",
            )
        })?;

        let mut report = Report::clean();
        report.note(format!("{} shipped obligation(s)", obligations.len()));
        let mut proven = 0usize;
        for obligation in &obligations {
            let script_path = out_dir.join(format!("{}.smt2", obligation.rewrite));
            fs::write(&script_path, obligation.to_smt2().as_bytes()).map_err(|error| {
                GateError::new(
                    format!("failed to write {}: {error}", script_path.display()),
                    "make the target directory writable, then run the gate again",
                )
            })?;
            match verdict(&solver, &script_path)?.as_str() {
                "unsat" => proven += 1,
                "sat" => report.find(Finding::new(
                    format!(
                        "rewrite `{}` is unsound: z3 found a counter-model",
                        obligation.rewrite
                    ),
                    format!(
                        "repair the rewrite in vyre-foundation or narrow its guard, then re-run; the counter-model is reproducible with `z3 -smt2 {}`",
                        script_path.display()
                    ),
                )),
                other => report.find(Finding::new(
                    format!(
                        "rewrite `{}` is undischarged: z3 answered `{other}`",
                        obligation.rewrite
                    ),
                    format!(
                        "strengthen the obligation until z3 decides it, or narrow the rewrite; the script is at {}",
                        script_path.display()
                    ),
                )),
            }
        }
        report.note(format!("{proven} obligation(s) proven unsat"));
        Ok(report)
    }
}

fn verdict(solver: &Path, script: &Path) -> Result<String, GateError> {
    let output = Command::new(solver)
        .arg("-smt2")
        .arg(script)
        .output()
        .map_err(|error| {
            GateError::new(
                format!("failed to spawn {}: {error}", solver.display()),
                "repair the z3 installation, then run the gate again",
            )
        })?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.lines().next().unwrap_or("").trim().to_string())
}

fn solver_path() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|entry| entry.join("z3"))
        .find(|candidate| candidate.is_file())
}
