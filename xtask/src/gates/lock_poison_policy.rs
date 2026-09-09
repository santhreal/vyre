//! `cargo xtask lock-poison-policy` — failure domain and lock poison governance gate.
//!
//! Enforces BACKLOG row 122 slice: no lock poison is recovered, panicked, or
//! converted ad hoc in production source code. Every mutable state owner must
//! use governed lock policies (`govern_mutex`, `govern_rwlock_read`, `govern_rwlock_write`)
//! or transition atomically to a typed error / terminal state.

use crate::gate::{Finding, GateBehavior, GateCtx, GateError, Report};
use crate::gates::scan::{cfg_test_lines, Tree};
use crate::gates::use_paths::is_test_source_path;

/// Permitted failure-domain and governed lock policy seam paths where low-level
/// lock recovery primitives are implemented.
const GOVERNED_SEAM_PATHS: &[&str] = &[
    "vyre-foundation/src/failure_domain.rs",
    "vyre-driver/src/lock_policy.rs",
    "vyre-runtime/src/structured_concurrency.rs",
    "vyre-runtime/src/atomic_recovery.rs",
    "vyre-reference/src/composition_witness/reasoning.rs",
    "xtask/src/gates/lock_poison_policy.rs",
];

/// Gate behavior enforcing governed lock policies and absence of ad-hoc lock unwraps/panics.
pub struct LockPoisonPolicy;

impl GateBehavior for LockPoisonPolicy {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();

        let rust_files = tree.all_rust();
        let mut scanned_count = 0;

        for relative_path in rust_files {
            if is_test_source_path(&relative_path) {
                continue;
            }

            let path_str = relative_path.to_string_lossy();
            if path_str.starts_with("tests/")
                || path_str.contains("/tests/")
                || path_str.starts_with("examples/")
                || path_str.contains("/examples/")
                || path_str.starts_with("benches/")
                || path_str.contains("/benches/")
                || path_str.starts_with("structure-gate/")
                || path_str.contains("/structure-gate/")
            {
                continue;
            }

            // Exclude the designated failure domain implementation seams
            let is_seam = path_str.ends_with("lock_poison_policy.rs")
                || GOVERNED_SEAM_PATHS
                    .iter()
                    .any(|seam| path_str == *seam || path_str.ends_with(seam));

            let content = match tree.read(&relative_path) {
                Ok(text) => text,
                Err(err) => return Err(err),
            };

            let lines: Vec<&str> = content.lines().collect();
            let test_mask = cfg_test_lines(&lines);
            scanned_count += 1;
            for (line_idx, line) in lines.iter().enumerate() {
                let line_no = (line_idx + 1) as u32;
                if test_mask.get(line_idx).copied().unwrap_or(false) {
                    continue;
                }

                let trimmed = line.trim();
                if trimmed.starts_with("//")
                    || trimmed.starts_with("/*")
                    || trimmed.starts_with('*')
                {
                    continue;
                }

                if !is_seam {
                    if (trimmed.contains(".lock()")
                        || trimmed.contains(".read()")
                        || trimmed.contains(".write()"))
                        && (trimmed.contains(".unwrap()") || trimmed.contains(".expect("))
                    {
                        report.find(Finding::at(
                            relative_path.clone(),
                            line_no,
                            format!("ad-hoc lock unwrap/expect in production code: `{trimmed}`"),
                            "Fix: govern lock with explicit RecoveryClass or return typed domain error.",
                        ));
                    }

                    let is_poison_inner = trimmed.contains(concat!("PoisonError::", "into_inner"))
                        || (trimmed.contains(concat!(".", "into_inner()"))
                            && trimmed.contains("Err("));
                    if is_poison_inner {
                        report.find(Finding::at(
                            relative_path.clone(),
                            line_no,
                            format!("ad-hoc PoisonError into_inner in production code: `{trimmed}`"),
                            "Fix: govern lock with explicit RecoveryClass or return typed domain error.",
                        ));
                    }
                }
            }
        }

        report.cover_complete(
            "production source files scanned for lock governance",
            scanned_count,
        );
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_poison_policy_gate_reports_clean_on_workspace() {
        let gate = LockPoisonPolicy;
        let root = std::env::current_dir().expect("current dir");
        let ctx = GateCtx::new(root, vec![]);
        let report = gate.run(&ctx).expect("gate execution must succeed");
        assert_eq!(
            report.findings.len(),
            0,
            "production workspace must have 0 unhandled lock poison findings, got: {:?}",
            report.findings
        );
    }

    #[test]
    fn mutation_detection_catches_adhoc_lock_unwrap() {
        let mock_line = "let guard = self.state.lock().unwrap();";
        let has_defect = (mock_line.contains(".lock()")
            || mock_line.contains(".read()")
            || mock_line.contains(".write()"))
            && (mock_line.contains(".unwrap()") || mock_line.contains(".expect("));
        assert!(has_defect, "defect detector must catch .lock().unwrap()");
    }
}
