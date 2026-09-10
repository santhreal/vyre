//! `cargo xtask lock-poison-policy` — failure domain and lock poison governance gate.
//!
//! Every owner of mutable shared state states its failure policy by name. A
//! poisoned lock means a thread panicked while holding it, so the guarded value
//! may be half written. The workspace answers that in one place,
//! `vyre-foundation::failure_domain`, which records an owner, the state it
//! guards, and a `RecoveryClass`. This gate rejects the five shapes that answer
//! it somewhere else instead.
//!
//! `OnceLock` and `LazyLock` are deliberately not a category. Neither carries a
//! poison flag and neither exposes the half-initialized value, so an
//! initializer panic leaves no recoverable state and no decision to record. A
//! detector there would report on every static in the tree and prove nothing.

use std::path::Path;

use crate::gate::{Finding, GateBehavior, GateCtx, GateError, Report};
use crate::gates::scan::{cfg_test_lines, Tree};
use crate::gates::use_paths::is_test_source_path;

pub mod state_owners;

/// Files that own a lock poison policy.
///
/// This is not an exemption list. Each entry is the single definition of a
/// decision, and the decision is what the rest of the tree is required to call.
/// `failure_domain` states the workspace policy. `vyre-driver/src/lock_policy.rs`
/// renames that decision into a `BackendError` for backend callers and adds no
/// policy of its own. `xtask/src/lock_policy.rs` is the same decision in the one
/// layer that may not depend on a vyre crate, because a gate resolves a checkout
/// root while the workspace does not compile. This file holds the pattern text
/// it searches for.
const POLICY_OWNER_PATHS: &[&str] = &[
    "vyre-foundation/src/failure_domain.rs",
    "vyre-driver/src/lock_policy.rs",
    "xtask/src/lock_policy.rs",
    "xtask/src/gates/lock_poison_policy.rs",
];

/// A way to answer a poisoned lock outside the policy owners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Violation {
    /// The acquisition and the `unwrap` or `expect` share one line.
    UnwrapOnAcquire,
    /// The acquisition ends a line and the `unwrap` or `expect` opens the next.
    /// Line-at-a-time scanning misses this shape, and `rustfmt` produces it
    /// whenever the receiver expression is long.
    UnwrapAcrossLines,
    /// A closure or match arm takes the guarded value out of the `PoisonError`
    /// and continues. The half-written state is then indistinguishable from a
    /// correct one.
    IntoInnerRecovery,
    /// A pattern match keeps the `Ok` arm and drops the poisoned one. The work
    /// the branch existed to do is skipped and nothing reports it.
    SilentDiscard,
    /// The poison flag is cleared outside a policy owner. Clearing it is the
    /// act of accepting the state, so it belongs with the record of who
    /// accepted it and why.
    HandRolledClearPoison,
    /// An inline closure turns the `PoisonError` into another error type. The
    /// recovery class is lost at the conversion, so every caller downstream
    /// reads a message where a decision belongs. Reported only under the roots
    /// whose owners record a recovery class.
    AdHocPoisonConversion,
}

impl Violation {
    /// Every category this gate detects.
    pub const ALL: &'static [Violation] = &[
        Violation::UnwrapOnAcquire,
        Violation::UnwrapAcrossLines,
        Violation::IntoInnerRecovery,
        Violation::SilentDiscard,
        Violation::HandRolledClearPoison,
        Violation::AdHocPoisonConversion,
    ];

    /// What the finding reports about the source line.
    fn what(self) -> &'static str {
        match self {
            Violation::UnwrapOnAcquire => "lock acquired with unwrap or expect",
            Violation::UnwrapAcrossLines => {
                "lock acquired with unwrap or expect on the following line"
            }
            Violation::IntoInnerRecovery => "poisoned guard unwrapped with into_inner",
            Violation::SilentDiscard => "poisoned lock discarded by a pattern match",
            Violation::HandRolledClearPoison => "poison flag cleared outside a policy owner",
            Violation::AdHocPoisonConversion => {
                "poisoned lock converted by an inline closure, discarding the recovery class"
            }
        }
    }

    /// The corrective action the finding names.
    fn fix(self) -> &'static str {
        match self {
            Violation::UnwrapOnAcquire | Violation::UnwrapAcrossLines => {
                "call a vyre_foundation::failure_domain policy with an owner, the state it guards, \
                 and a RecoveryClass"
            }
            Violation::IntoInnerRecovery | Violation::HandRolledClearPoison => {
                "call a reclaim_poisoned_* or govern_* policy, which records the owner and the \
                 state before accepting the value and clears the flag itself"
            }
            Violation::SilentDiscard => {
                "call a vyre_foundation::failure_domain policy. Skipping the branch loses the work \
                 it existed to do and reports nothing"
            }
            Violation::AdHocPoisonConversion => {
                "acquire through a govern_* policy and convert its TypedRecoveryError with #[from], \
                 which carries the owner, the guarded state and the recovery class"
            }
        }
    }
}

/// Whether a source line acquires a lock.
fn acquires_lock(line: &str) -> bool {
    line.contains(".lock()") || line.contains(".read()") || line.contains(".write()")
}

/// Whether the `into_inner` at `index` takes a guard out of a `PoisonError`
/// rather than consuming an owned `Mutex`, `Cell`, or `RefCell`.
///
/// The distinguishing mark is the poison binding the call reads from, which
/// `rustfmt` may leave a few lines above when the recovery is a block rather
/// than an expression. The search stops at the end of the enclosing statement,
/// so an ordinary `Cell::into_inner` is not judged by whatever the line before
/// it happened to do.
fn laundering_context(lines: &[&str], test_mask: &[bool], index: usize) -> bool {
    let marks = |text: &str| {
        text.contains("PoisonError")
            || text.contains("Err(")
            || text.contains("unwrap_or_else(")
            || acquires_lock(text)
    };
    let mut at = index;
    for step in 0..4 {
        let Some(line) = lines.get(at) else {
            break;
        };
        let trimmed = line.trim();
        let masked = test_mask.get(at).copied().unwrap_or(false);
        if !masked && !is_comment(trimmed) {
            // A line above that closes a statement belongs to a different one,
            // so nothing in it describes this `into_inner`.
            if step > 0 && (trimmed.ends_with(';') || trimmed.ends_with('}')) {
                break;
            }
            if marks(trimmed) {
                return true;
            }
        }
        let Some(previous) = at.checked_sub(1) else {
            break;
        };
        at = previous;
    }
    false
}

/// Every category present in one file, each paired with the line it is reported
/// at.
///
/// Reading the whole file rather than one line at a time is what lets a rule
/// see a chain `rustfmt` split: the acquisition, the recovery it chose, and the
/// value it produces can sit on three separate lines.
pub fn violations_in(lines: &[&str], test_mask: &[bool]) -> Vec<(usize, Violation)> {
    let mut found = Vec::new();
    let live = |at: usize| {
        lines.get(at).map(|line| line.trim()).filter(|trimmed| {
            !test_mask.get(at).copied().unwrap_or(false)
                && !trimmed.is_empty()
                && !is_comment(trimmed)
        })
    };

    for index in 0..lines.len() {
        let Some(trimmed) = live(index) else {
            continue;
        };

        if trimmed.contains(concat!("clear_", "poison()")) {
            found.push((index, Violation::HandRolledClearPoison));
        }

        if trimmed.contains(concat!(".", "into_inner()"))
            && laundering_context(lines, test_mask, index)
        {
            found.push((index, Violation::IntoInnerRecovery));
        }

        // A chain is analyzed once, from its head. A line beginning with `.` is
        // a continuation of the chain above it and was already read there.
        if trimmed.starts_with('.') {
            continue;
        }
        let mut chain = vec![(index, trimmed)];
        let mut at = index + 1;
        while let Some(next) = live(at) {
            if !next.starts_with('.') {
                break;
            }
            chain.push((at, next));
            at += 1;
        }
        let Some((acquired_at, _)) = chain.iter().copied().find(|(_, text)| acquires_lock(text))
        else {
            continue;
        };
        let joined: String = chain.iter().map(|(_, text)| *text).collect();

        if joined.contains(".unwrap()") || joined.contains(".expect(") {
            found.push((
                acquired_at,
                if chain.len() == 1 {
                    Violation::UnwrapOnAcquire
                } else {
                    Violation::UnwrapAcrossLines
                },
            ));
        }

        // Each of these turns a poisoned lock into an ordinary absent value, so
        // the caller cannot tell a panic from an empty map and the branch is
        // skipped without a report. `is_quarantined` returning false this way
        // hands out a device the process quarantined.
        let discards = joined.contains(concat!(".", "ok()"))
            || joined.contains(".unwrap_or(")
            || joined.contains(concat!(".unwrap_or_", "default()"))
            || joined.contains(".map_or(")
            || joined.contains(concat!(".", "is_ok()"))
            || joined.contains(concat!(".", "is_err()"))
            || joined.contains("if let Ok(")
            || joined.contains("while let Ok(")
            || joined.contains("let Ok(");
        if discards {
            found.push((acquired_at, Violation::SilentDiscard));
        }

        // An inline closure over the poison error renames it and drops the
        // decision: the caller receives a string where a recovery class
        // belongs, and no owner or guarded state is recorded. A named policy
        // function passed to `map_err` is the shape this requires instead, so
        // only a closure literal is reported.
        if joined.contains(".map_err(|") {
            found.push((acquired_at, Violation::AdHocPoisonConversion));
        }
    }

    found
}

/// Whether the line is a comment rather than code.
fn is_comment(trimmed: &str) -> bool {
    trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('*')
}

/// Whether a file is reached only through a `#[cfg(test)]` module declaration.
///
/// A whole file of tests carries no marker of its own, so `cfg_test_lines` sees
/// production code in it and the stem may be anything. The declaring module is
/// where the fact is recorded, so that is where this reads it.
fn declared_under_cfg_test(tree: &Tree, relative_path: &Path) -> bool {
    let (parent, module) = match relative_path.file_name().and_then(|name| name.to_str()) {
        Some("mod.rs") => {
            let dir = relative_path.parent();
            (
                dir.and_then(Path::parent),
                dir.and_then(Path::file_name).and_then(|n| n.to_str()),
            )
        }
        _ => (
            relative_path.parent(),
            relative_path.file_stem().and_then(|n| n.to_str()),
        ),
    };
    let (Some(parent), Some(module)) = (parent, module) else {
        return false;
    };

    ["mod.rs", "lib.rs", "main.rs"].iter().any(|declaring| {
        let Ok(text) = tree.read(&parent.join(declaring)) else {
            return false;
        };
        let lines: Vec<&str> = text.lines().collect();
        lines.iter().enumerate().any(|(at, line)| {
            let trimmed = line.trim();
            let declares = trimmed == format!("mod {module};")
                || trimmed == format!("pub mod {module};")
                || trimmed == format!("pub(crate) mod {module};");
            declares
                && at > 0
                && lines[..at]
                    .iter()
                    .rev()
                    .take_while(|above| above.trim().starts_with('#'))
                    .any(|above| above.contains("cfg(test)"))
        })
    })
}

/// Whether the path is a policy owner, which states a decision rather than
/// calling one.
fn is_policy_owner(path: &str) -> bool {
    POLICY_OWNER_PATHS
        .iter()
        .any(|owner| path == *owner || path.ends_with(owner))
}

/// Gate behavior enforcing that every poisoned lock is answered by a named policy.
pub struct LockPoisonPolicy;

impl GateBehavior for LockPoisonPolicy {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let mut scanned_count = 0;
        let mut outside_roots = 0usize;

        for relative_path in tree.all_rust() {
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

            if is_policy_owner(&path_str) {
                continue;
            }

            if declared_under_cfg_test(&tree, &relative_path) {
                continue;
            }

            let governed_root = state_owners::OWNER_CONTRACT_ROOTS
                .iter()
                .any(|root| path_str.starts_with(root));

            let content = tree.read(&relative_path)?;
            let lines: Vec<&str> = content.lines().collect();
            let test_mask = cfg_test_lines(&lines);
            scanned_count += 1;

            for (index, violation) in violations_in(&lines, &test_mask) {
                if violation == Violation::AdHocPoisonConversion && !governed_root {
                    outside_roots += 1;
                    continue;
                }
                report.find(Finding::at(
                    relative_path.clone(),
                    (index + 1) as u32,
                    format!("{}: `{}`", violation.what(), lines[index].trim()),
                    violation.fix(),
                ));
            }
        }

        let inventoried = state_owners::check(&tree, &mut report)?;

        if outside_roots > 0 {
            report.note(format!(
                "{outside_roots} lock acquisition(s) converted by an inline closure outside \
                 {}; those crates state no recovery contract for their owners yet",
                state_owners::OWNER_CONTRACT_ROOTS.join(" and ")
            ));
        }

        report.cover_complete(
            "production source files scanned for lock governance",
            scanned_count,
        );
        report.cover_complete(
            "mutable state owners closed against a recovery contract",
            inventoried,
        );
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run the production detector over one synthetic file.
    fn detect(source: &str) -> Vec<Violation> {
        let lines: Vec<&str> = source.lines().collect();
        let mask = cfg_test_lines(&lines);
        violations_in(&lines, &mask)
            .into_iter()
            .map(|(_, violation)| violation)
            .collect()
    }

    /// The checkout root.
    ///
    /// A unit test runs with the cwd set to its own crate directory, so reading
    /// the tree from there reaches `xtask` and nothing else. That is a green run
    /// over 151 files reported as a clean workspace. Resolved by walking up from
    /// the working directory rather than from a compiled-in manifest path, which
    /// names whichever checkout last built this unit through the shared target
    /// directory.
    fn workspace_root() -> std::path::PathBuf {
        structure_gate::workspace_root()
    }

    #[test]
    fn lock_poison_policy_gate_reports_clean_on_workspace() {
        let gate = LockPoisonPolicy;
        let ctx = GateCtx::new(workspace_root(), vec![]);
        let report = gate.run(&ctx).expect("gate execution must succeed");

        // A run that reached no file reports zero findings and says nothing. The
        // scan count is what separates a clean tree from an empty one, and only
        // the second is a green run that proves nothing.
        let scanned: usize = report
            .coverage
            .iter()
            .filter(|row| row.subject.contains("scanned for lock governance"))
            .map(|row| row.discovered)
            .sum();
        assert!(
            scanned > 500,
            "the gate must have reached the production tree, scanned {scanned} files"
        );

        assert_eq!(
            report.findings.len(),
            0,
            "production workspace must have 0 unhandled lock poison findings, got: {:?}",
            report.findings
        );
    }

    /// Each category has a source shape the production detector reports. The
    /// match has no catch-all arm, so a new category fails to compile until a
    /// shape is recorded for it.
    #[test]
    fn every_violation_category_is_detected() {
        for category in Violation::ALL {
            let source = match category {
                Violation::UnwrapOnAcquire => "let guard = self.state.lock().unwrap();",
                Violation::UnwrapAcrossLines => {
                    "let guard = self\n    .very_long_receiver\n    .state\n    .lock()\n    .expect(\"poisoned\");"
                }
                Violation::IntoInnerRecovery => {
                    "let guard = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());"
                }
                Violation::SilentDiscard => {
                    "if let Ok(mut guard) = self.state.lock() { guard.push(item); }"
                }
                Violation::HandRolledClearPoison => "self.state.clear_poison();",
                Violation::AdHocPoisonConversion => {
                    "let guard = self.state.lock().map_err(|error| Error::State(error.to_string()))?;"
                }
            };
            assert!(
                detect(source).contains(category),
                "{category:?} must be reported for its own source shape, got {:?}",
                detect(source)
            );
        }
    }

    /// A chain `rustfmt` split is the layout a line-at-a-time rule cannot read,
    /// and it is how a poisoned lock reaches a caller as an ordinary `false`.
    #[test]
    fn a_split_chain_that_fails_open_is_reported() {
        let split = "\
pub fn is_quarantined(&self, lease_id: &str) -> bool {
    self.quarantined_leases
        .lock()
        .map(|set| set.contains(lease_id))
        .unwrap_or(false)
}
";
        assert_eq!(detect(split), vec![Violation::SilentDiscard]);
    }

    /// A governed call site is the shape the tree is required to use, so the
    /// gate reports nothing on it. Each line here is a real shape from the
    /// migrated tree, including the three that only look like violations: a
    /// `Cell::into_inner` far from any lock, an `if let Ok` over a result that
    /// is not a lock, and a `map_err` handed a named policy function rather
    /// than a closure that renames the poison.
    #[test]
    fn governed_call_sites_are_not_reported() {
        let governed = "\
let guard = govern_mutex_restartable(&self.state, OWNER, STATE, HashMap::clear);
let table = reclaim_poisoned_write(&self.table, OWNER, STATE);
let state = reclaim_poisoned_condvar_wait(&ready, &lock, state, OWNER, STATE);
let held = reclaim_poisoned_for_teardown(self.state.lock(), OWNER, STATE);
let bytes = self.buffer.read().unwrap_or_else(|_| poisoned_buffer_byte_lock());
let value = cell.into_inner();
if let Ok(text) = std::fs::read_to_string(path) { use_it(text); }
let named = self.state.lock().map_err(BackendError::poisoned_lock)?;
";
        assert_eq!(
            detect(governed),
            Vec::new(),
            "a governed call site must not be reported"
        );
    }

    /// Test code states its own policy, so the mask keeps it out of the report.
    #[test]
    fn cfg_test_code_is_masked() {
        let source = "\
fn production() {}

#[cfg(test)]
mod tests {
    #[test]
    fn takes_the_lock() {
        let guard = SHARED.lock().unwrap();
    }
}
";
        assert_eq!(detect(source), Vec::new(), "cfg(test) code must be masked");
    }

    /// Every owner named in the list is a real file, so a rename cannot leave a
    /// stale entry silently exempting nothing while its file is scanned.
    #[test]
    fn every_policy_owner_path_exists() {
        let root = workspace_root();
        for owner in POLICY_OWNER_PATHS {
            assert!(
                root.join(owner).is_file(),
                "policy owner {owner} does not exist"
            );
        }
    }
}
