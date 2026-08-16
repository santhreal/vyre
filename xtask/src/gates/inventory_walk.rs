//! Where a link-time registry may be walked.
//!
//! `inventory::iter` walks every registration the binary linked. The walk itself
//! is not the defect: a registry has to be read once to build an index. The
//! defect is walking it again on each lookup, which turns a probe into a scan
//! that grows with the number of registered items, and that is a difference of
//! placement rather than of text.
//!
//! So the rule is structural: an occurrence must sit inside a construction that
//! runs once, a `LazyLock` or `OnceLock` initializer or a function such an
//! initializer names. Anywhere else it runs per call. The previous rule read
//! lines and allowed whole files, which reported the six occurrences inside the
//! one-time builder it had already exempted and stayed silent about the lookups
//! that actually scanned.

use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::{self, Tree};

/// Source roots whose registries are read on a dispatch or lookup path.
const ROOTS: &[&str] = &[
    "vyre-driver/src",
    "vyre-driver-cuda/src",
    "vyre-driver-spirv/src",
    "vyre-driver-wgpu/src",
    "vyre-foundation/src",
    "vyre-libs/src",
    "vyre-primitives/src",
    "vyre-runtime/src",
];

/// How a once-only construction is spelled.
const FREEZE_MARKERS: &[&str] = &["LazyLock::new(", "OnceLock", "get_or_init(", "OnceCell"];

/// Registry walks that run per lookup rather than once.
pub struct InventoryWalk;

impl Gate for InventoryWalk {
    fn name(&self) -> &'static str {
        "hot-path-inventory"
    }

    fn help(&self) -> &'static str {
        "inventory::iter outside a once-only index construction"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let files = tree.rust(ROOTS)?;
        let mut findings = Vec::new();
        let mut walks = 0usize;
        for path in &files {
            if scan::is_test_tree(path) {
                continue;
            }
            let text = tree.read(path)?;
            let (counted, unfrozen) = walks_in(&text);
            walks += counted;
            findings.extend(unfrozen.into_iter().map(|(line, statement)| {
                Finding::at(
                    PathBuf::from(path),
                    line,
                    format!("registry walk on a lookup path: {statement}"),
                    "build the index once behind a LazyLock or OnceLock and probe it here, so a \
                     lookup does not grow with the registration count",
                )
            }));
        }
        let mut report = Report::with_findings(findings);
        report.notes.push(format!(
            "{walks} registry walk(s) in {} production file(s)",
            files.len()
        ));
        Ok(report)
    }
}

/// Every walk in one file, and the ones that run per lookup.
fn walks_in(text: &str) -> (usize, Vec<(u32, String)>) {
    let lines: Vec<&str> = text.lines().collect();
    let test_only = scan::cfg_test_lines(&lines);
    let frozen = frozen_regions(&lines);
    let mut counted = 0usize;
    let mut unfrozen = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let code = scan::scan_code(line);
        if !code.code.contains("inventory::iter") {
            continue;
        }
        if test_only.get(index).copied().unwrap_or(false) {
            continue;
        }
        counted += 1;
        if frozen.get(index).copied().unwrap_or(false) {
            continue;
        }
        let line_number = u32::try_from(index + 1).unwrap_or(u32::MAX);
        unfrozen.push((line_number, scan::statement_at(&lines, index)));
    }
    (counted, unfrozen)
}

/// One flag per line: whether code there runs at most once per process.
///
/// A `static` initializer is frozen where it stands, a function is frozen when a
/// `LazyLock` or `OnceLock` initializer names it, and a function is frozen when
/// frozen code calls it: a builder split across three helpers reads the registry
/// exactly as once as one that inlines them.
fn frozen_regions(lines: &[&str]) -> Vec<bool> {
    let declared = declared_names(lines);
    let mut named = frozen_initializer_names(lines);
    let mut frozen = regions_of(lines, &named);
    for _ in 0..declared.len() {
        let reached: BTreeSet<String> = called_names(lines, &frozen)
            .intersection(&declared)
            .cloned()
            .collect();
        if reached.is_subset(&named) {
            break;
        }
        named.extend(reached);
        frozen = regions_of(lines, &named);
    }
    frozen
}

/// One flag per line, for the initializers and the named functions given.
///
/// A region stays open until its body has opened and closed again, because a
/// signature that wraps its parameters over several lines carries no brace on the
/// line that declares it.
fn regions_of(lines: &[&str], named: &BTreeSet<String>) -> Vec<bool> {
    let mut frozen = vec![false; lines.len()];
    let mut open = None;
    let mut body_started = false;
    let mut depth = 0i32;
    for (index, line) in lines.iter().enumerate() {
        let code = scan::scan_code(line);
        let opens_frozen = FREEZE_MARKERS
            .iter()
            .any(|marker| code.code.contains(marker))
            || declared_name(&code.code).is_some_and(|name| named.contains(&name));
        if opens_frozen && open.is_none() {
            open = Some(depth);
            body_started = false;
        }
        if open.is_some() {
            frozen[index] = true;
        }
        depth += code.brace_delta;
        if let Some(open_depth) = open {
            if depth > open_depth {
                body_started = true;
            } else if body_started {
                open = None;
            }
        }
    }
    frozen
}

/// Every function this file declares.
fn declared_names(lines: &[&str]) -> BTreeSet<String> {
    lines
        .iter()
        .filter_map(|line| declared_name(&scan::scan_code(line).code))
        .collect()
}

/// Free functions and associated functions called from the frozen lines.
///
/// A method call is excluded: `rows.count()` names a method on a value, not a
/// function this file could have declared, and a file that happens to declare
/// `fn count` must not be exempted by it.
fn called_names(lines: &[&str], frozen: &[bool]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for (index, line) in lines.iter().enumerate() {
        if !frozen.get(index).copied().unwrap_or(false) {
            continue;
        }
        let code = scan::scan_code(line).code;
        let bytes = code.as_bytes();
        let mut start = None;
        for (offset, byte) in bytes.iter().enumerate() {
            let identifier = byte.is_ascii_alphanumeric() || *byte == b'_';
            if identifier {
                start = start.or(Some(offset));
                continue;
            }
            if let Some(from) = start.take() {
                if *byte == b'(' && !is_method_call(&code, from) {
                    names.insert(code[from..offset].to_owned());
                }
            }
        }
    }
    names
}

/// Whether the identifier starting at `from` is reached through a value.
fn is_method_call(code: &str, from: usize) -> bool {
    code[..from].ends_with('.')
}

/// Function names a once-only initializer delegates to.
///
/// `LazyLock::new(BackendRegistry::build)` names `build`; a closure names
/// nothing, and its body is frozen by position instead.
fn frozen_initializer_names(lines: &[&str]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for line in lines {
        let code = scan::scan_code(line);
        for marker in FREEZE_MARKERS {
            let Some(rest) = code.code.split_once(marker).map(|(_, rest)| rest) else {
                continue;
            };
            let argument = rest.split(')').next().unwrap_or_default().trim();
            if argument.is_empty() || argument.starts_with('|') {
                continue;
            }
            if let Some(name) = argument.rsplit("::").next() {
                let name = name.trim_end_matches(',').trim();
                if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    names.insert(name.to_owned());
                }
            }
        }
    }
    names
}

/// The function or method a line declares, when it declares one.
fn declared_name(code: &str) -> Option<String> {
    let rest = code.trim_start();
    let rest = rest.strip_prefix("pub ").unwrap_or(rest);
    let rest = rest
        .strip_prefix("const ")
        .or_else(|| rest.strip_prefix("async "))
        .unwrap_or(rest);
    let name = rest.strip_prefix("fn ")?;
    let name = name.split(['(', '<']).next()?.trim();
    (!name.is_empty()).then(|| name.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::fixture_checkout;

    /// The gate's report for a tree where one file under one root holds `text`.
    ///
    /// Every root has to exist: the rule refuses to scan a path that does not,
    /// which is what keeps a moved directory from reading as a clean run.
    fn run(text: &str) -> Report {
        let owners: Vec<String> = ROOTS.iter().map(|root| format!("{root}/owner.rs")).collect();
        let mut files: Vec<(&str, &str)> = owners
            .iter()
            .map(|path| (path.as_str(), "fn owner() {}\n"))
            .collect();
        files.push(("vyre-driver/src/registry.rs", text));
        let (_temporary, root) = fixture_checkout::checkout(&files);
        InventoryWalk
            .run(&GateCtx::new(root, Vec::new()))
            .expect("the gate reads a fixture tree")
    }

    /// The note a clean run states, for a tree carrying `walks` walks.
    fn note(walks: usize) -> String {
        format!(
            "{walks} registry walk(s) in {} production file(s)",
            ROOTS.len() + 1
        )
    }

    #[test]
    fn a_walk_inside_a_lazy_static_initializer_is_clean() {
        let report = run(
            r"
static INDEX: LazyLock<BTreeMap<&str, &Row>> = LazyLock::new(|| {
    let mut index = BTreeMap::new();
    for row in inventory::iter::<Row> {
        index.insert(row.id, row);
    }
    index
});

pub fn row(id: &str) -> Option<&'static Row> {
    INDEX.get(id).copied()
}
",
        );

        assert!(
            report.findings.is_empty(),
            "a walk that runs once is the prescribed shape: {:?}",
            report.findings
        );
        assert_eq!(report.notes, vec![note(1)]);
    }

    #[test]
    fn a_walk_in_a_function_a_once_lock_names_is_clean() {
        let report = run(
            r"
static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::build);

impl Registry {
    fn build() -> Self {
        let rows: Vec<&Row> = inventory::iter::<Row>.into_iter().collect();
        Self { rows }
    }
}
",
        );

        assert!(
            report.findings.is_empty(),
            "a builder the initializer names runs once: {:?}",
            report.findings
        );
    }

    /// WHY: a builder that outgrew one function is still a builder. Splitting the
    /// index construction into helpers is the normal reason a registry walk sits
    /// several calls below the static, and a rule that only read the function the
    /// initializer names reported every one of those helpers.
    #[test]
    fn a_helper_the_builder_calls_is_frozen_with_it() {
        let report = run(
            r"
static REGISTRY: LazyLock<Registry> = LazyLock::new(|| Registry::build());

impl Registry {
    fn build() -> Self {
        Self { rows: freeze_rows() }
    }
}

fn freeze_rows() -> Vec<&'static Row> {
    inventory::iter::<Row>.into_iter().collect()
}
",
        );

        assert!(
            report.findings.is_empty(),
            "a helper reached only from the builder runs once too: {:?}",
            report.findings
        );
    }

    /// WHY: the first version of this rule closed a region on the line that
    /// declared it whenever that line carried no brace, so every builder whose
    /// parameters wrap over several lines had a body the rule read as a lookup
    /// path. Both real registry freezers in this workspace are written that way.
    #[test]
    fn a_builder_whose_signature_wraps_still_covers_its_body() {
        let report = run(
            r"
static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::build);

impl Registry {
    fn build() -> Self {
        Self {
            rows: freeze_rows(
                &[],
            ),
        }
    }
}

fn freeze_rows(
    ignored: &[u8],
) -> Vec<&'static Row> {
    let _ = ignored;
    inventory::iter::<Row>.into_iter().collect()
}
",
        );

        assert!(
            report.findings.is_empty(),
            "the wrapped freezer body is inside the region: {:?}",
            report.findings
        );
    }

    /// WHY: transitivity is what makes the rule usable, and also what could make
    /// it vacuous. A helper reached from a lookup must still be reported, or a
    /// scan can be hidden one call deep.
    #[test]
    fn a_helper_a_lookup_calls_is_still_a_lookup_path() {
        let report = run(
            r"
pub fn row(id: &str) -> Option<&'static Row> {
    scan_rows().into_iter().find(|row| row.id == id)
}

fn scan_rows() -> Vec<&'static Row> {
    inventory::iter::<Row>.into_iter().collect()
}
",
        );

        assert_eq!(
            report.findings.len(),
            1,
            "the walk below the lookup is reported: {:?}",
            report.findings
        );
    }

    /// WHY: this is the shape the rule exists for, and the one the previous rule
    /// could not see. A lookup that walks the registry is linear in the number of
    /// registrations, so it degrades as the workspace links more of them, which is
    /// exactly when nobody is looking at it.
    #[test]
    fn a_walk_on_a_lookup_path_is_reported_with_its_statement() {
        let report = run(
            r"
pub fn row(id: &str) -> Option<&'static Row> {
    inventory::iter::<Row>().find(|row| row.id == id)
}
",
        );

        assert_eq!(
            report.findings.len(),
            1,
            "one lookup walk is one finding: {:?}",
            report.findings
        );
        let finding = &report.findings[0];
        assert!(
            finding.message.contains("inventory::iter::<Row>().find("),
            "the statement is quoted so the reader sees the scan: {finding:?}"
        );
        assert_eq!(finding.line, Some(3));
    }

    #[test]
    fn a_walk_in_a_test_module_is_not_a_lookup_path() {
        let report = run(
            r"
#[cfg(test)]
mod tests {
    #[test]
    fn every_row_is_registered() {
        assert!(inventory::iter::<Row>.into_iter().count() > 0);
    }
}
",
        );

        assert!(
            report.findings.is_empty(),
            "a test may enumerate the registry it is asserting about: {:?}",
            report.findings
        );
        assert_eq!(
            report.notes,
            vec![note(0)],
            "and it is not counted as a walk either"
        );
    }
}
