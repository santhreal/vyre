//! Semantic ownership, placement, and consolidation closure for registered operations.
//!
//! Composition is judged from the live program and registration metadata. Source
//! similarity remains useful for discovery, but it never establishes ownership.

use super::*;
use std::path::{Path, PathBuf};

/// Organization role of a production file under `vyre-libs/src/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum FileRole {
    /// Registered operation implementation.
    OperationImplementation,
    /// Shared semantic builder consumed by registered operations.
    SharedBuilder,
    /// Domain contract, type, or algorithm helper module.
    DomainContractOrType,
    /// Crate-level plumbing.
    CratePlumbing,
}

/// Classify all organization roles for one production file under `vyre-libs/src/`.
/// Returns all matching roles so overlapping/conflicting classifications are detected.
pub(super) fn classify_file_roles(
    path: &str,
    registered_sources: &BTreeSet<&str>,
) -> Vec<FileRole> {
    let mut roles = Vec::new();
    let normalized = path.replace('\\', "/");

    let is_registered = registered_sources.contains(normalized.as_str());

    // 1. Operation Implementation: explicitly registered in operation registry
    if is_registered {
        roles.push(FileRole::OperationImplementation);
    }

    let Some(rel) = normalized.strip_prefix("vyre-libs/src/") else {
        return roles;
    };
    let parts: Vec<&str> = rel.split('/').collect();
    let filename = parts.last().copied().unwrap_or_default();
    let first = parts.first().copied().unwrap_or_default();

    // 2. Crate-level plumbing
    if matches!(
        rel,
        "lib.rs" | "prelude.rs" | "fixture_bytes.rs" | "test_parity_oracles.rs"
    ) || matches!(first, "intern" | "plumbing")
    {
        roles.push(FileRole::CratePlumbing);
    }

    // 3. Shared builder
    if (first == "builder"
        || matches!(
            filename,
            "builder.rs" | "builders.rs" | "build.rs" | "emit.rs"
        ))
        && filename != "registrations.rs"
    {
        roles.push(FileRole::SharedBuilder);
    }

    // 4. Domain contract, type, or algorithm supporting module
    if is_authorized_domain_contract_or_type(rel)
        && !is_registered
        && first != "builder"
        && first != "plumbing"
        && first != "intern"
        && !matches!(
            rel,
            "lib.rs" | "prelude.rs" | "fixture_bytes.rs" | "test_parity_oracles.rs"
        )
        && !matches!(
            filename,
            "builder.rs" | "builders.rs" | "build.rs" | "emit.rs"
        )
    {
        roles.push(FileRole::DomainContractOrType);
    }

    roles
}

/// Whether a relative path under `vyre-libs/src/` is an authorized domain contract or type.
pub(super) fn is_authorized_domain_contract_or_type(rel_path: &str) -> bool {
    let parts: Vec<&str> = rel_path.split('/').collect();
    let Some(first) = parts.first().copied() else {
        return false;
    };
    is_recognized_domain(first)
}

/// Single-role classification helper for simple lookups.
pub(super) fn classify_file_role(
    path: &str,
    registered_sources: &BTreeSet<&str>,
) -> Option<FileRole> {
    let roles = classify_file_roles(path, registered_sources);
    if roles.len() == 1 {
        Some(roles[0])
    } else {
        None
    }
}

/// Judge semantic ownership in both directions: every attributed child must
/// exist, every operation with the same semantic body must have one owner, and
/// every file in `vyre-libs` must have exactly one mechanically checkable role.
pub(super) fn check_semantic_organization(report: &mut Report, ops: &[OpInfo]) -> usize {
    report.note("[11/11] Semantic ownership, placement, and consolidation".to_string());
    let mut findings = Vec::new();
    let known = ops.iter().map(|op| op.id.as_str()).collect::<BTreeSet<_>>();
    let registered_sources = ops
        .iter()
        .map(|op| op.source_file.as_str())
        .collect::<BTreeSet<_>>();

    for op in ops {
        check_source_placement(op, &mut findings);
        for node in op.program.entry() {
            check_attribution(op, node, &known, &mut findings);
        }
    }

    for (index, left) in ops.iter().enumerate() {
        for right in ops.iter().skip(index + 1) {
            check_pair(left, right, &mut findings);
        }
    }

    check_vyre_libs_file_roles(&registered_sources, &mut findings);

    let count = findings.len();
    for finding in findings {
        report.find(finding);
    }
    if count == 0 {
        report
            .note("  semantic ownership and file roles are closed in both directions".to_string());
    }
    count
}

fn check_vyre_libs_file_roles(registered_sources: &BTreeSet<&str>, findings: &mut Vec<Finding>) {
    let Some(root) = workspace_root() else {
        findings.push(Finding::new(
            "workspace root is not reachable",
            "run from the vyre workspace checkout root",
        ));
        return;
    };
    let libs_src = root.join("vyre-libs/src");
    if !libs_src.is_dir() {
        findings.push(Finding::new(
            format!(
                "production source root `{}` is missing or is not a directory",
                libs_src.display()
            ),
            "restore a readable vyre-libs/src directory before judging file roles",
        ));
        return;
    }

    let files = match rust_files_under(&libs_src) {
        Ok(files) => files,
        Err(error) => {
            findings.push(Finding::new(
                format!(
                    "cannot walk production files under `{}`: {error}",
                    libs_src.display()
                ),
                "repair the unreadable path so every production file can be classified",
            ));
            return;
        }
    };
    for path in &files {
        let rel_path = match path.strip_prefix(&root) {
            Ok(relative) => relative.to_string_lossy().replace('\\', "/"),
            Err(error) => {
                findings.push(Finding::new(
                    format!(
                        "production file `{}` is outside workspace root `{}`: {error}",
                        path.display(),
                        root.display()
                    ),
                    "keep the authoritative production walk inside the workspace root",
                ));
                continue;
            }
        };

        let roles = classify_file_roles(&rel_path, registered_sources);
        if roles.is_empty() {
            findings.push(Finding::in_file(
                &rel_path,
                format!("production file `{rel_path}` has no recognized organization role"),
                "assign it to an operation, a shared builder, a domain type/contract, or plumbing",
            ));
        } else if roles.len() > 1 {
            findings.push(Finding::in_file(
                &rel_path,
                format!(
                    "production file `{rel_path}` matches multiple conflicting organization roles: {roles:?}"
                ),
                "keep exactly one organization role per file (Section 182.12.3)",
            ));
        }

        check_for_duplicate_block_skeletons(path, &rel_path, findings);
    }
}

fn rust_files_under(root: &Path) -> Result<Vec<PathBuf>, walkdir::Error> {
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(root) {
        let entry = entry?;
        if entry.file_type().is_file()
            && entry.path().extension().and_then(|ext| ext.to_str()) == Some("rs")
        {
            files.push(entry.into_path());
        }
    }
    Ok(files)
}

/// Canonical block skeleton names that must live exclusively in `vyre-libs/src/builder/`.
pub(super) const CANONICAL_BLOCK_SKELETONS: &[&str] = &[
    "build_indexed_map",
    "strided_accumulate_child",
    "strided_accumulate2_child",
    "strided_writeback_child",
    "ReductionComposer",
    "CsrTraversalComposer",
    "TableStateMachineComposer",
];

fn check_for_duplicate_block_skeletons(
    path: &Path,
    rel_path: &str,
    findings: &mut Vec<Finding>,
) {
    if rel_path.starts_with("vyre-libs/src/builder/") || rel_path == "vyre-libs/src/builder.rs" {
        return;
    }
    let Ok(text) = std::fs::read_to_string(path) else {
        return;
    };
    for skeleton in CANONICAL_BLOCK_SKELETONS {
        let pattern_fn = format!("fn {skeleton}");
        let pattern_struct = format!("struct {skeleton}");
        if text.contains(&pattern_fn) || text.contains(&pattern_struct) {
            findings.push(Finding::in_file(
                rel_path,
                format!(
                    "duplicate block skeleton definition `{skeleton}` outside canonical `vyre-libs/src/builder/` module"
                ),
                "reuse the canonical block skeleton from `crate::builder::*` instead of duplicating skeleton definitions in domain modules",
            ));
        }
    }
}

pub(super) const RECOGNIZED_DOMAINS: &[&str] = &[
    "analysis",
    "bitset",
    "decode",
    "device",
    "encoding",
    "fixpoint",
    "geom",
    "graph",
    "hash",
    "label",
    "llm",
    "logical",

    "math",
    "nfa",
    "nn",
    "opt",
    "parsing",
    "pattern",
    "predicate",
    "reasoning",
    "reduce",
    "representation",
    "rule",

    "scheduling",
    "security",
    "solvers",
    "text",
    "topology",
    "vfs",
    "visual",
];

pub(super) fn is_recognized_domain(name: &str) -> bool {
    RECOGNIZED_DOMAINS.binary_search(&name).is_ok()
}

fn check_source_placement(op: &OpInfo, findings: &mut Vec<Finding>) {
    let Some((_owner, declared_domain)) = operation_owner(&op.id) else {
        findings.push(Finding::new(
            format!("operation `{}` has no crate and domain namespace", op.id),
            "name it `<owner-crate>::<domain>::<operation>` so its canonical owner is mechanically decidable",
        ));
        return;
    };

    if op.source_file.is_empty() || op.source_file == "<unattributed>" {
        findings.push(Finding::new(
            format!("operation `{}` has no registration source attribution", op.id),
            "construct the registration through the track-caller constructor so the registry records its owning source file",
        ));
        return;
    }

    let normalized = op.source_file.replace('\\', "/");
    let Some((owning_crate, rest)) = normalized
        .split_once("vyre-libs/src/")
        .map(|(_, rest)| ("vyre-libs", rest))
        .or_else(|| {
            normalized
                .split_once("vyre-primitives/src/")
                .map(|(_, rest)| ("vyre-primitives", rest))
        })
    else {
        findings.push(Finding::in_file(
            &op.source_file,
            format!(
                "operation `{}` registration is outside `vyre-libs/src/` or `vyre-primitives/src/`",
                op.id
            ),
            "move the registration into the owning crate's source tree",
        ));
        return;
    };

    if op.tier == Tier::T2 && owning_crate != "vyre-primitives" {
        findings.push(Finding::in_file(
            &op.source_file,
            format!(
                "Category C hardware operation `{}` must live in `vyre-primitives/src/hardware/`",
                op.id
            ),
            "move the Category C hardware operation to `vyre-primitives/src/hardware/`",
        ));
        return;
    }

    let segments: Vec<&str> = rest
        .split('/')
        .map(|segment| segment.trim_end_matches(".rs"))
        .collect();
    let matches_domain = segments.contains(&declared_domain)
        || (declared_domain == "matching" && segments.contains(&"scan"))
        || (declared_domain == "quant" && segments.contains(&"nn"))
        || (declared_domain == "optim" && segments.contains(&"nn"));

    if !matches_domain {
        let source_domain = segments.first().copied().unwrap_or_default();
        findings.push(Finding::in_file(
            &op.source_file,
            format!(
                "operation `{}` declares domain `{declared_domain}` but its registration lives in domain `{source_domain}`",
                op.id
            ),
            format!(
                "move the semantic owner to `{owning_crate}/src/{declared_domain}/`, or rename the operation into `{owning_crate}::{source_domain}::*` when its effects and contract are domain-specific"
            ),
        ));
    }
}

fn operation_owner(id: &str) -> Option<(&str, &str)> {
    let mut segments = id.split("::");
    let owner = segments.next()?;
    let domain = segments.next()?;
    (!owner.is_empty() && !domain.is_empty()).then_some((owner, domain))
}

fn check_attribution(
    op: &OpInfo,
    node: &Node,
    known: &BTreeSet<&str>,
    findings: &mut Vec<Finding>,
) {
    if let Node::Region {
        source_region: Some(parent),
        generator,
        ..
    } = node
    {
        let child = generator.as_str();
        if !known.contains(child) && !vyre_foundation::composition::is_anonymous_generator(child) {
            findings.push(Finding::in_file(
                &op.source_file,
                format!(
                    "operation `{}` attributes a composed region to unregistered child `{child}`",
                    op.id
                ),
                "register the child semantic owner and compose it by operation id, or mark the region anonymous when it owns no reusable operation",
            ));
        }
        if parent.as_str() != op.id && !known.contains(parent.as_str()) {
            findings.push(Finding::in_file(
                &op.source_file,
                format!(
                    "operation `{}` carries unknown composition parent `{}`",
                    op.id,
                    parent.as_str()
                ),
                "preserve the registered parent operation id when transplanting the child region",
            ));
        }
    }
    for body in vyre_foundation::visit::child_bodies(node) {
        for child in body {
            check_attribution(op, child, known, findings);
        }
    }
}

fn check_pair(left: &OpInfo, right: &OpInfo, findings: &mut Vec<Finding>) {
    if left.children.contains(&right.id) || right.children.contains(&left.id) {
        return;
    }

    if left.semantic_fingerprint == right.semantic_fingerprint {
        findings.push(consolidation_finding(
            left,
            right,
            "have byte-identical canonical programs after erasing only the owner id",
        ));
    }
}

fn consolidation_finding(left: &OpInfo, right: &OpInfo, evidence: &str) -> Finding {
    let left_domain = operation_owner(&left.id).map(|(_, domain)| domain);
    let right_domain = operation_owner(&right.id).map(|(_, domain)| domain);
    let direction = if left_domain == right_domain {
        "keep one parameterized semantic owner in that domain and make every caller compose it"
    } else if left.tier == Tier::T3 && right.tier == Tier::T3 {
        "promote the shared semantic body to the lowest common substrate domain and make both domains compose it"
    } else {
        "keep the lowest-level canonical owner and make the higher-level operation compose it"
    };
    Finding::new(
        format!(
            "operations `{}` and `{}` {evidence}, but neither composes the other",
            left.id, right.id
        ),
        direction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre::ir::{BufferDecl, DataType};

    fn fixture(id: &'static str, value: u32) -> OpInfo {
        build_info(
            id,
            Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
                [1, 1, 1],
                vec![Node::store("out", Expr::u32(0), Expr::u32(value))],
            ),
        )
    }
    fn fixture_distinct(id: &'static str) -> OpInfo {
        build_info(
            id,
            Program::wrapped(
                vec![
                    BufferDecl::read("in", 0, DataType::U32).with_count(1),
                    BufferDecl::output("out", 1, DataType::U32).with_count(1),
                ],
                [1, 1, 1],
                vec![Node::store(
                    "out",
                    Expr::u32(0),
                    Expr::load("in", Expr::u32(0)),
                )],
            ),
        )
    }

    /// WHY: exact semantic duplicates are the non-heuristic consolidation class.
    /// A same-domain copy and a cross-domain copy must both fail; differences in
    /// the literal body remain outside this exact-identity assertion.
    #[test]
    fn exact_semantic_duplicates_require_one_owner_in_every_domain_arrangement() {
        let same_domain_left = fixture("vyre-libs::math::left", 7);
        let same_domain_right = fixture("vyre-libs::math::right", 7);
        let cross_domain = fixture("vyre-libs::graph::right", 7);

        let mut findings = Vec::new();
        check_pair(&same_domain_left, &same_domain_right, &mut findings);
        check_pair(&same_domain_left, &cross_domain, &mut findings);
        assert_eq!(findings.len(), 2);
        assert!(findings[0].fix.contains("parameterized semantic owner"));
        assert!(findings[1].fix.contains("common substrate domain"));

        let distinct = fixture_distinct("vyre-libs::math::distinct");
        findings.clear();
        check_pair(&same_domain_left, &distinct, &mut findings);
        assert!(findings.is_empty());
    }

    #[test]
    fn source_domain_must_equal_registered_domain() {
        let mut op = fixture("vyre-libs::math::sum", 1);
        op.source_file = "vyre-libs/src/nn/sum.rs".to_string();
        let mut findings = Vec::new();
        check_source_placement(&op, &mut findings);
        assert_eq!(findings.len(), 1);
        assert!(findings[0].message.contains("declares domain `math`"));
    }

    /// WHY: an unreadable walk leaves production files unclassified. Dropping
    /// the walk error would let the semantic ownership gate report a clean tree
    /// for a subject universe it never observed.
    #[test]
    fn production_file_walk_fails_closed() {
        let root = PathBuf::from("/path/that/does/not/exist/vyre-libs/src");
        assert!(rust_files_under(&root).is_err());
    }

    #[test]
    fn file_roles_classify_every_vyre_libs_file_uniquely() {
        let registered = BTreeSet::from(["vyre-libs/src/math/sin.rs"]);
        assert_eq!(
            classify_file_roles("vyre-libs/src/math/sin.rs", &registered),
            vec![FileRole::OperationImplementation]
        );
        assert_eq!(
            classify_file_roles("vyre-libs/src/builder/elementwise.rs", &registered),
            vec![FileRole::SharedBuilder]
        );
        assert_eq!(
            classify_file_roles("vyre-libs/src/lib.rs", &registered),
            vec![FileRole::CratePlumbing]
        );
        assert_eq!(
            classify_file_roles(
                "vyre-libs/src/graph/csr_frontier_queue/graph_validation.rs",
                &registered
            ),
            vec![FileRole::DomainContractOrType]
        );
        assert_eq!(
            classify_file_roles("vyre-libs/src/dumping_ground.rs", &registered),
            vec![]
        );
    }

    /// WHY: Section 182.12.8 and fail-by-default require a newly added unclassified file to fail role closure.
    #[test]
    fn new_unregistered_nested_file_fails_role_closure() {
        let registered = BTreeSet::from(["vyre-libs/src/math/sin.rs"]);
        let new_file = "vyre-libs/src/foo/new_copy.rs";
        let roles = classify_file_roles(new_file, &registered);
        assert!(
            roles.is_empty(),
            "new unclassified file must have zero recognized roles"
        );
        assert_eq!(classify_file_role(new_file, &registered), None);

        let dumping_ground = "vyre-libs/src/unauthorized_root_copy.rs";
        let root_roles = classify_file_roles(dumping_ground, &registered);
        assert!(
            root_roles.is_empty(),
            "unauthorized root copy must fail role classification"
        );
    }

    /// WHY: Section 182.12.3 requires rejecting any file with more than one organization class.
    #[test]
    fn overlapping_file_roles_fails_role_closure() {
        let registered = BTreeSet::from(["vyre-libs/src/builder/elementwise.rs"]);
        let roles = classify_file_roles("vyre-libs/src/builder/elementwise.rs", &registered);
        assert_eq!(
            roles,
            vec![FileRole::OperationImplementation, FileRole::SharedBuilder],
            "file claiming both operation implementation and shared builder must report overlapping roles"
        );
        assert_eq!(
            classify_file_role("vyre-libs/src/builder/elementwise.rs", &registered),
            None
        );
    }

    /// WHY: block skeletons must have exactly one canonical owner in `builder/`.
    #[test]
    fn duplicate_block_skeleton_fails_closure() {
        let temp = tempfile::tempdir().expect("temporary dir");
        let fake_file = temp.path().join("math_copy.rs");
        std::fs::write(
            &fake_file,
            "pub fn build_indexed_map() -> vyre::ir::Program { todo!() }\n",
        )
        .expect("write fake file");
        let mut findings = Vec::new();
        check_for_duplicate_block_skeletons(
            &fake_file,
            "vyre-libs/src/math/math_copy.rs",
            &mut findings,
        );
        assert_eq!(findings.len(), 1);
        assert!(findings[0]
            .message
            .contains("duplicate block skeleton definition `build_indexed_map`"));
    }
}
