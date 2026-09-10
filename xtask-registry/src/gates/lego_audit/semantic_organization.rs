//! Check 11: semantic ownership, placement, and consolidation closure.
//!
//! Composition is judged from the live program and registration metadata. Source
//! similarity remains useful for discovery, but it never establishes ownership.
//!
//! Placement is judged against the crate family the operation id names, and the
//! family is resolved from the workspace roster on every run. The Category A
//! owner is a facade crate plus one crate per domain partition, so the rule
//! that named `vyre-libs/src` judged the facade alone: every domain directory
//! moved under `vyre-libs-<domain>/src/`, which reported all 349 registrations
//! as living outside their owning crate and left the file-role closure and the
//! block-skeleton scan running over the one `lib.rs` the facade still holds.

use super::*;
use std::path::{Path, PathBuf};

use structure_gate::{CATEGORY_A_CRATE, CATEGORY_C_CRATE};
use syn::punctuated::Punctuated;

/// Organization role of a production file in the Category A crate family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum FileRole {
    /// Registered operation implementation.
    OperationImplementation,
    /// Shared semantic builder consumed by registered operations.
    SharedBuilder,
    /// Whole-graph compositions spanning across domains.
    WholeGraphComposition,
    /// Domain contract, type, or algorithm helper module.
    DomainContractOrType,
    /// Crate-level plumbing.
    CratePlumbing,
}

/// Files that are crate-level plumbing wherever a family crate holds one: the
/// crate root and the curated re-export surface.
///
/// Two further rows stood here, `fixture_bytes.rs` and `test_parity_oracles.rs`,
/// and no file in the family answered to either. A classifier arm that matches
/// nothing reads as a reviewed placement for a file shape the tree does not
/// have; without the arm such a file arrives with no role and fails closure
/// until someone records a decision for it.
const CRATE_PLUMBING_FILES: [&str; 2] = ["lib.rs", "prelude.rs"];

/// Directory holding what a composition of any domain needs around the IR it
/// builds. The `intern` row beside it named no directory and was removed for
/// the same reason as the two plumbing files above.
const CRATE_PLUMBING_DIR: &str = "plumbing";

/// Directory of the compositions that span domains rather than belonging to one.
const WHOLE_GRAPH_DIR: &str = "graph_compositions";

/// Directory, and single-file spelling, of a family crate's shared builder.
///
/// The role is the shared builder module of a family crate, so it is the
/// `builder` module at the crate root and nothing else. Matching any file
/// *named* `builder.rs` gave the role to five domain-local helpers nested
/// inside `nn`, `parsing`, `pattern` and `rule`, which are domain algorithm
/// modules and are classified as such.
const SHARED_BUILDER_DIR: &str = "builder";

/// Module names a family crate declares that name plumbing rather than a
/// composition domain. Each is a file role above, so the role rules and the
/// domain roster read one list.
const NON_DOMAIN_MODULES: [&str; 4] = [
    SHARED_BUILDER_DIR,
    CRATE_PLUMBING_DIR,
    WHOLE_GRAPH_DIR,
    "prelude",
];

/// Whether `rel` is the crate root or the curated prelude of its member.
fn is_crate_plumbing(rel: &str, first: &str) -> bool {
    CRATE_PLUMBING_FILES.contains(&rel) || first == CRATE_PLUMBING_DIR
}

/// Whether `rel` is the shared builder module of its member.
fn is_shared_builder(rel: &str, first: &str) -> bool {
    first == SHARED_BUILDER_DIR || rel == "builder.rs"
}

/// Classify the organization roles of one production file, named by its path
/// under its member's `src` directory.
///
/// Returns every matching role so an overlapping classification is reported
/// rather than resolved by arm order. The member crate does not enter the rule:
/// the domain directory carries the role, and which partition crate holds that
/// directory is a packaging decision above it.
///
/// A registered file is an operation implementation and never also a builder or
/// a whole-graph composition, because registering is what the other two roles
/// are defined by not doing. That is read from the registry rather than from
/// the one file name, `registrations.rs`, that used to stand in for it.
pub(super) fn classify_file_roles(
    rel: &str,
    is_registered: bool,
    domains: &BTreeSet<String>,
) -> Vec<FileRole> {
    let mut roles = Vec::new();
    if is_registered {
        roles.push(FileRole::OperationImplementation);
    }

    let normalized = rel.replace('\\', "/");
    let parts: Vec<&str> = normalized.split('/').collect();
    let first = parts.first().copied().unwrap_or_default();

    let plumbing = is_crate_plumbing(&normalized, first);
    let whole_graph = first == WHOLE_GRAPH_DIR;
    let shared_builder = is_shared_builder(&normalized, first);

    if plumbing {
        roles.push(FileRole::CratePlumbing);
    }
    if whole_graph && !is_registered {
        roles.push(FileRole::WholeGraphComposition);
    }
    if shared_builder && !is_registered {
        roles.push(FileRole::SharedBuilder);
    }
    if domains.contains(first) && !is_registered && !plumbing && !whole_graph && !shared_builder {
        roles.push(FileRole::DomainContractOrType);
    }

    roles
}

/// Judge semantic ownership in both directions: every attributed child must
/// exist, every operation with the same semantic body must have one owner, and
/// every file in the Category A family must have exactly one mechanically
/// checkable role.
pub(super) fn check_semantic_organization(report: &mut Report, ops: &[OpInfo]) -> usize {
    report.note("Semantic ownership, placement, and consolidation".to_string());
    let mut findings = Vec::new();
    let known = ops.iter().map(|op| op.id.as_str()).collect::<BTreeSet<_>>();
    let registered_sources = ops
        .iter()
        .map(|op| op.source_file.as_str())
        .collect::<BTreeSet<_>>();

    let Some(root) = workspace_root() else {
        report.find(Finding::new(
            "workspace root is not reachable",
            "run from the vyre workspace checkout root",
        ));
        return 1;
    };
    let roots = category_a_source_roots(&root);
    if roots.is_empty() {
        findings.push(Finding::new(
            format!("the workspace roster names no `{CATEGORY_A_CRATE}` crate"),
            "restore the Category A ownership family in the root manifest; a placement rule with no owning crate resolves every registration as misplaced",
        ));
    }

    for op in ops {
        check_source_placement(op, &roots, &mut findings);
        for node in op.program.entry() {
            check_attribution(op, node, &known, &mut findings);
        }
    }

    for (index, left) in ops.iter().enumerate() {
        for right in ops.iter().skip(index + 1) {
            check_pair(left, right, &mut findings);
        }
    }

    check_family_file_roles(&roots, &registered_sources, &mut findings);

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

/// Every production file in the Category A family carries exactly one role, and
/// every canonical block skeleton is defined once, in a shared builder module.
fn check_family_file_roles(
    roots: &[CategoryASource],
    registered_sources: &BTreeSet<&str>,
    findings: &mut Vec<Finding>,
) {
    let domains = recognized_domains(roots);
    let mut skeleton_homes: BTreeSet<&'static str> = BTreeSet::new();

    for source in roots {
        if !structure_gate::source_scan::carries_rust_source(&source.src) {
            findings.push(Finding::new(
                format!(
                    "Category A member `{}` carries no Rust source under `src`",
                    source.member
                ),
                "restore readable Rust source under the member, or drop it from the workspace roster",
            ));
            continue;
        }

        let files = match rust_files_under(&source.src) {
            Ok(files) => files,
            Err(error) => {
                findings.push(Finding::new(
                    format!(
                        "cannot walk production files under `{}/src`: {error}",
                        source.member
                    ),
                    "repair the unreadable path so every production file can be classified",
                ));
                continue;
            }
        };

        for path in &files {
            let rel = match path.strip_prefix(&source.src) {
                Ok(relative) => relative.to_string_lossy().replace('\\', "/"),
                Err(error) => {
                    findings.push(Finding::new(
                        format!(
                            "production file `{}` is outside member source root `{}/src`: {error}",
                            path.display(),
                            source.member
                        ),
                        "keep the authoritative production walk inside the member source root",
                    ));
                    continue;
                }
            };
            let workspace_rel = format!("{}/src/{rel}", source.member);
            let is_registered = registered_sources.contains(workspace_rel.as_str());

            let roles = classify_file_roles(&rel, is_registered, &domains);
            if roles.is_empty() {
                findings.push(Finding::in_file(
                    &workspace_rel,
                    format!(
                        "production file `{workspace_rel}` has no recognized organization role"
                    ),
                    "assign it to an operation, a shared builder, a domain type/contract, or plumbing",
                ));
            } else if roles.len() > 1 {
                findings.push(Finding::in_file(
                    &workspace_rel,
                    format!(
                        "production file `{workspace_rel}` matches multiple conflicting organization roles: {roles:?}"
                    ),
                    "keep exactly one organization role per file",
                ));
            }

            check_block_skeletons(
                path,
                &workspace_rel,
                &rel,
                &mut skeleton_homes,
                findings,
            );
        }
    }

    for skeleton in CANONICAL_BLOCK_SKELETONS {
        if skeleton_homes.contains(skeleton) {
            continue;
        }
        findings.push(Finding::new(
            format!(
                "no `{SHARED_BUILDER_DIR}` module in the Category A family defines the canonical block skeleton `{skeleton}`"
            ),
            "repoint the row at the name the skeleton was renamed to, or delete it: a row naming nothing scans for nothing while reading as coverage of the skeleton",
        ));
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

/// Block skeletons whose one definition belongs to a shared builder module.
///
/// A skeleton is the composition scaffold several domains emit around, so a
/// second definition of one is duplication of the shape the builder owns.
/// [`check_family_file_roles`] holds every row to a definition inside a
/// `builder` module, so a renamed skeleton fails as a dead row rather than
/// quietly leaving the scan looking for a name nothing carries.
pub(super) const CANONICAL_BLOCK_SKELETONS: &[&str] = &[
    "build_indexed_map",
    "strided_accumulate_child",
    "strided_accumulate2_child",
    "strided_writeback_child",
    "ReductionComposer",
    "CsrTraversalComposer",
    "TableStateMachineComposer",
];

/// Judge one file's block skeleton definitions, and record the ones a shared
/// builder module owns.
fn check_block_skeletons(
    path: &Path,
    workspace_rel: &str,
    rel: &str,
    homes: &mut BTreeSet<&'static str>,
    findings: &mut Vec<Finding>,
) {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            // A file this check cannot open is a file whose skeleton
            // definitions went unread, which reads the same as having none.
            findings.push(Finding::in_file(
                workspace_rel,
                format!(
                    "cannot read `{workspace_rel}` to check it for duplicate block skeletons: {error}"
                ),
                "restore the file, or drop it from the tracked tree if it is gone",
            ));
            return;
        }
    };
    let declared = match declared_names(&text) {
        Ok(declared) => declared,
        Err(error) => {
            findings.push(Finding::in_file(
                workspace_rel,
                format!("cannot parse `{workspace_rel}` as Rust: {error}"),
                "keep checked-in Rust source syntactically parseable; an unparsed file is a file whose declarations went unread",
            ));
            return;
        }
    };

    let canonical = is_shared_builder(rel, rel.split('/').next().unwrap_or_default());
    for skeleton in CANONICAL_BLOCK_SKELETONS {
        if !declared.contains(*skeleton) {
            continue;
        }
        if canonical {
            homes.insert(*skeleton);
            continue;
        }
        findings.push(Finding::in_file(
            workspace_rel,
            format!(
                "duplicate block skeleton definition `{skeleton}` outside the shared `{SHARED_BUILDER_DIR}` module"
            ),
            "reuse the canonical block skeleton from `crate::builder::*` instead of duplicating skeleton definitions in domain modules",
        ));
    }
}

/// Names of the functions and types one Rust source declares.
///
/// The scan used to ask whether the text held `fn <name>`, which reads a doc
/// comment, a call split across lines, and `fn <name>_inner` as a definition
/// of `<name>`, and reads a test fixture as production code. A declaration is
/// a syntactic fact, so it is taken from the parse tree, and a source that
/// does not parse is an error rather than a source that declares nothing.
fn declared_names(text: &str) -> Result<BTreeSet<String>, syn::Error> {
    let syntax = syn::parse_file(text)?;
    let mut declared = BTreeSet::new();
    declared_item_names(&syntax.items, &mut declared);
    Ok(declared)
}

/// Names of the functions and types a set of items declares, at any nesting.
///
/// An item no production build compiles is left out: a fixture behind a test
/// gate is not a second definition of anything a shipped binary holds.
fn declared_item_names(items: &[syn::Item], out: &mut BTreeSet<String>) {
    for item in items {
        match item {
            syn::Item::Fn(function) if !is_test_gated(&function.attrs) => {
                out.insert(function.sig.ident.to_string());
            }
            syn::Item::Struct(structure) if !is_test_gated(&structure.attrs) => {
                out.insert(structure.ident.to_string());
            }
            syn::Item::Impl(block) if !is_test_gated(&block.attrs) => {
                for member in &block.items {
                    if let syn::ImplItem::Fn(function) = member {
                        if !is_test_gated(&function.attrs) {
                            out.insert(function.sig.ident.to_string());
                        }
                    }
                }
            }
            syn::Item::Mod(module) if !is_test_gated(&module.attrs) => {
                if let Some((_, inner)) = &module.content {
                    declared_item_names(inner, out);
                }
            }
            _ => {}
        }
    }
}

/// Whether no production build compiles the item these attributes gate.
fn is_test_gated(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("cfg")
            && attr
                .parse_args::<syn::Meta>()
                .is_ok_and(|meta| cfg_requires_test(&meta))
    })
}

/// Whether every configuration the predicate admits has `test` on.
///
/// `all(test, unix)` is test-only because one conjunct is; `any(test, feature =
/// "x")` compiles without `test` and is not. Anything else, a `not(..)`
/// included, is read as reachable without `test`, so an exotic gate leaves its
/// item in the production view rather than waving it through.
fn cfg_requires_test(meta: &syn::Meta) -> bool {
    match meta {
        syn::Meta::Path(path) => path.is_ident("test"),
        syn::Meta::List(list) if list.path.is_ident("all") => list
            .parse_args_with(Punctuated::<syn::Meta, syn::token::Comma>::parse_terminated)
            .is_ok_and(|inner| inner.iter().any(cfg_requires_test)),
        syn::Meta::List(list) if list.path.is_ident("any") => list
            .parse_args_with(Punctuated::<syn::Meta, syn::token::Comma>::parse_terminated)
            .is_ok_and(|inner| !inner.is_empty() && inner.iter().all(cfg_requires_test)),
        _ => false,
    }
}

/// Every composition domain the Category A crate family declares.
///
/// A domain is a module a family crate declares in its own `lib.rs` and that is
/// not one of the plumbing modules, so the roster is read from source on every
/// run and a new domain crate is judged without an edit here. What stood in
/// this place was a frozen list of thirty names, merged with the modules the
/// facade `lib.rs` declared; the facade declares none since the domains moved
/// into partition crates, so every run answered from the frozen list alone.
///
/// A family crate whose `lib.rs` cannot be read or parsed contributes no
/// domain, which leaves every file under it unclassified and fails role
/// closure loudly rather than passing a crate this never read.
pub(super) fn recognized_domains(roots: &[CategoryASource]) -> BTreeSet<String> {
    let mut domains = BTreeSet::new();
    for source in roots {
        let Ok(text) = std::fs::read_to_string(source.src.join("lib.rs")) else {
            continue;
        };
        let Ok(syntax) = syn::parse_file(&text) else {
            continue;
        };
        for item in syntax.items {
            if let syn::Item::Mod(declared) = item {
                let name = declared.ident.to_string();
                if !NON_DOMAIN_MODULES.contains(&name.as_str()) {
                    domains.insert(name);
                }
            }
        }
    }
    domains
}

/// The ownership family a registration path sits in, and the path under that
/// member's `src` directory.
///
/// The Category A family is resolved against the live roster, so a partition
/// crate is a legal home for a `vyre-libs::` registration the run after the
/// manifest names it. Category C is one crate and needs no roster.
fn owning_family<'a>(
    normalized: &'a str,
    roots: &[CategoryASource],
) -> Option<(&'static str, &'a str)> {
    for source in roots {
        let marker = format!("{}/src/", source.member);
        if let Some((_, rest)) = normalized.split_once(marker.as_str()) {
            return Some((CATEGORY_A_CRATE, rest));
        }
    }
    let marker = format!("{CATEGORY_C_CRATE}/src/");
    normalized
        .split_once(marker.as_str())
        .map(|(_, rest)| (CATEGORY_C_CRATE, rest))
}

/// A registration lives in the crate family its operation id names, in a
/// directory named by the domain the id declares.
///
/// Both halves are derived from the id: the first segment is the owning family
/// and the second is the domain, so moving an operation between crates
/// re-derives the rule instead of needing a table. The Category C hardware arm
/// that stood here reported "must live in `vyre-primitives/src/hardware/`"
/// while checking only the crate; the family rule below covers the crate and
/// the domain-segment rule covers the directory, so the claim and the check now
/// agree.
///
/// Three domain aliases stood beside the segment rule, `matching` for `scan`
/// and `quant` and `optim` for `nn`. No registered id declares `matching`, and
/// the `quant` and `optim` operations live under `nn/quant/` and `nn/optim/`,
/// which the segment rule reads directly. All three suppressed nothing.
fn check_source_placement(op: &OpInfo, roots: &[CategoryASource], findings: &mut Vec<Finding>) {
    let Some((owner, declared_domain)) = operation_owner(&op.id) else {
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
    let Some((family, rest)) = owning_family(&normalized, roots) else {
        findings.push(Finding::in_file(
            &op.source_file,
            format!(
                "operation `{}` registration is outside the `{CATEGORY_A_CRATE}` crate family and outside `{CATEGORY_C_CRATE}/src/`",
                op.id
            ),
            "move the registration into the source tree of the crate family its id names",
        ));
        return;
    };

    if family != owner {
        findings.push(Finding::in_file(
            &op.source_file,
            format!(
                "operation `{}` names owner `{owner}` but its registration lives in the `{family}` crate family",
                op.id
            ),
            format!(
                "move the registration into a `{owner}` crate, or rename the operation under `{family}::` when this is where its contract belongs"
            ),
        ));
        return;
    }

    let segments: Vec<&str> = rest
        .split('/')
        .map(|segment| segment.trim_end_matches(".rs"))
        .collect();

    if !segments.contains(&declared_domain) {
        let source_domain = segments.first().copied().unwrap_or_default();
        findings.push(Finding::in_file(
            &op.source_file,
            format!(
                "operation `{}` declares domain `{declared_domain}` but its registration lives in domain `{source_domain}`",
                op.id
            ),
            format!(
                "move the semantic owner into a `{declared_domain}` directory of the `{family}` family, or rename the operation into `{family}::{source_domain}::*` when its effects and contract are domain-specific"
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

    /// The live Category A roster, as every rule under test resolves it.
    fn live_roots() -> Vec<CategoryASource> {
        let roots = category_a_source_roots(&xtask::checkout::checkout_root());
        assert!(
            !roots.is_empty(),
            "the workspace roster must name at least the Category A facade"
        );
        roots
    }

    /// A partition crate of the family paired with a domain it declares and
    /// carries a directory for, chosen from the live roster rather than named
    /// here: the crate name is not the domain name in general, and one crate
    /// declares four of them.
    fn partition_domain(roots: &[CategoryASource], domains: &BTreeSet<String>) -> (String, String) {
        for source in roots {
            if source.member == CATEGORY_A_CRATE {
                continue;
            }
            for domain in domains {
                if source.src.join(domain).is_dir() {
                    return (source.member.clone(), domain.clone());
                }
            }
        }
        panic!("no partition crate in the Category A family carries a declared domain directory");
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

        let distinct = fixture("vyre-libs::math::distinct", 8);
        findings.clear();
        check_pair(&same_domain_left, &distinct, &mut findings);
        assert!(findings.is_empty());
    }

    /// WHY: the domain the id declares must be a directory on the registration's
    /// route, and the family the id names decides which crates that route may
    /// run through. The subject is the live roster and the live domain roster,
    /// so a partition crate added tomorrow is covered without an edit here.
    #[test]
    fn a_registration_in_a_partition_crate_is_placed_by_its_own_domain() {
        let roots = live_roots();
        let domains = recognized_domains(&roots);
        let (member, domain) = partition_domain(&roots, &domains);

        let mut op = fixture("vyre-libs::placeholder::sum", 1);
        op.id = format!("{CATEGORY_A_CRATE}::{domain}::sum");
        op.source_file = format!("{member}/src/{domain}/sum.rs");
        let mut findings = Vec::new();
        check_source_placement(&op, &roots, &mut findings);
        assert!(
            findings.is_empty(),
            "a registration in the partition crate that carries its domain is placed: {findings:?}"
        );

        op.source_file = format!("{member}/src/lib.rs");
        let mut findings = Vec::new();
        check_source_placement(&op, &roots, &mut findings);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0]
            .message
            .contains(&format!("declares domain `{domain}`")));
    }

    /// WHY: the id's first segment is the owning family and nothing else decides
    /// it. Reporting only registrations outside both trees let a `vyre-libs::`
    /// operation register from `vyre-primitives/src/` unremarked, which is a
    /// composition claiming a hardware contract's home.
    #[test]
    fn a_registration_in_the_wrong_family_is_reported_in_both_directions() {
        let roots = live_roots();

        let mut composition = fixture("vyre-libs::math::sum", 1);
        composition.source_file = format!("{CATEGORY_C_CRATE}/src/math/sum.rs");
        let mut findings = Vec::new();
        check_source_placement(&composition, &roots, &mut findings);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0]
            .message
            .contains(&format!("lives in the `{CATEGORY_C_CRATE}` crate family")));

        let mut intrinsic = fixture("vyre-primitives::hardware::fma", 1);
        intrinsic.source_file = format!("{CATEGORY_A_CRATE}/src/hardware/fma.rs");
        let mut findings = Vec::new();
        check_source_placement(&intrinsic, &roots, &mut findings);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0]
            .message
            .contains(&format!("lives in the `{CATEGORY_A_CRATE}` crate family")));
    }

    /// WHY: the hardware arm this replaced named `vyre-primitives/src/hardware/`
    /// in its message and checked only the crate. A Category C operation
    /// registered outside its declared domain directory has to stay a finding,
    /// or the claim outlives the check.
    #[test]
    fn a_hardware_operation_outside_its_domain_directory_is_reported() {
        let roots = live_roots();
        let mut op = fixture("vyre-primitives::hardware::fma", 1);
        op.source_file = format!("{CATEGORY_C_CRATE}/src/scan/fma.rs");
        let mut findings = Vec::new();
        check_source_placement(&op, &roots, &mut findings);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].message.contains("declares domain `hardware`"));
    }

    /// WHY: an unreadable walk leaves production files unclassified. Dropping
    /// the walk error would let the semantic ownership gate report a clean tree
    /// for a subject universe it never observed.
    #[test]
    fn production_file_walk_fails_closed() {
        let root = PathBuf::from("/path/that/does/not/exist/vyre-libs/src");
        assert!(rust_files_under(&root).is_err());
    }

    /// WHY: the domain roster is what makes a domain directory classifiable, and
    /// it is read from the family crates' own `lib.rs`. A frozen list answered
    /// every run after the domains moved into partition crates, so a domain
    /// declared by a crate the list never named had no role at all.
    #[test]
    fn the_domain_roster_is_read_from_each_family_crate() {
        let fixture_root = tempfile::tempdir().expect("temporary family root");
        let src = fixture_root.path().join("src");
        std::fs::create_dir_all(&src).expect("member src");
        std::fs::write(
            src.join("lib.rs"),
            "pub mod fresh_domain;\npub mod builder;\npub mod plumbing;\npub mod prelude;\npub mod graph_compositions;\n",
        )
        .expect("member lib.rs");

        let roots = vec![CategoryASource {
            member: "vyre-libs-fresh".to_string(),
            src,
        }];
        let domains = recognized_domains(&roots);
        assert_eq!(
            domains,
            BTreeSet::from(["fresh_domain".to_string()]),
            "a declared module is a domain unless it is one of the plumbing modules"
        );

        let live = recognized_domains(&live_roots());
        assert!(
            live.len() > 1,
            "the live family declares more than one domain: {live:?}"
        );
        assert!(
            NON_DOMAIN_MODULES
                .iter()
                .all(|plumbing| !live.contains(*plumbing)),
            "no plumbing module is a domain: {live:?}"
        );
    }

    /// WHY: role closure is the whole file universe of the family, derived by
    /// walking the live roster. Judging a workspace-relative path against a
    /// single `vyre-libs/src/` prefix classified every partition file as
    /// nothing, and the closure passed because the facade holds one file.
    #[test]
    fn every_production_file_in_the_family_has_exactly_one_role() {
        let roots = live_roots();
        let domains = recognized_domains(&roots);
        let mut report = Report::clean();
        let ops = collect_ops(&mut report);
        let registered = ops
            .iter()
            .map(|op| op.source_file.as_str())
            .collect::<BTreeSet<_>>();
        let mut judged = 0usize;
        let mut wrong = Vec::new();
        for source in &roots {
            for path in rust_files_under(&source.src).expect("member source walk") {
                let rel = path
                    .strip_prefix(&source.src)
                    .expect("member-relative path")
                    .to_string_lossy()
                    .replace('\\', "/");
                let workspace_rel = format!("{}/src/{rel}", source.member);
                let roles = classify_file_roles(
                    &rel,
                    registered.contains(workspace_rel.as_str()),
                    &domains,
                );
                if roles.len() != 1 {
                    wrong.push(format!("{workspace_rel} {roles:?}"));
                }
                judged += 1;
            }
        }
        assert!(wrong.is_empty(), "{wrong:?}");
        assert!(
            judged > 1,
            "the family walk must reach more than the facade's `lib.rs`, reached {judged}"
        );
        assert!(
            !registered.is_empty(),
            "the live registry must name the source files the closure reads as operations"
        );
    }

    /// WHY: the shared-builder role is a family crate's `builder` module. Giving
    /// it to any file named `builder.rs` claimed five domain-local helpers, one
    /// of them four directories inside `nn`, were the shared builder.
    #[test]
    fn the_shared_builder_role_is_the_builder_module_of_a_family_crate() {
        let domains = recognized_domains(&live_roots());
        let domain = domains
            .iter()
            .next()
            .expect("the live family declares a domain")
            .clone();

        assert_eq!(
            classify_file_roles("builder/elementwise.rs", false, &domains),
            vec![FileRole::SharedBuilder]
        );
        assert_eq!(
            classify_file_roles("builder.rs", false, &domains),
            vec![FileRole::SharedBuilder]
        );
        assert_eq!(
            classify_file_roles(&format!("{domain}/layer/builder.rs"), false, &domains),
            vec![FileRole::DomainContractOrType],
            "a builder nested inside a domain is that domain's module"
        );
        assert_eq!(
            classify_file_roles("builder/registrations.rs", true, &domains),
            vec![FileRole::OperationImplementation],
            "a registered file under `builder/` is an operation implementation and nothing else"
        );
        assert_eq!(
            classify_file_roles("lib.rs", false, &domains),
            vec![FileRole::CratePlumbing]
        );
        assert_eq!(
            classify_file_roles("dumping_ground.rs", false, &domains),
            vec![],
            "a file in no domain and no plumbing role fails closure"
        );
    }

    /// WHY: fail-by-default requires a newly added unclassified file to fail
    /// role closure, at the crate root and nested under an unrecognized name.
    #[test]
    fn a_file_outside_every_declared_domain_fails_role_closure() {
        let domains = recognized_domains(&live_roots());
        assert!(!domains.contains("not_a_domain"));
        assert!(classify_file_roles("not_a_domain/new_copy.rs", false, &domains).is_empty());
        assert!(classify_file_roles("unauthorized_root_copy.rs", false, &domains).is_empty());
    }

    /// WHY: role closure rejects any file carrying more than one organization
    /// class. Registering an operation out of crate plumbing is the overlap
    /// that remains, because the plumbing role is decided by the path alone:
    /// a `lib.rs` or a file under `plumbing/` that the registry names is both
    /// an operation implementation and plumbing, and the placement has to be
    /// decided rather than resolved by arm order. The overlap this replaces,
    /// a registered file under `builder/`, is now impossible by construction:
    /// the builder and whole-graph roles are defined by not being registered.
    #[test]
    fn overlapping_file_roles_fails_role_closure() {
        let domains = recognized_domains(&live_roots());
        assert_eq!(
            classify_file_roles("lib.rs", true, &domains),
            vec![
                FileRole::OperationImplementation,
                FileRole::CratePlumbing
            ]
        );
        assert_eq!(
            classify_file_roles(
                &format!("{CRATE_PLUMBING_DIR}/registration.rs"),
                true,
                &domains
            ),
            vec![
                FileRole::OperationImplementation,
                FileRole::CratePlumbing
            ]
        );
        assert_eq!(
            classify_file_roles("builder/registrations.rs", true, &domains),
            vec![FileRole::OperationImplementation],
            "the builder role is defined by not being registered, so it cannot overlap"
        );
    }

    /// WHY: a skeleton definition is a syntactic fact. Asking whether the text
    /// held `fn <name>` counted a doc comment, a longer name with the same
    /// prefix, and a test fixture as definitions, and the subject set is the
    /// canonical skeleton list rather than one name chosen here.
    #[test]
    fn a_skeleton_definition_is_read_from_the_parse_tree() {
        for skeleton in CANONICAL_BLOCK_SKELETONS {
            let near_miss = format!(
                "//! Callers use `fn {skeleton}` from the builder.\n\
                 pub fn {skeleton}_inner() {{}}\n\
                 #[cfg(test)]\n\
                 mod tests {{\n    pub fn {skeleton}() {{}}\n    pub struct {skeleton};\n}}\n"
            );
            let declared = declared_names(&near_miss).expect("fixture parses");
            assert!(
                !declared.contains(*skeleton),
                "`{skeleton}` is declared by none of a doc comment, a longer name, or a test module: {declared:?}"
            );

            let real = format!("pub fn {skeleton}() {{}}\npub struct Other;\n");
            let declared = declared_names(&real).expect("fixture parses");
            assert!(declared.contains(*skeleton), "{declared:?}");
        }

        assert!(
            declared_names("fn broken( {").is_err(),
            "a source that does not parse is an error, never a source that declares nothing"
        );
    }

    /// WHY: the duplicate finding is what the skeleton rows exist for, and it
    /// has to name the file it read rather than the member root.
    #[test]
    fn duplicate_block_skeleton_fails_closure() {
        let skeleton = CANONICAL_BLOCK_SKELETONS
            .first()
            .expect("at least one canonical block skeleton");
        let temp = tempfile::tempdir().expect("temporary dir");
        let copy = temp.path().join("math_copy.rs");
        std::fs::write(
            &copy,
            format!("pub fn {skeleton}() -> vyre::ir::Program {{ unreachable!() }}\n"),
        )
        .expect("write duplicate");

        let mut homes = BTreeSet::new();
        let mut findings = Vec::new();
        check_block_skeletons(
            &copy,
            "vyre-libs-math/src/math/math_copy.rs",
            "math/math_copy.rs",
            &mut homes,
            &mut findings,
        );
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0]
            .message
            .contains(&format!("duplicate block skeleton definition `{skeleton}`")));
        assert!(homes.is_empty());

        let mut findings = Vec::new();
        check_block_skeletons(
            &copy,
            "vyre-libs-builder/src/builder/mod.rs",
            "builder/mod.rs",
            &mut homes,
            &mut findings,
        );
        assert!(findings.is_empty(), "{findings:?}");
        assert_eq!(homes, BTreeSet::from([*skeleton]));
    }

    /// WHY: a skeleton row naming a name no builder module defines scans for
    /// nothing while reading as coverage of the skeleton. The subject set is
    /// the row list and the evidence is the live family, so renaming a skeleton
    /// turns this red instead of silently retiring the row.
    #[test]
    fn every_canonical_block_skeleton_is_defined_in_a_shared_builder_module() {
        let roots = live_roots();
        let mut homes = BTreeSet::new();
        let mut findings = Vec::new();
        for source in &roots {
            for path in rust_files_under(&source.src).expect("member source walk") {
                let rel = path
                    .strip_prefix(&source.src)
                    .expect("member-relative path")
                    .to_string_lossy()
                    .replace('\\', "/");
                if !is_shared_builder(&rel, rel.split('/').next().unwrap_or_default()) {
                    continue;
                }
                check_block_skeletons(
                    &path,
                    &format!("{}/src/{rel}", source.member),
                    &rel,
                    &mut homes,
                    &mut findings,
                );
            }
        }
        assert!(findings.is_empty(), "{findings:?}");
        let rows = CANONICAL_BLOCK_SKELETONS
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        assert_eq!(
            homes, rows,
            "every canonical block skeleton is defined in a `builder` module of the family"
        );
    }
}
