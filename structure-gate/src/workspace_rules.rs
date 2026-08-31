//! Structural rules judging workspace roster, operation identities, category homes,
//! substrate definitions, frontend owners, backend admission, and registry linkage.

use std::collections::{BTreeMap, BTreeSet};

use crate::workspace_manifest::crate_ident;

/// Category A owner: every composition, meaning anything that returns a
/// `Program` built from existing IR, including compiler-internal domains such
/// as solvers, encoding, analysis, scheduling, device and graph dispatch. Who
/// calls it does not move it; only rewriting it in host Rust does.
pub(crate) const CATEGORY_A_CRATE: &str = "vyre-libs";
/// Category C owner: strict hardware intrinsics, one emitter arm and one
/// reference-interpreter arm each. Absorbed the former standalone hardware
/// crate on 2026-08-13; the intrinsics live in `vyre-primitives/src/hardware`.
pub(crate) const CATEGORY_C_CRATE: &str = "vyre-primitives";

/// Directory that owns every module named `*substrate*`.
///
/// `vyre_foundation::pass_substrate` owns the CPU pass math outright: the pass
/// engine imports those functions and wraps them in dispatch rather than
/// reimplementing them. Renaming the pass-engine crate retired the second and
/// third homes for the name, so foundation is the only one left and the
/// exemption list it used to need is gone.
pub(crate) const SUBSTRATE_HOME: &str = "vyre-foundation/src/pass_substrate";

/// Closed workspace roster. A new member is a reviewable change here first.
pub(crate) const ALLOWED_MEMBERS: &[&str] = &[
    "conform/vyre-conform",
    "conform/vyre-conform-spec",
    "vyre",
    "vyre-aot",
    // Sole owner of the registry link anchors: it names every crate that submits
    // into an inventory registry so no consumer has to.
    "vyre-registry-link",
    "vyre-bench",
    "vyre-debug",
    "vyre-driver",
    "vyre-driver-cuda",
    "vyre-driver-metal",
    "vyre-driver-reference",
    "vyre-driver-spirv",
    "vyre-driver-wgpu",
    "vyre-emit-metal",
    "vyre-emit-naga",
    "vyre-emit-ptx",
    "vyre-emit-spirv",
    "vyre-foundation",
    "vyre-libs",
    "vyre-lints",
    "vyre-lower",
    "vyre-macros",
    "vyre-megakernel",
    "vyre-primitives",
    "vyre-reference",
    "vyre-runtime",
    "vyre-safetensors",
    // Narrowed to the optimizer pass engine and renamed with that narrowing.
    "vyre-pass-engine",
    "vyre-spec",
    "vyre-test-support",
    "structure-gate",
    "xtask",
    // The xtask subcommands that link vyre. Split out so a source edit no
    // longer rebuilds the compiler before a text-reading gate can run.
    "xtask-evidence",
    "xtask-registry",
];

/// Source languages and the single crate that owns each frontend.
///
/// A source frontend is a pile of Category A compositions: it parses with
/// vyre operations, so `vyre-libs` owns it like any other composition. A
/// separate CPU pipeline over the same language is the second frontend this
/// rule exists to reject. The tree-sitter C shell that used to be one left
/// the workspace as its own product rather than growing here.
///
/// The rust owner ships outside this workspace, so no member matches it and
/// every workspace crate that grows rust frontend stages is a second frontend.
/// Dropping the row instead would stop judging the language altogether.
pub(crate) const FRONTEND_OWNERS: &[(&str, &str)] =
    &[("c", "vyre-libs"), ("rust", "vyre-frontend-rust")];

/// One registered operation, as read from source text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Registration {
    /// Workspace member that submits the registration.
    pub crate_name: String,
    /// Workspace-relative source file.
    pub file: String,
    /// Resolved operation id, e.g. `vyre-foundation::hash::adler32`.
    pub op_id: String,
    /// `OperationTier` variant named in the registration, when present.
    pub tier: Option<String>,
}

impl Registration {
    /// Crate namespace the operation id claims, e.g. `vyre-foundation`.
    fn claimed_crate(&self) -> &str {
        self.op_id.split("::").next().unwrap_or(&self.op_id)
    }

    /// Trailing segment of the operation id, e.g. `adler32`.
    fn leaf(&self) -> &str {
        self.op_id.rsplit("::").next().unwrap_or(&self.op_id)
    }
}

/// Reject workspace members outside the reviewed roster.
pub fn roster_failures(members: &[String]) -> Vec<String> {
    let mut failures = Vec::new();
    for member in members {
        if !ALLOWED_MEMBERS.contains(&member.as_str()) {
            failures.push(format!(
                "workspace member `{member}` is not on the reviewed roster; a product crate belongs outside this workspace, and a new platform crate is added to ALLOWED_MEMBERS in the same change"
            ));
        }
    }
    for allowed in ALLOWED_MEMBERS {
        if !members.iter().any(|member| member == allowed) {
            failures.push(format!(
                "roster lists `{allowed}` but the workspace does not contain it; delete the stale roster entry"
            ));
        }
    }
    failures
}

/// Reject operation registrations outside the two category owners.
pub fn registration_owner_failures(registrations: &[Registration]) -> Vec<String> {
    registrations
        .iter()
        .filter(|reg| reg.crate_name != CATEGORY_A_CRATE && reg.crate_name != CATEGORY_C_CRATE)
        .map(|reg| {
            format!(
                "{} registers `{}`; only {CATEGORY_A_CRATE} (Category A) and {CATEGORY_C_CRATE} (Category C) own operations",
                reg.file, reg.op_id
            )
        })
        .collect()
}

/// Reject one semantic operation carrying two identities, and ids that name a
/// crate the workspace does not have.
///
/// The namespace of an id is the crate that minted it, frozen from then on.
/// This rule used to require it to equal the crate the registration lives in,
/// which reported all 130 operations that moved to `vyre-libs` keeping their
/// `vyre-primitives::` ids. Where an operation lives is
/// [`registration_owner_failures`] and [`category_home_failures`], both of
/// which read the file the registration is written in. What is left here is
/// what the id itself can answer: two crates must not claim one kernel, and a
/// namespace must name a member the workspace carries.
pub fn operation_identity_failures(
    registrations: &[Registration],
    members: &[String],
) -> Vec<String> {
    let mut failures = Vec::new();
    let mut by_leaf: BTreeMap<&str, Vec<&Registration>> = BTreeMap::new();
    for reg in registrations {
        by_leaf.entry(reg.leaf()).or_default().push(reg);
    }
    for (leaf, regs) in by_leaf {
        let mut namespaces: Vec<&str> = regs.iter().map(|reg| reg.claimed_crate()).collect();
        namespaces.sort_unstable();
        namespaces.dedup();
        if namespaces.len() > 1 {
            let ids: Vec<&str> = regs.iter().map(|reg| reg.op_id.as_str()).collect();
            failures.push(format!(
                "operation `{leaf}` is registered under {} identities ({}); one kernel gets one id, and the higher layer calls it instead of re-registering it",
                ids.len(),
                ids.join(", ")
            ));
        }
    }
    let member_crates: BTreeSet<&str> = members
        .iter()
        .map(|member| member.rsplit('/').next().unwrap_or(member))
        .collect();
    for reg in registrations {
        let claimed = reg.claimed_crate();
        if claimed.starts_with("vyre-") && !member_crates.contains(claimed) {
            failures.push(format!(
                "{} registers `{}`, whose namespace names `{claimed}`; no workspace member carries that name, so the id was minted by a crate that never existed or was renamed without a migration",
                reg.file, reg.op_id
            ));
        }
    }
    failures
}

/// Reject a Category A operation in the Category C crate and the reverse.
///
/// Both sides are read from the tree. The tier is the one the registration
/// declares in its own source, and the home is the crate whose `src` holds the
/// file that registration is written in. Neither is the operation id: the id
/// namespace is frozen at mint time, so 130 operations that moved crate still
/// spell the crate they left.
pub fn category_home_failures(registrations: &[Registration]) -> Vec<String> {
    let mut failures = Vec::new();
    for reg in registrations {
        let Some(tier) = reg.tier.as_deref() else {
            continue;
        };
        let hardware = matches!(tier, "Intrinsic" | "Hardware");
        if hardware
            && reg.crate_name == CATEGORY_A_CRATE
            && !reg.op_id.starts_with("vyre-primitives::")
        {
            failures.push(format!(
                "{} registers Category C `{}` in {CATEGORY_A_CRATE}; hardware-contract operations live in {CATEGORY_C_CRATE}",
                reg.file, reg.op_id
            ));
        }
        if !hardware && reg.crate_name == CATEGORY_C_CRATE {
            failures.push(format!(
                "{} registers Category A `{}` in {CATEGORY_C_CRATE}; a composition over existing IR variants lives in {CATEGORY_A_CRATE}",
                reg.file, reg.op_id
            ));
        }
    }
    failures
}

/// Reject a second home for the substrate concept.
///
/// `vyre-foundation` owns the name. A type, trait, or module that restates it
/// anywhere else is a second definition of one concept, and the two drift
/// silently because nothing compares them.
pub fn substrate_home_failures(paths: &[String]) -> Vec<String> {
    paths
        .iter()
        .filter(|path| !path.starts_with(SUBSTRATE_HOME))
        .map(|path| {
            format!(
                "`{path}` names the substrate concept outside {SUBSTRATE_HOME}; one concept gets one home"
            )
        })
        .collect()
}

/// Reject two crates owning a frontend for one source language.
///
/// A frontend announces itself two ways: a language-named stage directory
/// (`.../c/lex/`), or a crate name that says so (`vyre-frontend-c`). Keying
/// only on the directory missed a flat crate whose whole job was the second
/// frontend, which is the shape this rule exists to catch.
pub fn frontend_owner_failures(paths: &[(String, String)]) -> Vec<String> {
    let mut failures = Vec::new();
    for (language, owner) in FRONTEND_OWNERS {
        for (found_crate, path) in paths {
            if found_crate == owner {
                continue;
            }
            if crate_declares_frontend(found_crate, language) {
                failures.push(format!(
                    "`{found_crate}` is a second {language} frontend crate; {owner} owns the {language} frontend"
                ));
            } else if path_names_language(path, language) {
                failures.push(format!(
                    "`{path}` puts a {language} frontend in {found_crate}; {owner} owns the {language} frontend"
                ));
            }
        }
    }
    failures.sort();
    failures.dedup();
    failures
}

/// True when a crate name declares itself the frontend for `language`.
pub(crate) fn crate_declares_frontend(crate_name: &str, language: &str) -> bool {
    let mut names_frontend = false;
    let mut names_language = false;
    for token in crate_name.split(['-', '_']) {
        names_frontend |= token.eq_ignore_ascii_case("frontend");
        names_language |= token.eq_ignore_ascii_case(language);
    }
    names_frontend && names_language
}

/// Names `vyre-driver` owns for every backend, that a backend must not define.
///
/// `admit` and `admit_modules` are the admission decision itself; the other two
/// are the rejection vocabulary it answers with.
pub(crate) const SHARED_ADMISSION_HELPERS: &[&str] =
    &["invalid_module", "compile_error", "admit", "admit_modules"];

/// Call spellings that route a backend through the shared admission decision.
///
/// `admit_modules` is the descriptor-bound form: it calls `admit` and then
/// decodes each admitted module in the backend's own dialect, which is the only
/// part of materialization that differs per target. Accepting the bare `admit`
/// as well keeps a backend that needs the admitted list without the decode
/// callback inside the rule.
pub(crate) const SHARED_ADMISSION_CALLS: &[&str] = &["materialize::admit(", "admit_modules("];

/// Reject a concrete backend that decides target-payload admission by itself.
///
/// Admitting a payload is a property of the neutral artifact and the payload
/// envelope, so it is identical for every target. It was nonetheless written
/// once per backend, and the copies drifted until a payload two backends
/// rejected was accepted by the other two. `vyre_driver::materialize` is the
/// single decision; a backend that reimplements it has reopened that class.
pub fn materializer_admission_failures(materializers: &[(String, String)]) -> Vec<String> {
    let mut failures = Vec::new();
    for (path, text) in materializers {
        for helper in SHARED_ADMISSION_HELPERS {
            if text.contains(&format!("fn {helper}(")) {
                failures.push(format!(
                    "`{path}` defines its own `{helper}`; call `vyre_driver::materialize::{helper}` instead"
                ));
            }
        }
        if !SHARED_ADMISSION_CALLS
            .iter()
            .any(|call| text.contains(call))
        {
            failures.push(format!(
                "`{path}` does not admit its target payload through `vyre_driver::materialize`"
            ));
        }
    }
    failures
}

/// Crate that owns every registry link anchor in this workspace.
pub(crate) const REGISTRY_LINK_OWNER: &str = "vyre-registry-link";

/// One `use <crate> as _;` read from member sources.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscardingImport {
    /// Workspace-relative source file.
    pub file: String,
    /// Crate identifier as the import writes it, e.g. `vyre_libs`.
    pub named: String,
}

/// Reject a discarding import that names a crate submitting inventory registrations.
///
/// An `inventory` registration lives in the object file of the declaring crate,
/// and a linker keeps an archive member out of an rlib only when a symbol inside
/// it is referenced. `use vyre_libs as _;` names the crate and references no
/// symbol, so the registrations were dropped from every binary that did not
/// otherwise call into that crate: the production binary saw all 354 operation
/// registrations while three registry rules iterated an empty registry and
/// passed. A `const` backend id is no anchor either, because it inlines at the
/// use site. Reading the registry through `REGISTRY_LINK_OWNER` calls a real
/// function in each source crate, which is what keeps the object file in.
#[must_use]
pub fn registry_link_failures(submitters: &[String], imports: &[DiscardingImport]) -> Vec<String> {
    let mut failures = Vec::new();
    for import in imports {
        let Some(submitter) = submitters
            .iter()
            .find(|submitter| crate_ident(submitter) == import.named)
        else {
            continue;
        };
        failures.push(format!(
            "`{}` names `{submitter}` with `use {} as _;`, which references no symbol in it, so the linker drops that crate's inventory registrations and every registry read in this binary judges a partial set; read the registry through `{REGISTRY_LINK_OWNER}` instead",
            import.file, import.named
        ));
    }
    failures
}
pub(crate) fn path_names_language(path: &str, language: &str) -> bool {
    path.split(['/', '.'])
        .any(|segment| segment.eq_ignore_ascii_case(language))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registration(crate_name: &str, op_id: &str, tier: Option<&str>) -> Registration {
        Registration {
            crate_name: crate_name.to_string(),
            file: format!("{crate_name}/src/op.rs"),
            op_id: op_id.to_string(),
            tier: tier.map(str::to_string),
        }
    }

    #[test]
    fn a_third_registering_crate_is_rejected() {
        let failures = registration_owner_failures(&[registration(
            "vyre-pass-engine",
            "vyre-pass-engine::graph::toposort",
            Some("Library"),
        )]);

        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("vyre-pass-engine"));
    }

    #[test]
    fn the_two_category_owners_are_accepted() {
        let failures = registration_owner_failures(&[
            registration("vyre-libs", "vyre-libs::hash::adler32", None),
            registration(
                "vyre-primitives",
                "vyre-primitives::atomic::compare_exchange",
                None,
            ),
        ]);

        assert!(failures.is_empty(), "{failures:?}");
    }

    /// Every workspace member the identity rule judges against.
    fn roster() -> Vec<String> {
        ALLOWED_MEMBERS
            .iter()
            .map(|name| (*name).to_string())
            .collect()
    }

    #[test]
    fn one_kernel_registered_under_two_namespaces_is_rejected() {
        let failures = operation_identity_failures(
            &[
                registration("vyre-foundation", "vyre-foundation::hash::adler32", None),
                registration("vyre-libs", "vyre-libs::hash::adler32", None),
            ],
            &roster(),
        );

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("2 identities")),
            "{failures:?}"
        );
    }

    #[test]
    fn same_leaf_under_one_namespace_is_accepted() {
        let failures = operation_identity_failures(
            &[
                registration("vyre-foundation", "vyre-foundation::hash::adler32", None),
                registration("vyre-foundation", "vyre-foundation::graph::toposort", None),
            ],
            &roster(),
        );

        assert!(failures.is_empty(), "{failures:?}");
    }

    /// A frozen id keeps the namespace of the crate that minted it.
    ///
    /// Requiring the namespace to equal the crate the registration lives in
    /// reported all 130 operations that moved to `vyre-libs` and kept their
    /// `vyre-primitives::` ids. Where an operation lives is judged by the rules
    /// that read the file it is written in.
    #[test]
    fn an_operation_that_moved_crate_keeps_its_minting_namespace() {
        let failures = operation_identity_failures(
            &[registration(
                "vyre-libs",
                "vyre-primitives::graph::toposort",
                None,
            )],
            &roster(),
        );

        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn an_id_naming_no_workspace_member_is_rejected() {
        let failures = operation_identity_failures(
            &[registration(
                "vyre-libs",
                "vyre-departed::hash::adler32",
                None,
            )],
            &roster(),
        );

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("no workspace member carries that name")),
            "{failures:?}"
        );
    }

    #[test]
    fn a_hardware_operation_in_the_category_a_crate_is_rejected() {
        let failures = category_home_failures(&[registration(
            "vyre-libs",
            "vyre-libs::atomic::compare_exchange",
            Some("Intrinsic"),
        )]);

        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("Category C"));
    }

    #[test]
    fn a_composition_in_the_category_c_crate_is_rejected() {
        let failures = category_home_failures(&[registration(
            "vyre-primitives",
            "vyre-primitives::hash::adler32",
            Some("Library"),
        )]);

        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("Category A"));
    }

    #[test]
    fn each_category_in_its_own_home_is_accepted() {
        let failures = category_home_failures(&[
            registration("vyre-libs", "vyre-libs::hash::adler32", Some("Library")),
            registration(
                "vyre-primitives",
                "vyre-primitives::atomic::compare_exchange",
                Some("Intrinsic"),
            ),
        ]);

        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn a_second_substrate_home_is_rejected() {
        // Illustrative names: the second homes this rule caught in the tree
        // (`vyre-libs/src/substrate_catalog.rs`, `vyre-driver/src/speculation_substrate.rs`)
        // have been renamed, so the fixture keeps the shape rather than a path.
        let failures = substrate_home_failures(&[
            "vyre-foundation/src/pass_substrate/semiring_closure.rs".to_string(),
            "vyre-driver/src/speculation_substrate.rs".to_string(),
            "vyre-libs/src/matmul_substrate.rs".to_string(),
        ]);

        assert_eq!(failures.len(), 2, "{failures:?}");
        assert!(failures.iter().any(|f| f.contains("speculation_substrate")));
        assert!(failures.iter().any(|f| f.contains("matmul_substrate")));
    }

    #[test]
    fn the_foundation_pass_substrate_home_is_accepted() {
        // ARCHITECTURE.md: foundation owns the CPU pass math and the pass engine
        // imports it, so this is the one home the name has. Every other home
        // stays a failure.
        let failures = substrate_home_failures(&[
            "vyre-foundation/src/pass_substrate/semiring_closure.rs".to_string(),
            "vyre-foundation/src/pass_substrate/mod.rs".to_string(),
        ]);

        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn a_second_c_frontend_crate_is_rejected_by_its_name() {
        // vyre-libs owns the C frontend because it is built from Category A
        // compositions. The tree-sitter shell crate that was the second one
        // kept a flat layout with no lex/ directory, so only its name gave it
        // away. It has left the workspace; the rule keeps a replacement out.
        let failures = frontend_owner_failures(&[
            (
                "vyre-libs".to_string(),
                "vyre-libs/src/parsing/c/lex/keyword.rs".to_string(),
            ),
            ("vyre-frontend-c".to_string(), "vyre-frontend-c".to_string()),
        ]);

        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(
            failures[0].contains("`vyre-frontend-c` is a second c frontend crate"),
            "{failures:?}"
        );
    }

    #[test]
    fn a_language_stage_directory_outside_the_owner_is_rejected() {
        // The other signal: the crate name says nothing, but it holds a
        // language-named lexer stage that belongs to the owning crate.
        let failures = frontend_owner_failures(&[(
            "vyre-driver-wgpu".to_string(),
            "vyre-driver-wgpu/src/parsing/c/lex/keyword.rs".to_string(),
        )]);

        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(
            failures[0].contains("puts a c frontend in vyre-driver-wgpu"),
            "{failures:?}"
        );
    }

    #[test]
    fn the_declared_owner_of_a_language_is_accepted() {
        // Control: naming a frontend is only a failure for a non-owner, and
        // the rust owner is a crate whose own name declares the language.
        let failures = frontend_owner_failures(&[
            (
                "vyre-frontend-rust".to_string(),
                "vyre-frontend-rust".to_string(),
            ),
            (
                "vyre-libs".to_string(),
                "vyre-libs/src/parsing/c/lex/keyword.rs".to_string(),
            ),
        ]);

        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn a_product_crate_on_the_roster_is_rejected() {
        let failures = roster_failures(&["vyre-foundation".to_string(), "vyre-scan".to_string()]);

        assert!(
            failures.iter().any(|f| f.contains("vyre-scan")),
            "{failures:?}"
        );
    }

    fn discarding_import(file: &str, named: &str) -> DiscardingImport {
        DiscardingImport {
            file: file.to_string(),
            named: named.to_string(),
        }
    }

    #[test]
    fn a_discarding_import_of_a_submitting_crate_is_rejected() {
        let failures = registry_link_failures(
            &["vyre-libs".to_string(), "vyre-driver-cuda".to_string()],
            &[discarding_import(
                "conform/vyre-conform/tests/ulp_audit.rs",
                "vyre_libs",
            )],
        );

        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("vyre-libs"));
        assert!(failures[0].contains("ulp_audit.rs"));
        assert!(failures[0].contains(REGISTRY_LINK_OWNER));
    }

    #[test]
    fn a_discarding_import_of_a_crate_that_registers_nothing_is_accepted() {
        let failures = registry_link_failures(
            &["vyre-libs".to_string()],
            &[discarding_import("vyre/src/lib.rs", "vyre_spec")],
        );

        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn a_crate_identifier_matches_its_hyphenated_crate_name() {
        let failures = registry_link_failures(
            &["vyre-driver-reference".to_string()],
            &[discarding_import(
                "conform/vyre-conform/src/main.rs",
                "vyre_driver_reference",
            )],
        );

        assert_eq!(failures.len(), 1);
    }
}
