//! The workspace lint floor rejects each defect shape it claims to reject.
//!
//! WHY: `[workspace.lints.rust]` denies `dead_code`, `unused_variables`,
//! `unreachable_pub` and `unfulfilled_lint_expectations`, and every member
//! inherits it. Nothing proved that a member inheriting the table actually
//! fails on those shapes: the table could be edited to `warn`, a level could be
//! dropped, or a future edition could change a default, and the tree would keep
//! compiling with the signal gone. A gate that certifies a policy it never
//! exercises is worse than no gate.
//!
//! Each case below is a whole crate that inherits the levels read from the root
//! manifest at run time, so a level lowered in the table turns the matching case
//! RED rather than quietly widening what ships. The clean case fails if the
//! floor rejects code it should accept.
//!
//! Not covered here: a lint the table denies and no case names. That direction
//! is closed by `every_denied_lint_has_a_mutation_case`, which reads the table
//! and requires a case per denied level. A member that never inherits the table
//! is closed by `every_workspace_member_inherits_the_lint_table`: the levels are
//! declared once at the root and apply to a crate only through its own `[lints]
//! workspace = true`, so a new member added without that line escapes every deny
//! silently.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// One defect shape, the lint that must reject it, and the crate that carries it.
struct Mutation {
    /// Case name, used as the fixture crate name and in failure output.
    name: &'static str,
    /// The denied lint whose diagnostic must name this shape.
    lint: &'static str,
    /// `src/lib.rs` for the fixture crate.
    source: &'static str,
}

const MUTATIONS: &[Mutation] = &[
    Mutation {
        name: "unused_private_item",
        lint: "dead_code",
        source: "//! Fixture.\nfn never_called() {}\n",
    },
    Mutation {
        name: "unreachable_public_item",
        lint: "unreachable_pub",
        source: "//! Fixture.\nmod inner {\n    /// Doc.\n    pub fn reachable_from_nowhere() {}\n}\n\n/// Doc.\npub fn entry() {\n    inner::reachable_from_nowhere();\n}\n",
    },
    Mutation {
        name: "stale_expectation",
        lint: "unfulfilled_lint_expectations",
        source: "//! Fixture.\n#[expect(dead_code, reason = \"the expectation is stale\")]\nfn called_after_all() {}\n\n/// Doc.\npub fn entry() {\n    called_after_all();\n}\n",
    },
    Mutation {
        name: "feature_only_orphan",
        lint: "dead_code",
        source: "//! Fixture.\nfn helper() {}\n\n/// Doc.\n#[cfg(feature = \"extra\")]\npub fn entry() {\n    helper();\n}\n",
    },
    Mutation {
        name: "test_only_production_caller",
        lint: "dead_code",
        source: "//! Fixture.\nfn helper() {}\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn exercises_the_helper() {\n        super::helper();\n    }\n}\n",
    },
    Mutation {
        name: "undeclared_feature_cfg",
        lint: "unexpected_cfgs",
        source: "//! Fixture.\n/// Doc.\n#[cfg(feature = \"undeclared\")]\npub fn entry() {}\n",
    },
    Mutation {
        name: "unused_binding",
        lint: "unused_variables",
        source: "//! Fixture.\n/// Doc.\npub fn entry(count: u32) {\n    let _ = 1u32;\n}\n",
    },
];

/// The clean crate the floor must accept, so a case failing proves the shape
/// rather than a broken fixture.
const CLEAN: &str = "//! Fixture.\nfn helper() -> u32 {\n    1\n}\n\n/// Doc.\npub fn entry(count: u32) -> u32 {\n    count + helper()\n}\n";

/// The edition the root manifest declares for every member.
///
/// A fixture on a different edition would prove the policy on a language the
/// tree does not compile with.
fn declared_edition() -> String {
    let manifest = structure_gate::workspace_root().join("Cargo.toml");
    let text = fs::read_to_string(&manifest)
        .unwrap_or_else(|err| panic!("cannot read {}: {err}", manifest.display()));
    let table: toml::Table = toml::from_str(&text).expect("the root manifest must be valid TOML");
    table
        .get("workspace")
        .and_then(|workspace| workspace.get("package"))
        .and_then(|package| package.get("edition"))
        .and_then(toml::Value::as_str)
        .expect("the root manifest must declare [workspace.package] edition")
        .to_string()
}

/// The `[workspace.lints.rust]` levels the root manifest declares.
fn declared_lint_levels() -> BTreeMap<String, String> {
    let manifest = structure_gate::workspace_root().join("Cargo.toml");
    let text = fs::read_to_string(&manifest)
        .unwrap_or_else(|err| panic!("cannot read {}: {err}", manifest.display()));
    let table: toml::Table = toml::from_str(&text).expect("the root manifest must be valid TOML");
    let rust = table
        .get("workspace")
        .and_then(|workspace| workspace.get("lints"))
        .and_then(|lints| lints.get("rust"))
        .and_then(toml::Value::as_table)
        .expect("the root manifest must declare [workspace.lints.rust]");
    rust.iter()
        .map(|(lint, value)| {
            let level = match value {
                toml::Value::String(level) => level.clone(),
                toml::Value::Table(entry) => entry
                    .get("level")
                    .and_then(toml::Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                _ => String::new(),
            };
            (lint.clone(), level)
        })
        .collect()
}

/// Render the fixture crate and return its directory.
fn write_fixture(root: &Path, name: &str, source: &str) -> PathBuf {
    let levels = declared_lint_levels();
    let mut lints = String::new();
    for (lint, level) in &levels {
        lints.push_str(&format!("{lint} = \"{level}\"\n"));
    }

    let edition = declared_edition();
    let dir = root.join(name);
    fs::create_dir_all(dir.join("src")).expect("fixture crate directories must be creatable");
    fs::write(
        dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"{name}\"\nversion = \"0.0.0\"\nedition = \"{edition}\"\n\n\
             [features]\nextra = []\n\n\
             [lib]\npath = \"src/lib.rs\"\n\n\
             [lints.rust]\n{lints}\n\
             [workspace]\n"
        ),
    )
    .expect("fixture manifest must be writable");
    fs::write(dir.join("src/lib.rs"), source).expect("fixture source must be writable");
    dir
}

/// Run `cargo check` on a fixture crate and return its combined diagnostics.
///
/// The `extra` feature stays off, which is what makes the feature-only orphan
/// case an orphan.
fn check_fixture(dir: &Path) -> (bool, String) {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let mut command = Command::new(cargo);
    command.arg("check").arg("--quiet").arg("--offline");
    command.current_dir(dir);
    // The fixture is outside the workspace, so it resolves its own target
    // directory under its own root. Nothing here overrides a build setting.
    command.env_remove("CARGO_TARGET_DIR");
    command.env_remove("RUSTFLAGS");
    let output = command
        .output()
        .unwrap_or_else(|err| panic!("cannot run cargo check in {}: {err}", dir.display()));
    let mut diagnostics = String::from_utf8_lossy(&output.stderr).to_string();
    diagnostics.push_str(&String::from_utf8_lossy(&output.stdout));
    (output.status.success(), diagnostics)
}

#[test]
fn the_workspace_lint_floor_rejects_every_mutation_and_accepts_clean_source() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("lint-policy-mutations");
    fs::create_dir_all(&root).expect("the fixture root must be creatable");

    let clean_dir = write_fixture(&root, "lint_floor_clean", CLEAN);
    let (clean_ok, clean_diagnostics) = check_fixture(&clean_dir);
    assert!(
        clean_ok,
        "the workspace lint floor rejects clean source, so every case below proves nothing:\n{clean_diagnostics}"
    );

    let mut survived = Vec::new();
    for mutation in MUTATIONS {
        let dir = write_fixture(&root, mutation.name, mutation.source);
        let (compiled, diagnostics) = check_fixture(&dir);
        // A lint reaches a fixture through the `[lints]` table, so cargo passes
        // it on the command line and rustc echoes the flag spelling:
        // `-D dead-code` for a table entry written `dead_code`. Reading the
        // diagnostics for the table spelling alone found neither, and every
        // case reported as unrejected while the floor was doing its job.
        let named = diagnostics.replace('-', "_");
        if compiled {
            survived.push(format!(
                "{}: compiled clean; `{}` did not reject it",
                mutation.name, mutation.lint
            ));
        } else if !named.contains(mutation.lint) {
            survived.push(format!(
                "{}: failed without naming `{}`:\n{diagnostics}",
                mutation.name, mutation.lint
            ));
        }
    }

    assert!(
        survived.is_empty(),
        "the workspace lint floor certifies shapes it does not reject:\n{}",
        survived.join("\n")
    );
}

#[test]
fn every_denied_lint_has_a_mutation_case() {
    let levels = declared_lint_levels();
    let denied: Vec<&String> = levels
        .iter()
        .filter(|(_, level)| level.as_str() == "deny" || level.as_str() == "forbid")
        .map(|(lint, _)| lint)
        .collect();
    assert!(
        denied.len() >= 4,
        "the level scan found {} denied lints, which is a broken scan rather than a small policy",
        denied.len()
    );

    // The split between what the compiler proves here and what a gate proves is
    // declared once, beside the gate that reads it, so a lint promoted to `deny`
    // is answered for in one place rather than two that can disagree.
    let covered: Vec<&str> = MUTATIONS.iter().map(|case| case.lint).collect();
    let missing: Vec<&str> = denied
        .iter()
        .map(|lint| lint.as_str())
        .filter(|lint| {
            !xtask::gates::lint_hygiene::LINTS_OWNED_ELSEWHERE.contains(lint)
                && !covered.contains(lint)
        })
        .collect();

    assert!(
        missing.is_empty(),
        "denied with no mutation case, so nothing proves the level does anything: {missing:?}"
    );
}

/// Every path `[workspace] members` names, from the root manifest.
fn declared_members() -> Vec<String> {
    let manifest = structure_gate::workspace_root().join("Cargo.toml");
    let text = fs::read_to_string(&manifest)
        .unwrap_or_else(|err| panic!("cannot read {}: {err}", manifest.display()));
    let table: toml::Table = toml::from_str(&text).expect("the root manifest must be valid TOML");
    let members = table
        .get("workspace")
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array)
        .expect("the root manifest must declare [workspace] members");
    members
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .expect("every members entry must be a path string")
                .to_string()
        })
        .collect()
}

/// Whether a member manifest opts into the workspace lint table.
fn inherits_workspace_lints(manifest: &Path) -> bool {
    let text = fs::read_to_string(manifest)
        .unwrap_or_else(|err| panic!("cannot read {}: {err}", manifest.display()));
    let table: toml::Table = toml::from_str(&text)
        .unwrap_or_else(|err| panic!("{} must be valid TOML: {err}", manifest.display()));
    table
        .get("lints")
        .and_then(toml::Value::as_table)
        .and_then(|lints| lints.get("workspace"))
        .and_then(toml::Value::as_bool)
        .unwrap_or(false)
}

#[test]
fn every_workspace_member_inherits_the_lint_table() {
    let root = structure_gate::workspace_root();
    let members = declared_members();
    assert!(
        members.len() >= 20,
        "the members scan found {} entries, which is a broken scan rather than a small workspace",
        members.len()
    );

    let mut escaping = Vec::new();
    for member in &members {
        let manifest = root.join(member).join("Cargo.toml");
        assert!(
            manifest.is_file(),
            "[workspace] members names {member}, which carries no manifest"
        );
        if !inherits_workspace_lints(&manifest) {
            escaping.push(member.clone());
        }
    }

    assert!(
        escaping.is_empty(),
        "these members declare no `[lints] workspace = true`, so every denied lint in \
         [workspace.lints.rust] is a warning there and the shapes this suite proves rejected are \
         accepted: {escaping:?}"
    );
}
