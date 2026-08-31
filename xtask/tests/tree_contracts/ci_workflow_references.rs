//! Every name a CI step passes to cargo or to a shell resolves in this tree.
//!
//! WHY: a workflow step is text until CI runs it, so a target that is renamed,
//! merged into another target, or deleted leaves the step naming something that
//! does not exist, and nothing local goes red. That shipped:
//! `architectural-invariants.yml` ran `--test architecture_docs --test
//! canonical_first_workgroup_guard` for months after both became modules of the
//! `tree_contracts` target, so the whole architecture gate would have failed
//! with `no test target named` on the first run that reached it.
//!
//! The existing CI inspector cannot see this. It asserts that a command STRING
//! appears in a workflow file, which is exactly as true for a command naming a
//! deleted target as for one naming a live target.
//!
//! Every name is extracted from the workflow files and resolved against the
//! tree at run time, so a step added tomorrow is judged tomorrow, and renaming
//! a target without updating its workflow is red here rather than in CI.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use super::workspace_sources::workspace_root;

/// One cargo or shell name a workflow step passes, and where it came from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Reference {
    workflow: String,
    kind: Kind,
    name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    Package,
    Test,
    Bin,
    Script,
    Subcommand,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::Package => "package (-p)",
            Kind::Test => "test target (--test)",
            Kind::Bin => "binary (--bin)",
            Kind::Script => "script path",
            Kind::Subcommand => "xtask subcommand",
        }
    }
}

/// Workflow files, read from the tree rather than listed here.
fn workflow_files(root: &Path) -> Vec<(String, Vec<String>)> {
    let directory = root.join(".github/workflows");
    let entries = fs::read_dir(&directory)
        .unwrap_or_else(|error| panic!("Fix: {} must be readable: {error}", directory.display()));
    let mut files = Vec::new();
    for entry in entries {
        let path = entry
            .expect("Fix: a workflow directory entry must be readable")
            .path();
        if !path
            .extension()
            .is_some_and(|ext| ext == "yml" || ext == "yaml")
        {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("Fix: a workflow file name must be UTF-8")
            .to_string();
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", path.display()));
        files.push((name, run_commands(&text)));
    }
    files.sort();
    assert!(
        !files.is_empty(),
        "Fix: no workflow files under {}, so this gate guards nothing",
        directory.display()
    );
    files
}

/// The shell command text of every `run:` step, one string per step.
///
/// A step is the unit of scope: `--test` without `-p` addresses whatever the
/// same command line names, and nothing a neighbouring step names. Reading the
/// whole file instead would resolve a target against another step's package,
/// and would also read prose out of `name:` lines and comments.
///
/// Both YAML block forms are handled by indentation: `run:` on one line takes
/// the following more-indented lines, and `run: <command>` is its own step.
fn run_commands(text: &str) -> Vec<String> {
    let mut commands = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let Some(column) = line.find("run:") else {
            continue;
        };
        let inline = line[column + "run:".len()..].trim();
        let mut command = String::new();
        if !inline.is_empty() && inline != "|" && inline != ">-" && inline != ">" && inline != "|-"
        {
            command.push_str(inline);
        }
        let indent = line.len() - line.trim_start().len();
        while let Some(next) = lines.peek() {
            if next.trim().is_empty() {
                lines.next();
                continue;
            }
            let next_indent = next.len() - next.trim_start().len();
            if next_indent <= indent {
                break;
            }
            command.push(' ');
            command.push_str(next.trim());
            lines.next();
        }
        if !command.trim().is_empty() {
            commands.push(command);
        }
    }
    commands
}

/// Packages whose binary takes a registered subcommand as its first argument.
///
/// `xtask` dispatches by name, and the other two implement rows of the same
/// table, so a step naming any of them passes a subcommand the same way.
const SUBCOMMAND_PACKAGES: [&str; 3] = ["xtask", "xtask-registry", "xtask-evidence"];

/// Every cargo and script name one step's command line passes.
///
/// Tokenizing on whitespace is enough because the input is a shell command
/// line: `--test foo` and `-p bar` are adjacent tokens, and a script is a token
/// that starts with `scripts/`. A `${{ }}` expression and a `$name` shell
/// variable are both skipped, since neither resolves to a name in this tree.
fn references(workflow: &str, command: &str) -> Vec<Reference> {
    let mut found = Vec::new();
    let mut push = |kind: Kind, name: &str| {
        let name = name.trim_matches(['"', '\'', '`']);
        if name.is_empty() || name.contains('$') {
            return;
        }
        found.push(Reference {
            workflow: workflow.to_string(),
            kind,
            name: name.to_string(),
        });
    };
    // Split compound scripts into individual command segments so that flags
    // and options like `--` or `-p` on one command are not attributed to
    // another command in the same multi-line or piped step.
    for segment in command.split(|c| {
        c == '\n' || c == ';' || c == '|' || c == '&' || c == '`' || c == '(' || c == ')'
    }) {
        let tokens: Vec<&str> = segment.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }

        let is_cargo = tokens
            .iter()
            .any(|t| matches!(*t, "cargo" | "cargo_full" | "./cargo_full"));

        let is_cargo_run = is_cargo && tokens.contains(&"run");

        let addresses_dispatcher = tokens.windows(2).any(|pair| {
            (matches!(pair[0], "-p" | "--package") && SUBCOMMAND_PACKAGES.contains(&pair[1]))
                || (pair[0] == "--bin" && SUBCOMMAND_PACKAGES.contains(&pair[1]))
        }) || tokens.iter().any(|token| {
            token
                .strip_prefix("--package=")
                .is_some_and(|name| SUBCOMMAND_PACKAGES.contains(&name))
                || token
                    .strip_prefix("--bin=")
                    .is_some_and(|name| SUBCOMMAND_PACKAGES.contains(&name))
        });

        let addresses_other_bin = tokens
            .windows(2)
            .any(|pair| pair[0] == "--bin" && !SUBCOMMAND_PACKAGES.contains(&pair[1]))
            || tokens.iter().any(|token| {
                token
                    .strip_prefix("--bin=")
                    .is_some_and(|name| !SUBCOMMAND_PACKAGES.contains(&name))
            });

        for (index, token) in tokens.iter().enumerate() {
            let next = tokens.get(index + 1).copied().unwrap_or_default();
            match *token {
                "-p" | "--package" if is_cargo => push(Kind::Package, next),
                "--test" if is_cargo => push(Kind::Test, next),
                "--bin" if is_cargo => push(Kind::Bin, next),
                _ => {
                    if let Some(name) = token.strip_prefix("--test=").filter(|_| is_cargo) {
                        push(Kind::Test, name);
                    } else if let Some(name) = token.strip_prefix("--bin=").filter(|_| is_cargo) {
                        push(Kind::Bin, name);
                    } else if let Some(name) = token.strip_prefix("--package=").filter(|_| is_cargo)
                    {
                        push(Kind::Package, name);
                    } else if token.starts_with("scripts/") {
                        push(Kind::Script, token);
                    }
                }
            }
        }

        if is_cargo_run && addresses_dispatcher && !addresses_other_bin {
            for (index, token) in tokens.iter().enumerate() {
                if *token != "--" {
                    continue;
                }
                let Some(name) = tokens.get(index + 1) else {
                    continue;
                };
                if !name.starts_with('-') {
                    push(Kind::Subcommand, name);
                }
            }
        }
    }
    found
}

/// Package names declared by the workspace members, read from their manifests.
fn declared_packages(root: &Path) -> BTreeSet<String> {
    let mut packages = BTreeSet::new();
    for member in structure_gate::workspace_members(root) {
        let manifest = root.join(&member).join("Cargo.toml");
        let Ok(text) = fs::read_to_string(&manifest) else {
            continue;
        };
        if let Some(name) = manifest_package_name(&text) {
            packages.insert(name);
        }
    }
    assert!(
        !packages.is_empty(),
        "Fix: the workspace roster resolved no packages, so this gate guards nothing"
    );
    packages
}

fn manifest_package_name(text: &str) -> Option<String> {
    toml::from_str::<toml::Value>(text)
        .ok()?
        .get("package")?
        .get("name")?
        .as_str()
        .map(str::to_string)
}

/// Whether `package` declares or auto-discovers a target of `kind` named `name`.
///
/// Cargo auto-discovers `tests/NAME.rs` and `tests/NAME/main.rs`, and the same
/// two shapes under `src/bin` for a binary, so both are accepted alongside an
/// explicit `[[test]]` or `[[bin]]` block.
fn target_exists(root: &Path, package: &str, kind: Kind, name: &str) -> bool {
    let directory = structure_gate::member_directory(root, package);
    let (auto_dir, table) = match kind {
        Kind::Test => ("tests", "test"),
        Kind::Bin => ("src/bin", "bin"),
        _ => return true,
    };
    let auto = directory.join(auto_dir);
    if auto.join(format!("{name}.rs")).is_file() || auto.join(name).join("main.rs").is_file() {
        return true;
    }
    if kind == Kind::Bin && package == name && directory.join("src/main.rs").is_file() {
        return true;
    }
    let Ok(text) = fs::read_to_string(directory.join("Cargo.toml")) else {
        return false;
    };
    let Ok(manifest) = toml::from_str::<toml::Value>(&text) else {
        return false;
    };
    manifest
        .get(table)
        .and_then(toml::Value::as_array)
        .is_some_and(|blocks| {
            blocks.iter().any(|block| {
                block
                    .get("name")
                    .and_then(toml::Value::as_str)
                    .is_some_and(|declared| declared == name)
            })
        })
}

/// Packages this one command names, empty when it addresses the workspace.
fn packages_named(references: &[Reference]) -> Vec<String> {
    references
        .iter()
        .filter(|reference| reference.kind == Kind::Package)
        .map(|reference| reference.name.clone())
        .collect()
}

/// WHY: `-p` is a cargo package selector and also `mkdir`'s parents flag, and
/// `cp -p` preserves mode. Reading every `-p` as a package reported
/// `target/criterion` as a package no member declares, which is a finding about
/// a shell command the gate does not describe. The class is every selector
/// flag: a `find -name x` would have failed the same way as `--test x`.
#[test]
fn a_selector_flag_outside_cargo_names_nothing() {
    assert_eq!(
        references("bench.yml", "mkdir -p target/criterion"),
        Vec::new()
    );
    assert_eq!(
        references(
            "bench.yml",
            "cp -a baseline/target/criterion/. target/criterion/"
        ),
        Vec::new()
    );
    assert_eq!(references("ci.yml", "find . -name vyre-libs"), Vec::new());

    let cargo = references("ci.yml", "cargo test -p vyre-libs");
    assert_eq!(cargo.len(), 1, "a cargo package selector is still read");
    assert_eq!(cargo[0].name, "vyre-libs");
}

/// WHY: a step that loops over packages spells the selector `-p "$package"`,
/// and reading that literally reports `$package` as a package no member
/// declares. The value is produced by the shell at run time, so it is not a
/// name this tree can resolve, exactly like a `${{ }}` expression. The class is
/// every selector kind, not just `-p`: a `--test "$name"` would have failed the
/// same way.
#[test]
fn a_shell_variable_is_not_read_as_a_name() {
    let command = "cargo test -p \"$package\" --test \"$suite\" --bin ${{ matrix.bin }}";
    assert_eq!(references("ci.yml", command), Vec::new());

    let literal = references("ci.yml", "cargo test -p xtask");
    assert_eq!(literal.len(), 1, "a literal package is still read");
    assert_eq!(literal[0].name, "xtask");
}

#[test]
fn every_package_a_workflow_names_is_a_workspace_member() {
    let root = workspace_root();
    let packages = declared_packages(&root);
    let mut offenders = Vec::new();

    for (workflow, commands) in workflow_files(&root) {
        for command in &commands {
            for reference in references(&workflow, command) {
                if reference.kind == Kind::Package && !packages.contains(&reference.name) {
                    offenders.push(format!(
                        "{}: {} `{}`",
                        reference.workflow,
                        reference.kind.label(),
                        reference.name
                    ));
                }
            }
        }
    }
    offenders.sort();
    offenders.dedup();

    assert!(
        offenders.is_empty(),
        "{offenders:#?} name a package no workspace member declares, so the step fails only \
         once CI reaches it. Fix: name the package the roster in the root Cargo.toml declares, \
         or drop the step."
    );
}

#[test]
fn every_target_a_workflow_names_exists_in_the_package_it_addresses() {
    let root = workspace_root();
    let packages = declared_packages(&root);
    let mut offenders = Vec::new();
    let mut checked = 0_usize;

    for (workflow, commands) in workflow_files(&root) {
        for command in &commands {
            let found = references(&workflow, command);
            let scoped = packages_named(&found);
            for reference in &found {
                if !matches!(reference.kind, Kind::Test | Kind::Bin) {
                    continue;
                }
                checked += 1;
                let candidates: Vec<String> = if scoped.is_empty() {
                    packages.iter().cloned().collect()
                } else {
                    scoped.clone()
                };
                let resolved = candidates
                    .iter()
                    .any(|package| target_exists(&root, package, reference.kind, &reference.name));
                if !resolved {
                    offenders.push(format!(
                        "{}: {} `{}` in {:?}",
                        reference.workflow,
                        reference.kind.label(),
                        reference.name,
                        if scoped.is_empty() {
                            vec!["(whole workspace)".to_string()]
                        } else {
                            scoped.clone()
                        }
                    ));
                }
            }
        }
    }
    offenders.sort();
    offenders.dedup();

    assert!(
        checked > 0,
        "Fix: no workflow step names a cargo target, so this gate guards nothing"
    );
    assert!(
        offenders.is_empty(),
        "{offenders:#?} name a cargo target that does not exist, so the step fails with `no \
         test target named` or `no bin target named` on the first CI run that reaches it. Fix: \
         name the target the package declares or auto-discovers. A target merged into another \
         one takes its workflow step with it."
    );
}

/// A workflow step runs no repository script.
///
/// WHY: every CI assertion is a registered `xtask` gate, so a step that shells
/// into a repository script carries an assertion the registry, the baseline and
/// the subset roster cannot see. The scripts still tracked are operator actions
/// with no assertion, recorded in `xtask/script-assertion-ledger.md`, and no
/// workflow invokes one. This goes red on the first step that does, whether the
/// script is published or not.
#[test]
fn no_workflow_step_runs_a_repository_script() {
    let root = workspace_root();
    let mut offenders = Vec::new();
    let mut commands_read = 0_usize;

    for (workflow, commands) in workflow_files(&root) {
        for command in &commands {
            commands_read += 1;
            for reference in references(&workflow, command) {
                if reference.kind != Kind::Script {
                    continue;
                }
                offenders.push(format!("{}: {}", reference.workflow, reference.name));
            }
        }
    }
    offenders.sort();
    offenders.dedup();

    assert!(
        commands_read > 0,
        "Fix: no workflow step reached this gate, so it guards nothing"
    );
    assert!(
        offenders.is_empty(),
        "{offenders:#?} run a repository script from a workflow step. Fix: register the \
         assertion as an `xtask` gate, name that gate in the step, and record the departure in \
         `xtask/script-assertion-ledger.md`."
    );
}

/// Every subcommand a workflow passes to the runner is dispatchable.
///
/// WHY: `xtask` resolves a subcommand by name and exits with an error when the
/// name is not in the table. A row renamed or removed leaves the step naming a
/// command that no longer exists, and nothing local goes red: the package
/// exists, the binary exists, and the name is one shell token in a `run:` line.
/// `xtask gates` judges the other direction, that a registered row is wired
/// into CI, so this closes the pair.
///
/// The dispatchable set is every gate plus the sweep runner, which is the one
/// accepted name that is not a gate. Comparing against the gate registry alone
/// reported every workflow step that runs the sweep as unregistered, which is
/// the whole tree-rules job.
#[test]
fn every_xtask_subcommand_a_workflow_names_is_dispatchable() {
    let root = workspace_root();
    let registered: BTreeSet<&str> = xtask::subcommands::registry()
        .iter()
        .map(|gate| gate.name())
        .chain([xtask::gates::sweep::RUNNER])
        .collect();
    let mut offenders = Vec::new();
    let mut checked = 0_usize;

    for (workflow, commands) in workflow_files(&root) {
        for command in &commands {
            for reference in references(&workflow, command) {
                if reference.kind != Kind::Subcommand {
                    continue;
                }
                checked += 1;
                if !registered.contains(reference.name.as_str()) {
                    offenders.push(format!("{}: `{}`", reference.workflow, reference.name));
                }
            }
        }
    }
    offenders.sort();
    offenders.dedup();

    assert!(
        checked > 0,
        "Fix: no workflow step passes a subcommand to the build-task dispatcher, so this gate guards nothing"
    );
    assert!(
        offenders.is_empty(),
        "{offenders:#?} pass a subcommand the dispatcher does not accept, so the step fails with \
         `unknown subcommand` on the first CI run that reaches it. Fix: name a registered gate or \
         the sweep runner, or drop the step with the row it followed."
    );
}
