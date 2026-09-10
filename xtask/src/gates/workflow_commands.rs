//! The cargo invocations a CI workflow runs.
//!
//! Two gates read the same thing out of `.github/workflows`: which packages a
//! lane builds, with which features, and which test targets it names. They
//! parsed it separately, and a parser copied is a parser that diverges on the
//! one question both are asking.
//!
//! Every invocation in these workflows begins `./cargo_full`, so splitting on
//! it yields one command per segment. A segment is cut at the next step header
//! so a `-p` in one step cannot borrow a `--features` from the next. `--test`
//! takes the next token as a target name; `--tests` is a different token and
//! selects every test target, which is the unfiltered case.
//!
//! Comment lines are dropped before the split. These workflows explain their
//! steps in prose, and prose quotes commands: reading a commented command as
//! coverage lets a sentence satisfy a rule while no lane runs anything, which
//! is the one way a gate can certify what it never checked. Dropping whole
//! comment lines cannot break a folded scalar, whose continuation lines are
//! argv and never start with `#`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::gate::GateError;
use crate::gates::scan::Tree;

/// Where the live workflows are.
pub const WORKFLOWS: &str = ".github/workflows";

/// One `./cargo_full` invocation read out of a workflow step.
pub struct WorkflowCommand {
    /// Packages the invocation names with `-p` or `--package`.
    pub packages: BTreeSet<String>,
    /// Features the invocation enables with `--features`.
    pub features: BTreeSet<String>,
    /// Test targets the invocation names with `--test`.
    pub targets: BTreeSet<String>,
}

/// Every workflow file in the tree, in a stable order.
pub fn workflow_files(tree: &Tree) -> Vec<&Path> {
    let mut files: Vec<&Path> = tree
        .paths()
        .iter()
        .filter(|path| {
            path.starts_with(WORKFLOWS) && path.extension().is_some_and(|kind| kind == "yml")
        })
        .map(PathBuf::as_path)
        .collect();
    files.sort_unstable();
    files
}

/// Every invocation in one workflow that enables `feature`.
#[must_use]
pub fn enabling(text: &str, feature: &str) -> Vec<WorkflowCommand> {
    let collapsed = text
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .flat_map(str::split_whitespace)
        .collect::<Vec<_>>()
        .join(" ");
    let mut commands = Vec::new();
    for segment in collapsed.split("./cargo_full").skip(1) {
        let command = segment.split("- name:").next().unwrap_or(segment);
        if !command.contains(feature) {
            continue;
        }
        let mut parsed = WorkflowCommand {
            packages: BTreeSet::new(),
            features: BTreeSet::new(),
            targets: BTreeSet::new(),
        };
        let mut tokens = command.split(' ');
        while let Some(token) = tokens.next() {
            let Some(value) = (match token {
                "-p" | "--package" | "--features" | "--test" => tokens.next(),
                _ => None,
            }) else {
                continue;
            };
            let value = value.trim_matches(['\'', '"']);
            match token {
                "-p" | "--package" => {
                    parsed.packages.insert(value.to_string());
                }
                "--test" => {
                    parsed.targets.insert(value.to_string());
                }
                _ => parsed.features.extend(
                    value
                        .split(',')
                        .filter(|feature| !feature.is_empty())
                        .map(str::to_string),
                ),
            }
        }
        commands.push(parsed);
    }
    commands
}

/// Every invocation across every workflow that enables `feature`.
pub fn enabling_in_tree(tree: &Tree, feature: &str) -> Result<Vec<WorkflowCommand>, GateError> {
    let mut commands = Vec::new();
    for path in workflow_files(tree) {
        commands.extend(enabling(&tree.read(path)?, feature));
    }
    Ok(commands)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    /// WHY: this grammar decides which packages two gates believe a lane builds
    /// and with what. Borrowing a `--features` across a step header, reading a
    /// commented example as a live command, or keeping the shell quotes on a
    /// value each make a gate certify a lane that does not exist.
    #[test]
    fn one_step_never_borrows_an_argument_from_the_next() {
        let workflow = concat!(
            "jobs:\n",
            "  gpu:\n",
            "    steps:\n",
            "      # run: ./cargo_full test -p vyre-ghost --features device-tests\n",
            "      - name: CUDA contracts\n",
            "        run: ./cargo_full test -p 'vyre-driver-cuda' --features \"cuda,device-tests\" --test all_tests\n",
            "      - name: WGPU contracts\n",
            "        run: ./cargo_full test -p vyre-driver-wgpu --features device-tests --tests\n",
            "      - name: unrelated\n",
            "        run: ./cargo_full check -p vyre-foundation\n",
        );

        let commands = enabling(workflow, "device-tests");
        assert_eq!(
            commands.len(),
            2,
            "a commented command and a command without the feature are not lanes"
        );
        assert_eq!(commands[0].packages, named(&["vyre-driver-cuda"]));
        assert_eq!(commands[0].features, named(&["cuda", "device-tests"]));
        assert_eq!(
            commands[0].targets,
            named(&["all_tests"]),
            "`--test` names one target"
        );
        assert_eq!(commands[1].packages, named(&["vyre-driver-wgpu"]));
        assert_eq!(commands[1].features, named(&["device-tests"]));
        assert!(
            commands[1].targets.is_empty(),
            "`--tests` selects every target and names none"
        );
    }

    /// WHY: a trailing flag with no value must not consume the next command's
    /// first token, and a workflow that runs nothing with the feature yields no
    /// lane rather than an empty one.
    #[test]
    fn a_flag_with_no_value_yields_no_lane_argument() {
        let commands = enabling(
            "run: ./cargo_full test --features device-tests -p\n",
            "device-tests",
        );
        assert_eq!(commands.len(), 1);
        assert!(commands[0].packages.is_empty());
        assert_eq!(commands[0].features, named(&["device-tests"]));

        assert!(enabling("run: ./cargo_full test -p vyre-foundation\n", "device-tests").is_empty());
    }
}
