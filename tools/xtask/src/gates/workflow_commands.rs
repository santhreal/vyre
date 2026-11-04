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
//!
//! An xtask gate that runs cargo itself is read as the cargo invocation it runs.
//! `XTASK_CARGO_RUNS` lists them, each from the constants the gate builds its
//! command from, so a step running `xtask cuda-parity --device` covers what that
//! gate compiles and nothing else.

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

/// An xtask gate invocation that runs one cargo build of one package.
pub struct XtaskCargoRun {
    /// The gate name following `xtask --`.
    pub gate: &'static str,
    /// The flag that makes the gate run cargo.
    pub flag: &'static str,
    /// The package the gate builds.
    pub package: &'static str,
    /// The comma-separated features the build enables. The build names no
    /// `--test`, so it compiles every test target.
    pub features: &'static str,
}

/// Every xtask gate invocation that builds a package itself.
const XTASK_CARGO_RUNS: &[XtaskCargoRun] = &[crate::gates::backend_parity::CUDA_DEVICE_RUN];

/// The cargo invocation an xtask gate step runs, when `command` is one.
fn xtask_cargo_run(command: &str) -> Option<WorkflowCommand> {
    let (runner, arguments) = command.split_once(" -- ")?;
    if !runner.split(' ').any(|token| token == "xtask") {
        return None;
    }
    let mut arguments = arguments.split(' ').filter(|token| !token.is_empty());
    let gate = arguments.next()?;
    let flags: Vec<&str> = arguments.collect();
    let run = XTASK_CARGO_RUNS
        .iter()
        .find(|run| run.gate == gate && flags.contains(&run.flag))?;
    Some(WorkflowCommand {
        packages: BTreeSet::from([run.package.to_string()]),
        features: run.features.split(',').map(str::to_string).collect(),
        targets: BTreeSet::new(),
    })
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
        if let Some(run) = xtask_cargo_run(command) {
            if run.features.contains(feature) {
                commands.push(run);
            }
            continue;
        }
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

        assert!(enabling(
            "run: ./cargo_full test -p vyre-foundation\n",
            "device-tests"
        )
        .is_empty());
    }

    /// WHY: `xtask cuda-parity --device` compiles every `vyre-driver-cuda` test
    /// target with `device-tests`, and a reader that saw only `./cargo_full
    /// test` read the CUDA crate as built by nothing once the raw cargo step
    /// folded into the gate. The gate without `--device` builds nothing, and an
    /// unlisted gate is not a build.
    #[test]
    fn an_xtask_gate_that_runs_cargo_is_the_build_it_runs() {
        for run in XTASK_CARGO_RUNS {
            assert!(
                crate::gates::GATES
                    .iter()
                    .any(|(name, _)| *name == run.gate),
                "`{}` is not a registered gate, so no step can run it",
                run.gate
            );
        }
        let workflow = concat!(
            "      - name: CUDA parity\n",
            "        run: ./cargo_full run -q -p xtask --bin xtask -- cuda-parity --device\n",
            "      - name: CUDA static\n",
            "        run: ./cargo_full run -q -p xtask --bin xtask -- cuda-parity\n",
            "      - name: unrelated gate\n",
            "        run: ./cargo_full run -q -p xtask --bin xtask -- file-size --device\n",
        );

        let commands = enabling(workflow, "device-tests");
        assert_eq!(commands.len(), 1, "only the device run is a build");
        let run = &crate::gates::backend_parity::CUDA_DEVICE_RUN;
        assert_eq!(commands[0].packages, named(&[run.package]));
        assert_eq!(
            commands[0].features,
            run.features.split(',').map(str::to_string).collect()
        );
        assert!(commands[0].targets.is_empty(), "the run names no `--test`");
        assert!(enabling(workflow, "spirv-val").is_empty());
    }
}
