//! What continuous integration must declare for a merge to mean anything.
//!
//! Two rules live here because they answer one question from two directions: the
//! matrix rule holds the hosted lane to the platforms and toolchains it is the
//! only cover for, and the required-context rule holds the branch protection
//! contract to workflows that still define every context it names and still fail
//! closed when a dependency does not run.
//!
//! Both read what a workflow declares rather than whether a word occurs in it.
//! The version of the matrix rule before this one asked grep whether the file
//! contained `stable` anywhere, which the word `stable` in a step name satisfies,
//! and whether it contained `macos-latest`, which a commented-out axis satisfies
//! too. The required-context rule ran only when an operator applied branch
//! protection by hand, so six assertions about the workflow set ran on no
//! schedule at all.

use std::collections::BTreeSet;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// The hosted matrix workflow.
const CI: &str = ".github/workflows/ci.yml";

/// The workflow that owns device parity, which the hosted matrix defers to.
const GPU_PARITY: &str = ".github/workflows/gpu-parity.yml";

/// The document branch protection is applied from.
const REQUIRED: &str = ".github/CI_REQUIRED.md";

/// The heading after which a listed workflow is informational, not required.
const DEEP_GATES: &str = "## Scheduled or Manual Deep Gates";

/// Operating systems the workspace is compiled on.
const REQUIRED_OS: &[&str] = &["ubuntu-latest", "macos-latest", "windows-latest"];

/// Toolchains the workspace is compiled with.
const REQUIRED_TOOLCHAINS: &[&str] = &["stable", "nightly"];

/// Tokens that would turn the device requirement into an option.
const GPU_ESCAPES: &[&str] = &["no-gpu", "gpu-feature"];

/// The hosted matrix covers every platform and toolchain it is the only cover for.
pub struct CiMatrix;

impl Gate for CiMatrix {
    fn name(&self) -> &'static str {
        "ci-matrix"
    }

    fn help(&self) -> &'static str {
        "hosted CI matrix axes, and device escape hatches inside them"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();

        for required in [CI, GPU_PARITY] {
            if !tree.exists(required) {
                report.find(Finding::in_file(
                    required,
                    "the workflow is missing",
                    "restore the workflow; the hosted matrix and the device parity workflow \
                     cover different platforms and neither substitutes for the other",
                ));
            }
        }
        if !tree.exists(CI) {
            return Ok(report);
        }

        let text = tree.read(CI)?;
        let axes = matrix_axes(&text);
        if axes.is_empty() {
            report.find(Finding::in_file(
                CI,
                "the workflow declares no matrix axis",
                "declare a strategy matrix; one job on one platform is not a matrix, and the \
                 axes are what this gate can hold to a set",
            ));
            return Ok(report);
        }
        report.note(format!("{} matrix axis value(s) declared", axes.len()));

        for missing in REQUIRED_OS.iter().filter(|value| !axes.contains(**value)) {
            report.find(Finding::in_file(
                CI,
                format!("no matrix axis declares `{missing}`"),
                "restore the platform to the matrix; it is the only place the workspace is \
                 compiled for that operating system",
            ));
        }
        for missing in REQUIRED_TOOLCHAINS
            .iter()
            .filter(|value| !axes.contains(**value))
        {
            report.find(Finding::in_file(
                CI,
                format!("no matrix axis declares the `{missing}` toolchain"),
                "restore the toolchain to the matrix; a lint or a soundness change lands on \
                 one of the two before it lands on both",
            ));
        }

        for (number, line) in crate::gates::scan::numbered(&text) {
            if let Some(escape) = crate::gates::scan::first_of(line, GPU_ESCAPES) {
                report.find(Finding::at(
                    CI,
                    number,
                    format!("device escape hatch `{escape}` in the hosted matrix"),
                    "assume a device exists and let the device parity workflow own the \
                     hardware; a probe failure is a configuration failure and is reported \
                     loudly rather than compiled away",
                ));
            }
        }

        Ok(report)
    }
}

/// Every required status context resolves to a job that still fails closed.
pub struct CiRequired;

impl Gate for CiRequired {
    fn name(&self) -> &'static str {
        "ci-required"
    }

    fn help(&self) -> &'static str {
        "required status contexts, the workflows that define them, and their fan-in jobs"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();

        if !tree.exists(REQUIRED) {
            report.find(Finding::in_file(
                REQUIRED,
                "the required status context document is missing",
                "restore the document; branch protection is applied from it, so without it \
                 the required set is whatever the remote happens to hold",
            ));
            return Ok(report);
        }

        let document = tree.read(REQUIRED)?;
        let sections = required_sections(&document);
        if sections.is_empty() {
            report.find(Finding::in_file(
                REQUIRED,
                "no required status context is declared",
                "list each blocking context under a `## From `<workflow>`` heading; an empty \
                 list applies branch protection that requires nothing",
            ));
            return Ok(report);
        }

        let contexts: usize = sections.iter().map(|section| section.contexts.len()).sum();
        report.note(format!(
            "{contexts} required status context(s) across {} workflow(s)",
            sections.len()
        ));

        let blocking: BTreeSet<&str> = sections
            .iter()
            .map(|section| section.workflow.as_str())
            .collect();
        for (line, workflow) in named_workflows(&document) {
            let live = tree.exists(&format!(".github/workflows/{workflow}"));
            let paused = tree.exists(&format!(".github/workflows-paused/{workflow}"));
            if !live && !paused {
                report.find(Finding::at(
                    REQUIRED,
                    line,
                    format!("`{workflow}` names a workflow file the checkout does not carry"),
                    "name a workflow under .github/workflows or .github/workflows-paused, or \
                     delete the row; a filename in prose was checked by nothing, so this \
                     document promised two lanes that had been paused for months",
                ));
                continue;
            }
            if paused && blocking.contains(workflow.as_str()) {
                report.find(Finding::at(
                    REQUIRED,
                    line,
                    format!("`{workflow}` is paused and is also named as a blocking section"),
                    "restore the workflow, or move its contexts under the deep-gate heading; a \
                     paused workflow cannot report a context branch protection waits for",
                ));
            }
        }

        for (workflow, line) in sweep_workflows(&tree)? {
            if blocking.contains(workflow.as_str()) {
                continue;
            }
            report.find(Finding::at(
                &format!(".github/workflows/{workflow}"),
                line,
                format!(
                    "`{workflow}` runs the gate sweep on a pull request and no section of \
                     {REQUIRED} makes it blocking"
                ),
                "list the workflow's sweep jobs under a `## From `<workflow>`` heading in \
                 .github/CI_REQUIRED.md; every registered gate is judged by this workflow, so \
                 leaving it advisory lets a pull request merge with the whole registry red",
            ));
        }

        for section in &sections {
            let path = format!(".github/workflows/{}", section.workflow);
            if !tree.exists(&path) {
                report.find(Finding::at(
                    REQUIRED,
                    section.line,
                    format!(
                        "`{}` names a workflow the checkout does not carry",
                        section.workflow
                    ),
                    "point the section at a workflow that exists, or drop the section; a \
                     required context no workflow can report keeps every pull request pending",
                ));
                continue;
            }

            let text = tree.read(&path)?;
            report.findings.extend(trigger_findings(&path, &text));

            let jobs = jobs(&text);
            for (line, context) in &section.contexts {
                let Some(job) = jobs.iter().find(|job| {
                    job.id == *context || job.name.as_deref() == Some(context.as_str())
                }) else {
                    report.find(Finding::at(
                        REQUIRED,
                        *line,
                        format!("no job in `{}` is named `{context}`", section.workflow),
                        "name the job exactly as the required context, or correct the context; \
                         branch protection matches on this string and a mismatch blocks every \
                         merge on a check that never arrives",
                    ));
                    continue;
                };
                report.findings.extend(fail_closed_findings(&path, job));
            }
        }

        Ok(report)
    }
}

/// Every workflow that runs the gate sweep on a pull request, with the line it
/// runs it on.
///
/// Read from the workflow set rather than from a name, because the rule is that
/// whichever workflow judges the registry on a change is the one branch
/// protection has to wait for. A gate registered in a subset a workflow runs is
/// exactly as binding as that workflow's status context, and a sweep nothing
/// blocks on lets a pull request merge with every gate red.
///
/// A workflow that runs the sweep on a tag or on demand is not held to this: it
/// judges a release rather than a change, and it reports no context a pull
/// request could wait for.
fn sweep_workflows(tree: &Tree) -> Result<Vec<(String, u32)>, GateError> {
    let mut found = Vec::new();
    for path in tree.paths() {
        let relative = path.to_string_lossy();
        let Some(name) = workflow_file(&relative) else {
            continue;
        };
        let text = tree.read(path)?;
        let triggers = text.split("\njobs:").next().unwrap_or(&text);
        if !triggers.lines().any(|line| line.trim() == "pull_request:") {
            continue;
        }
        if let Some((number, _)) = crate::gates::scan::numbered(&text)
            .into_iter()
            .find(|(_, line)| line.contains("xtask") && invokes_sweep(line))
        {
            found.push((name, number));
        }
    }
    Ok(found)
}

/// The file name of a workflow, for a path that is one.
fn workflow_file(relative: &str) -> Option<String> {
    let name = relative.strip_prefix(".github/workflows/")?;
    if name.contains('/') || !(name.ends_with(".yml") || name.ends_with(".yaml")) {
        return None;
    }
    Some(name.to_string())
}

/// Whether one command line invokes the `gates` subcommand.
fn invokes_sweep(line: &str) -> bool {
    let mut rest = line;
    while let Some(at) = rest.find("-- ") {
        rest = &rest[at + 3..];
        let token: String = rest
            .chars()
            .take_while(|character| character.is_ascii_alphanumeric() || *character == '-')
            .collect();
        if token == "gates" {
            return true;
        }
    }
    false
}

/// One `## From `<workflow>`` section of the required-context document.
struct RequiredSection {
    /// The workflow file name the heading names.
    workflow: String,
    /// The line the heading is on.
    line: u32,
    /// Each context the section lists, with the line it is listed on.
    contexts: Vec<(u32, String)>,
}

/// Every blocking section of the required-context document.
///
/// Reading the sections rather than the bullet list is what couples a context to
/// the workflow that has to define it, and the deep-gate heading ends the
/// blocking set: the rows under it name lanes that report on a schedule, so
/// holding them to a job name would report a workflow that is deliberately paused.
fn required_sections(document: &str) -> Vec<RequiredSection> {
    let mut sections: Vec<RequiredSection> = Vec::new();
    for (number, line) in crate::gates::scan::numbered(document) {
        let trimmed = line.trim();
        if trimmed.starts_with(DEEP_GATES) {
            break;
        }
        if let Some(rest) = trimmed.strip_prefix("## From ") {
            if let Some(workflow) = quoted(rest) {
                sections.push(RequiredSection {
                    workflow,
                    line: number,
                    contexts: Vec::new(),
                });
            }
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("- ") {
            if let (Some(context), Some(section)) = (quoted(rest), sections.last_mut()) {
                section.contexts.push((number, context));
            }
        }
    }
    sections
}

/// Every workflow file name the document quotes, with the line it is on.
///
/// The section headings and the deep-gate rows both name files, and the rows are
/// where the rot lived: `parse_required_ci_statuses` resolves the contexts under
/// a heading against real workflows, so a filename written as prose under the
/// deep-gate heading was read by nothing at all. Two of them named workflows
/// that had been moved to `workflows-paused`, and the document went on claiming
/// them as lanes for as long as nobody opened the directory.
fn named_workflows(document: &str) -> Vec<(u32, String)> {
    let mut named = Vec::new();
    for (number, line) in crate::gates::scan::numbered(document) {
        let mut rest = line;
        while let Some(quoted) = next_quoted(&mut rest) {
            if quoted.ends_with(".yml") || quoted.ends_with(".yaml") {
                named.push((number, quoted));
            }
        }
    }
    named
}

/// The next backtick-quoted span, advancing past it.
fn next_quoted(text: &mut &str) -> Option<String> {
    let (_, after_open) = text.split_once('`')?;
    let (inner, after_close) = after_open.split_once('`')?;
    *text = after_close;
    (!inner.is_empty()).then(|| inner.to_string())
}

/// The text between the first pair of backticks.
fn quoted(text: &str) -> Option<String> {
    let (_, rest) = text.split_once('`')?;
    let (inner, _) = rest.split_once('`')?;
    (!inner.is_empty()).then(|| inner.to_string())
}

/// What is wrong with the triggers of a workflow that carries a required context.
///
/// A required check has to be reported on every pull request and on `main`, so a
/// path filter is the failure mode this looks for: a filtered workflow does not
/// run on an unrelated change, the context never reports, and branch protection
/// either blocks forever or is turned off.
fn trigger_findings(path: &str, text: &str) -> Vec<Finding> {
    let region = text.split("\njobs:").next().unwrap_or(text);
    let mut findings = Vec::new();

    let declares = |needle: &str| region.lines().any(|line| line.trim() == needle);
    if !declares("pull_request:") {
        findings.push(Finding::in_file(
            path,
            "the workflow does not run on `pull_request`",
            "trigger the workflow on pull requests; a required context that only runs after \
             a merge cannot block one",
        ));
    }
    if !declares("push:") {
        findings.push(Finding::in_file(
            path,
            "the workflow does not run on `push`",
            "trigger the workflow on pushes to the default branch so the required context has \
             a result on the branch it protects",
        ));
    } else if !region
        .lines()
        .any(|line| line.trim().starts_with("branches:") && line.contains("main"))
    {
        findings.push(Finding::in_file(
            path,
            "the `push` trigger does not name `main`",
            "push-trigger the workflow on `main`; the protected branch is the one whose \
             required contexts have to exist",
        ));
    }

    for (number, line) in crate::gates::scan::numbered(region) {
        let trimmed = line.trim();
        if trimmed.starts_with("paths:") || trimmed.starts_with("paths-ignore:") {
            findings.push(Finding::at(
                path,
                number,
                "a required workflow filters on paths",
                "delete the path filter; a filtered required check is skipped rather than \
                 reported, which is a merge blocked on a result that never comes",
            ));
        }
    }

    findings
}

/// What is wrong with a fan-in job that runs whatever its dependencies did.
///
/// `if: always()` is how a fan-in job reports on a failed dependency instead of
/// being skipped, and it is also how such a job reports success on one. The two
/// are told apart by whether the job reads a dependency result and exits nonzero,
/// so the rule applies exactly to the jobs that opt into always running.
fn fail_closed_findings(path: &str, job: &Job) -> Vec<Finding> {
    if !job.body.contains("always()") {
        return Vec::new();
    }
    let mut findings = Vec::new();
    if !job.body.contains(".result") {
        findings.push(Finding::at(
            path,
            job.line,
            format!(
                "job `{}` always runs without reading a dependency result",
                job.id
            ),
            "test `needs.<job>.result` for every dependency; a job that always runs and never \
             looks reports success for a lane that failed",
        ));
    }
    if !job.body.contains("exit 1") {
        findings.push(Finding::at(
            path,
            job.line,
            format!("job `{}` always runs without a failing exit", job.id),
            "exit nonzero when a dependency did not succeed; printing the failure and exiting \
             zero is a green required check over a red lane",
        ));
    }
    findings
}

/// One job of a workflow.
struct Job {
    /// The job id, which is the mapping key under `jobs:`.
    id: String,
    /// The display name, which is what branch protection reports when it is set.
    name: Option<String>,
    /// The line the job id is on.
    line: u32,
    /// Every line of the job, including its steps.
    body: String,
}

/// Every job a workflow declares.
///
/// The blocks are cut by indentation because that is what makes a job id
/// distinguishable from a key inside a job: `name:` at the job's own depth is the
/// display name, and the same key two levels deeper is a step name.
fn jobs(text: &str) -> Vec<Job> {
    let mut jobs: Vec<Job> = Vec::new();
    let mut inside = false;
    let mut depth = 0;
    for (number, line) in crate::gates::scan::numbered(text) {
        if line.trim_end() == "jobs:" {
            inside = true;
            continue;
        }
        if !inside || line.trim().is_empty() {
            if inside {
                if let Some(job) = jobs.last_mut() {
                    job.body.push('\n');
                }
            }
            continue;
        }
        let column = indentation(line);
        if column == 0 {
            inside = false;
            continue;
        }
        if jobs.is_empty() {
            depth = column;
        }
        if column == depth {
            let Some(id) = line.trim().strip_suffix(':') else {
                continue;
            };
            jobs.push(Job {
                id: id.to_string(),
                name: None,
                line: number,
                body: String::new(),
            });
            continue;
        }
        let Some(job) = jobs.last_mut() else {
            continue;
        };
        if column == depth + 2 {
            if let Some(name) = line.trim().strip_prefix("name:") {
                job.name = Some(name.trim().trim_matches('"').to_string());
            }
        }
        job.body.push_str(line);
        job.body.push('\n');
    }
    jobs
}

/// Every value declared by an axis of a strategy matrix.
///
/// An axis is a mapping entry inside the block a `matrix:` key opens, and its
/// values are an inline sequence or a block sequence. Reading the block rather
/// than the file is the point: a value in a step name, a job name or a comment is
/// not an axis, and a commented-out axis covers nothing.
fn matrix_axes(text: &str) -> BTreeSet<String> {
    let mut values = BTreeSet::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if line.trim() != "matrix:" {
            continue;
        }
        let opening = indentation(line);
        while let Some(inner) = lines.peek() {
            if inner.trim().is_empty() {
                lines.next();
                continue;
            }
            if indentation(inner) <= opening {
                break;
            }
            let inner = lines.next().unwrap_or_default();
            let trimmed = inner.trim();
            if let Some(item) = trimmed.strip_prefix("- ") {
                values.insert(item.trim().trim_matches('"').to_string());
                continue;
            }
            let Some((_, tail)) = trimmed.split_once(':') else {
                continue;
            };
            let tail = tail.trim();
            let Some(inline) = tail
                .strip_prefix('[')
                .and_then(|rest| rest.strip_suffix(']'))
            else {
                continue;
            };
            for value in inline.split(',') {
                let value = value.trim().trim_matches('"');
                if !value.is_empty() {
                    values.insert(value.to_string());
                }
            }
        }
    }
    values
}

/// Leading spaces on a line.
fn indentation(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MATRIX: &str = "jobs:\n  \
        matrix:\n    \
        name: stable somewhere\n    \
        strategy:\n      \
        matrix:\n        \
        os: [ubuntu-latest, macos-latest]\n        \
        rust-toolchain:\n          \
        - stable\n          \
        - nightly\n    \
        steps:\n      \
        - run: echo windows-latest\n";

    const WORKFLOW: &str = "name: Demo\n\
        \non:\n  \
        pull_request:\n  \
        push:\n    \
        branches: [main]\n\
        \njobs:\n  \
        build:\n    \
        runs-on: ubuntu-latest\n    \
        steps:\n      \
        - name: Compile\n        \
        run: exit 1\n\n  \
        demo-release-gate:\n    \
        name: Demo release gate\n    \
        needs:\n      \
        - build\n    \
        if: ${{ always() }}\n    \
        steps:\n      \
        - run: echo checked\n";

    /// WHY: an axis value is what the matrix expands, and a word in a step name
    /// is not. The rule this replaced asked whether the file contained the text
    /// anywhere, which a job name, a comment or an echoed string satisfies, so a
    /// deleted axis passed as long as the word survived somewhere.
    #[test]
    fn only_a_matrix_axis_value_counts() {
        let axes = matrix_axes(MATRIX);
        assert!(axes.contains("ubuntu-latest"));
        assert!(axes.contains("macos-latest"));
        assert!(
            !axes.contains("windows-latest"),
            "a step body is not an axis: {axes:?}"
        );
        assert!(
            !axes.contains("stable somewhere"),
            "a job name is not an axis: {axes:?}"
        );
    }

    /// WHY: both YAML sequence forms are live in this repository, and reading one
    /// of them would silently halve the axes the gate can see.
    #[test]
    fn both_sequence_forms_are_read() {
        let axes = matrix_axes(MATRIX);
        assert!(axes.contains("stable"), "block sequence: {axes:?}");
        assert!(axes.contains("nightly"), "block sequence: {axes:?}");
    }

    /// WHY: branch protection matches a context against the display name when a
    /// job sets one and against the job id when it does not, so both have to
    /// resolve. A step name at a deeper indentation is neither.
    #[test]
    fn a_job_is_named_by_its_display_name_or_its_id() {
        let jobs = jobs(WORKFLOW);
        let ids: Vec<&str> = jobs.iter().map(|job| job.id.as_str()).collect();
        assert_eq!(ids, ["build", "demo-release-gate"]);
        assert_eq!(jobs[0].name, None, "a step name is not a job name");
        assert_eq!(jobs[1].name.as_deref(), Some("Demo release gate"));
    }

    /// WHY: the fail-closed rule reads the job that opted into always running,
    /// not the file. A workflow that exits nonzero in an unrelated build step
    /// satisfied the rule this replaced, because that rule searched the whole
    /// file for the three tokens.
    #[test]
    fn a_fan_in_job_that_ignores_its_dependency_results_is_reported() {
        let jobs = jobs(WORKFLOW);
        let gate = jobs
            .iter()
            .find(|job| job.id == "demo-release-gate")
            .expect("the fan-in job is declared");
        let findings = fail_closed_findings("demo.yml", gate);
        let messages: Vec<&str> = findings
            .iter()
            .map(|finding| finding.message.as_str())
            .collect();
        assert_eq!(
            messages,
            [
                "job `demo-release-gate` always runs without reading a dependency result",
                "job `demo-release-gate` always runs without a failing exit",
            ],
            "the `exit 1` in the build job belongs to the build job"
        );
        assert!(
            fail_closed_findings("demo.yml", &jobs[0]).is_empty(),
            "a job that does not always run is not a fan-in job"
        );
    }

    /// WHY: `parse_required_ci_statuses` resolves the contexts under a heading,
    /// so a workflow file named as prose under the deep-gate heading was read by
    /// nothing. Two rows named workflows that had been moved to
    /// `workflows-paused` and the document kept claiming them as lanes. Every
    /// quoted file name on a line must be seen, not just the first, or a row that
    /// names two workflows hides the second.
    #[test]
    fn every_quoted_workflow_file_name_is_read() {
        let document = "# Required\n\n\
            The `ci-required` gate reads this.\n\n\
            ## From `ci.yml` (every PR)\n\
            - `CI release gate`\n\n\
            ## Scheduled or Manual Deep Gates\n\n\
            - `fuzz.yml`  -  once targets exist.\n\
            - `gone.yml` replaces `also-gone.yaml`.\n";
        assert_eq!(
            named_workflows(document),
            vec![
                (5, "ci.yml".to_string()),
                (10, "fuzz.yml".to_string()),
                (11, "gone.yml".to_string()),
                (11, "also-gone.yaml".to_string()),
            ]
        );
    }

    /// WHY: the live document is the payload of branch protection, so the rule
    /// has to hold on the tree it ships with. A rule that only passes on a
    /// fixture proves the fixture.
    #[test]
    fn the_live_required_document_names_only_workflows_that_exist() {
        let root = crate::checkout::checkout_root();
        let tree = Tree::open(&root).expect("Fix: the checkout must be listable");
        let document = tree
            .read(REQUIRED)
            .expect("Fix: the document must be readable");
        let named = named_workflows(&document);
        assert!(
            !named.is_empty(),
            "the document must name at least one workflow"
        );
        for (line, workflow) in named {
            assert!(
                tree.exists(&format!(".github/workflows/{workflow}"))
                    || tree.exists(&format!(".github/workflows-paused/{workflow}")),
                "{REQUIRED}:{line} names `{workflow}`, which is in neither workflow directory"
            );
        }
    }

    /// WHY: the rows under the deep-gate heading name lanes that report on a
    /// schedule, and two of them name workflows that are deliberately paused.
    /// Reading them as required contexts would report the pause as a defect.
    #[test]
    fn a_deep_gate_row_is_not_a_required_context() {
        let document = "# Required\n\n\
            ## From `ci.yml` (every PR)\n\
            - `CI release gate`\n\n\
            ## Scheduled or Manual Deep Gates\n\n\
            - `fuzz.yml`  -  once targets exist.\n";
        let sections = required_sections(document);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].workflow, "ci.yml");
        let contexts: Vec<&str> = sections[0]
            .contexts
            .iter()
            .map(|(_, context)| context.as_str())
            .collect();
        assert_eq!(contexts, ["CI release gate"]);
    }

    /// WHY: a path filter is the one trigger shape that makes a required check
    /// skip rather than fail, which reads as a pull request blocked on a result
    /// that never arrives.
    #[test]
    fn a_path_filter_on_a_required_workflow_is_reported() {
        let filtered = "name: Demo\n\non:\n  pull_request:\n    paths:\n      - src/**\n  \
            push:\n    branches: [main]\n\njobs:\n  build:\n    steps: []\n";
        let findings = trigger_findings("demo.yml", filtered);
        let messages: Vec<&str> = findings
            .iter()
            .map(|finding| finding.message.as_str())
            .collect();
        assert_eq!(messages, ["a required workflow filters on paths"]);
        assert!(
            trigger_findings("demo.yml", WORKFLOW).is_empty(),
            "the live trigger shape is the passing one"
        );
    }

    /// WHY: the registry is only as binding as the status context that reports
    /// it. `gates.yml` ran every registered gate on every pull request and was
    /// named nowhere in the required document, so a pull request could merge
    /// with the whole registry red. A release lane that runs the same sweep on a
    /// tag is not a pull request context and is deliberately not held to this.
    #[test]
    fn a_pull_request_sweep_is_told_from_a_release_sweep() {
        assert!(invokes_sweep("run: ./cargo_full run -p xtask -- gates --subset cat-a"));
        assert!(invokes_sweep(
            "run: ./cargo_full run -q -p xtask --bin xtask -- gates"
        ));
        assert!(
            !invokes_sweep("run: ./cargo_full run -p xtask -- gate-canon"),
            "one gate is not the sweep"
        );
        assert!(
            !invokes_sweep("run: ./cargo_full test -- --nocapture"),
            "a test argument is not a subcommand"
        );
        assert_eq!(
            workflow_file(".github/workflows/gates.yml"),
            Some("gates.yml".to_string())
        );
        assert_eq!(workflow_file(".github/workflows-paused/book.yml"), None);
        assert_eq!(workflow_file(".github/workflows/actions/x/action.yml"), None);

        let root = crate::checkout::checkout_root();
        let tree = Tree::open(&root).expect("Fix: the checkout must be listable");
        let document = tree
            .read(REQUIRED)
            .expect("Fix: the document must be readable");
        let blocking: BTreeSet<String> = required_sections(&document)
            .into_iter()
            .map(|section| section.workflow)
            .collect();
        let sweeps = sweep_workflows(&tree).expect("Fix: the workflows must be readable");
        assert!(
            !sweeps.is_empty(),
            "some workflow must run the sweep on a pull request"
        );
        for (workflow, line) in sweeps {
            assert!(
                blocking.contains(&workflow),
                "{workflow}:{line} runs the sweep on a pull request and {REQUIRED} does not \
                 make it blocking"
            );
        }
    }
}
