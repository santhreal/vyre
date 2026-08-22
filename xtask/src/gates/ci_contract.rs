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

use crate::gate::{Finding, GateCtx, GateError, Report};
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

impl crate::gate::GateBehavior for CiMatrix {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        report.cover_complete("ci matrix workflows", tree.members()?.len());

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

impl crate::gate::GateBehavior for CiRequired {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        report.cover_complete("ci required workflows", tree.members()?.len());

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
            report
                .findings
                .extend(baseline_comparison_findings(&path, &text));

            let jobs = jobs(&text);
            for (line, context) in &section.contexts {
                let Some(job) = jobs.iter().find(|job| job.reported() == context.as_str()) else {
                    let display = jobs
                        .iter()
                        .find(|job| job.id == *context)
                        .and_then(|job| job.name.clone());
                    let message = match display {
                        Some(name) => format!(
                            "`{context}` is a job id in `{}` and that job reports as `{name}`",
                            section.workflow
                        ),
                        None => format!("no job in `{}` reports as `{context}`", section.workflow),
                    };
                    report.find(Finding::at(
                        REQUIRED,
                        *line,
                        message,
                        "name the required context exactly as the job reports it, which is the \
                         job's `name:` when it declares one and the job id otherwise; branch \
                         protection matches on this string and a mismatch blocks every merge on \
                         a check that never arrives",
                    ));
                    continue;
                };
                report.findings.extend(fail_closed_findings(&path, job));
            }
        }

        Ok(report)
    }
}

/// The shell every step of a Windows-capable workflow is written for.
const SHELL: &str = "bash";

/// A construct the oldest bash this tree's matrix runs does not provide, and
/// what to write instead.
///
/// macOS ships bash 3.2 as `/bin/bash` and carries no newer one on `PATH`
/// before it, so `#!/usr/bin/env bash` resolves to 3.2 on that runner. A bash 4
/// builtin in a tracked script is therefore a script that only runs on Linux,
/// and it fails there by doing nothing visible: `wait -n` printed
/// `invalid option` for every worker in the release shard pool, the pool
/// counted each one as a failed shard, and the conformance job reported four
/// failed workers on a run whose shards were never started.
const LEGACY_BASH_GAPS: &[(&str, &str)] = &[
    (
        "wait -n",
        "keep the worker pids in an array and `wait \"$pid\"` for the oldest while the pool is \
         full; that reports the same per-worker status without the bash 4 builtin",
    ),
    (
        "mapfile",
        "read the stream into the array explicitly: `arr=(); while IFS= read -r line; do \
         arr+=(\"$line\"); done < <(command)`",
    ),
    (
        "readarray",
        "read the stream into the array explicitly: `arr=(); while IFS= read -r line; do \
         arr+=(\"$line\"); done < <(command)`",
    ),
    (
        "declare -A",
        "carry the pairs in two indexed arrays or a delimited string; bash 3.2 has no \
         associative array",
    ),
    (
        "local -A",
        "carry the pairs in two indexed arrays or a delimited string; bash 3.2 has no \
         associative array",
    ),
    (
        "typeset -A",
        "carry the pairs in two indexed arrays or a delimited string; bash 3.2 has no \
         associative array",
    ),
    (
        "|&",
        "write `2>&1 |`, which every bash in the matrix parses the same way",
    ),
    (
        "&>>",
        "write `>>file 2>&1`, which every bash in the matrix parses the same way",
    ),
];

/// True when the path names a shell script this gate reads.
fn is_shell_script(relative: &str) -> bool {
    let name = relative.rsplit('/').next().unwrap_or(relative);
    name.ends_with(".sh") || name.ends_with(".bash") || !name.contains('.')
}

/// The case-modification operator inside a parameter expansion on this line, if
/// any. `${v^^}`, `${v,,}`, `${v^}` and `${v,}` are all bash 4.
///
/// The operator has to follow the name and nothing else. A `,` or `^` that ends
/// some other expansion is not one: `${list:+$list,}` appends a literal comma
/// and `${sep:-,}` defaults to one, and both parse in bash 3.2.
fn case_modification_expansion(line: &str) -> Option<char> {
    let mut index = 0;
    while let Some(open) = line[index..].find("${") {
        let start = index + open + 2;
        let Some(close) = line[start..].find('}') else {
            return None;
        };
        let end = start + close;
        if let Some(operator) = case_modification_operator(&line[start..end]) {
            return Some(operator);
        }
        index = end + 1;
    }
    None
}

/// The case-modification operator that makes up the whole suffix of one
/// expansion body, if that is all the body carries after the name.
fn case_modification_operator(body: &str) -> Option<char> {
    let name_len = body
        .find(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .unwrap_or(body.len());
    if name_len == 0 {
        return None;
    }
    let mut suffix = &body[name_len..];
    // `${names[1]^^}` carries the same operator behind a subscript.
    if let Some(rest) = suffix.strip_prefix('[') {
        match rest.find(']') {
            Some(closing) => suffix = &rest[closing + 1..],
            None => return None,
        }
    }
    match suffix {
        "^" | "^^" => Some('^'),
        "," | ",," => Some(','),
        _ => None,
    }
}

/// Every bash 4 construct in one script.
fn legacy_bash_findings(relative: &str, text: &str) -> Vec<Finding> {
    let mut findings = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let numbered = u32::try_from(index + 1).unwrap_or(u32::MAX);
        if line.trim_start().starts_with('#') {
            continue;
        }
        for (construct, fix) in LEGACY_BASH_GAPS {
            if line.contains(construct) {
                findings.push(Finding::at(
                    relative,
                    numbered,
                    format!("`{construct}` needs bash 4"),
                    *fix,
                ));
            }
        }
        if let Some(operator) = case_modification_expansion(line) {
            findings.push(Finding::at(
                relative,
                numbered,
                format!("`${{name{operator}}}` case modification needs bash 4"),
                "fold the case with `tr '[:lower:]' '[:upper:]'` or compare both spellings; \
                 bash 3.2 has no case-modification expansion",
            ));
        }
    }
    findings
}

/// Every step of a workflow that runs on Windows declares the shell it is
/// written for, and every tracked shell script stays inside the language the
/// oldest bash in the matrix speaks.
///
/// A `run:` step with no `shell:` key runs under `bash` on a Linux or macOS
/// runner and under PowerShell on a Windows one, so the same script text is two
/// different programs depending on the axis. The one step in this tree that ever
/// named PowerShell could not run at all: `throw "no perl at $perl: ..."` reads
/// `$perl:` as a scope-qualified variable, which is a parse error, so the step
/// failed on every Windows job in the matrix and the failure said nothing about
/// perl. A shell nobody here writes for is a shell nobody here reviews.
///
/// The same argument reaches the scripts those steps call. One dialect per tree
/// means the dialect the whole matrix provides, so a bash 4 builtin belongs to
/// no lane here even though it runs on this workstation.
pub struct CiShell;

impl crate::gate::GateBehavior for CiShell {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let mut steps = 0;
        let mut scripts = 0;

        for path in tree.paths() {
            let relative = path.to_string_lossy().to_string();
            if is_shell_script(&relative) {
                let text = tree.read(path)?;
                if !text.starts_with("#!") {
                    continue;
                }
                scripts += 1;
                report
                    .findings
                    .extend(legacy_bash_findings(&relative, &text));
                continue;
            }
            if !is_workflow(&relative) {
                continue;
            }
            let text = tree.read(path)?;
            if !runs_on_windows(&text) {
                continue;
            }
            for step in run_steps(&text) {
                steps += 1;
                match step.shell.as_deref() {
                    Some(SHELL) => {}
                    Some(other) => report.find(Finding::at(
                        &relative,
                        step.line,
                        format!("the step is written for `{other}`"),
                        "write the step for bash and declare `shell: bash`; every Windows \
                         runner carries the Git bash this tree already writes every other \
                         step in, and a second dialect is a script the reviewer of this \
                         file cannot read",
                    )),
                    None => report.find(Finding::at(
                        &relative,
                        step.line,
                        "the step declares no shell",
                        "declare `shell: bash`; a step with no shell runs under PowerShell on \
                         a Windows runner, where the same text is a different program",
                    )),
                }
            }
        }

        report.cover_complete("windows-capable workflow steps", steps);
        report.cover_complete("tracked shell scripts", scripts);
        Ok(report)
    }
}

/// A superseded run holds the runner the current sha is waiting for.
///
/// Pushing again replaces the commit the previous run was measuring, so that run
/// reports on code no ref points at. On a hosted runner that only spends
/// minutes. On a self-hosted device runner, which takes one job at a time, the
/// obsolete run holds the device for as long as its queue takes and the run for
/// the current sha does not start until it finishes: the device lane then
/// reports a verdict from several pushes ago, or reports nothing before the
/// branch moves again.
///
/// The group has to vary by ref for the same reason. A constant group makes two
/// branches cancel each other, so a push to one branch discards the run another
/// branch was waiting on, which is a worse failure than the one the group was
/// added to fix.
///
/// A workflow that only a tag, a schedule or a dispatch starts is not held to
/// this. A tag is not superseded by the next push, and cancelling one would
/// discard a release recording.
pub struct CiConcurrency;

impl crate::gate::GateBehavior for CiConcurrency {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let mut workflows = 0;

        for path in tree.paths() {
            let relative = path.to_string_lossy().to_string();
            if !is_workflow(&relative) {
                continue;
            }
            let text = tree.read(path)?;
            if !triggered_by_a_change(&text) {
                continue;
            }
            workflows += 1;
            let Some(concurrency) = concurrency_block(&text) else {
                report.find(Finding::in_file(
                    &relative,
                    "a change starts this workflow and it declares no `concurrency` group",
                    "add a top-level `concurrency:` with `group: <workflow>-${{ github.ref }}` \
                     and `cancel-in-progress: true`, so the next push retires the run it \
                     superseded instead of queueing behind it",
                ));
                continue;
            };
            match concurrency.group.as_deref() {
                None => report.find(Finding::at(
                    &relative,
                    concurrency.line,
                    "the `concurrency` block names no group",
                    "give it `group: <workflow>-${{ github.ref }}`; a block with no group \
                     key is not a concurrency group at all",
                )),
                Some(group) if !varies_by_ref(group) => report.find(Finding::at(
                    &relative,
                    concurrency.line,
                    format!("the concurrency group `{group}` is the same on every ref"),
                    "put `${{ github.ref }}` in the group; a constant group makes a push to \
                     one branch cancel the run another branch is waiting on",
                )),
                Some(_) => {}
            }
            match concurrency.cancel_in_progress.as_deref() {
                Some("true") => {}
                Some(other) => report.find(Finding::at(
                    &relative,
                    concurrency.line,
                    format!("the group is declared with `cancel-in-progress: {other}`"),
                    "set `cancel-in-progress: true`; a group that queues instead of \
                     cancelling makes the obsolete run hold the runner first",
                )),
                None => report.find(Finding::at(
                    &relative,
                    concurrency.line,
                    "the group does not declare `cancel-in-progress`",
                    "set `cancel-in-progress: true`; the default queues the new run behind \
                     the superseded one, which is the delay the group exists to remove",
                )),
            }
        }

        report.cover_complete("change-triggered workflows", workflows);
        Ok(report)
    }
}

/// Whether a group expression distinguishes one ref from another.
fn varies_by_ref(group: &str) -> bool {
    group.contains("github.ref") || group.contains("github.head_ref")
}

/// The top-level `concurrency:` mapping of a workflow.
struct Concurrency {
    /// One-based line the block opens on.
    line: u32,
    /// The `group:` expression, verbatim.
    group: Option<String>,
    /// The `cancel-in-progress:` value, verbatim.
    cancel_in_progress: Option<String>,
}

/// The top-level `concurrency:` mapping, if the workflow declares one.
///
/// Only a key at column zero is the workflow's own: `concurrency:` indented
/// inside a job scopes that job alone and does not retire the run.
fn concurrency_block(text: &str) -> Option<Concurrency> {
    let mut found: Option<Concurrency> = None;
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        if indentation(line) == 0 {
            if found.is_some() {
                break;
            }
            let Some(rest) = line.strip_prefix("concurrency:") else {
                continue;
            };
            let mut opened = Concurrency {
                line: index as u32 + 1,
                group: None,
                cancel_in_progress: None,
            };
            read_concurrency_keys(rest, &mut opened);
            found = Some(opened);
            continue;
        }
        if let Some(opened) = found.as_mut() {
            read_concurrency_keys(line, opened);
        }
    }
    found
}

/// Read whichever concurrency keys a fragment carries into the block.
///
/// The fragment is either an indented line of a block mapping or the remainder
/// of an inline one, so a flow mapping is unwrapped once at its own boundaries.
/// Stripping braces from each entry instead would eat the closing `}}` of a
/// `${{ github.ref }}` group and read it as a constant.
fn read_concurrency_keys(fragment: &str, block: &mut Concurrency) {
    let fragment = fragment.trim();
    let fragment = fragment
        .strip_prefix('{')
        .map_or(fragment, |rest| rest.strip_suffix('}').unwrap_or(rest));
    for part in fragment.split(',') {
        let part = part.trim();
        if let Some(value) = part.strip_prefix("group:") {
            block.group = Some(
                value
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_string(),
            );
        } else if let Some(value) = part.strip_prefix("cancel-in-progress:") {
            block.cancel_in_progress = Some(value.trim().to_string());
        }
    }
}

/// Whether a push to a branch or a pull request starts the workflow.
///
/// A `push:` that filters on `tags:` and names no `branches:` is a tag trigger,
/// and a tag is not superseded by the next push.
fn triggered_by_a_change(text: &str) -> bool {
    let mut inside_on = false;
    let mut inside_push = false;
    let mut push_declared = false;
    let mut push_branches = false;
    let mut push_tags = false;
    for line in text.lines() {
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let indent = indentation(line);
        if indent == 0 {
            if inside_on {
                break;
            }
            let Some(rest) = line
                .strip_prefix("on:")
                .or_else(|| line.strip_prefix("\"on\":"))
                .or_else(|| line.strip_prefix("'on':"))
            else {
                continue;
            };
            if rest.contains("pull_request") || rest.contains("push") {
                return true;
            }
            inside_on = true;
            continue;
        }
        if !inside_on {
            continue;
        }
        let trimmed = line.trim();
        if indent <= 2 {
            inside_push = false;
            if trimmed.starts_with("pull_request:") || trimmed.starts_with("pull_request_target:") {
                return true;
            }
            if trimmed.starts_with("push:") {
                inside_push = true;
                push_declared = true;
            }
            continue;
        }
        if inside_push {
            if trimmed.starts_with("branches:") {
                push_branches = true;
            } else if trimmed.starts_with("tags:") {
                push_tags = true;
            }
        }
    }
    push_declared && (push_branches || !push_tags)
}

/// Whether the path is a workflow file, running or paused.
fn is_workflow(relative: &str) -> bool {
    let Some(name) = relative
        .strip_prefix(".github/workflows/")
        .or_else(|| relative.strip_prefix(".github/workflows-paused/"))
    else {
        return false;
    };
    !name.contains('/') && (name.ends_with(".yml") || name.ends_with(".yaml"))
}

/// Whether any job of the workflow can run on a Windows runner.
///
/// A label is a `runs-on:` value, an `os:` axis value, or a bare sequence entry,
/// which is how a block-style axis writes one. Nothing else is a label: a
/// sequence entry carrying a key is a step, so `- name: windows-latest notes`
/// names a step and puts no job on Windows, and neither does a comment or a
/// command that mentions the label.
fn runs_on_windows(text: &str) -> bool {
    text.lines().any(|line| {
        let code = line.trim();
        if code.starts_with('#') {
            return false;
        }
        let value = if let Some(value) = code.strip_prefix("runs-on:") {
            value
        } else if let Some(value) = code.strip_prefix("os:") {
            value
        } else if let Some(entry) = code.strip_prefix("- ") {
            if entry.contains(':') {
                return false;
            }
            entry
        } else {
            return false;
        };
        value.contains("windows-")
    })
}

/// One step of a workflow that runs a command.
struct RunStep {
    /// The line the `run:` key is on.
    line: u32,
    /// The shell the step declares, if it declares one.
    shell: Option<String>,
}

/// Every step of the workflow that runs a command, with the shell it declares.
///
/// A step is a sequence entry, so an entry at the depth the current one opened
/// starts the next step and a shallower line ends the list. Keys are read only at
/// the step's own depth: the body of a `run:` block scalar is indented past it,
/// and a line of shell that happens to read `shell: pwsh` or `- item` is script
/// text rather than a key. The order of the keys inside a step does not matter,
/// so a step that declares its shell after its command is read the same as one
/// that declares it before.
fn run_steps(text: &str) -> Vec<RunStep> {
    let mut steps = Vec::new();
    let mut open: Option<usize> = None;
    let mut run: Option<u32> = None;
    let mut shell: Option<String> = None;
    for (number, line) in crate::gates::scan::numbered(text) {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let column = indentation(line);
        let entry = trimmed.starts_with("- ") && open.is_none_or(|start| column <= start);
        let closed = open.is_some_and(|start| column < start);
        if entry || closed {
            if let Some(line) = run.take() {
                steps.push(RunStep {
                    line,
                    shell: shell.take(),
                });
            }
            run = None;
            shell = None;
            open = entry.then_some(column);
        }
        let Some(start) = open else {
            continue;
        };
        if !entry && column != start + 2 {
            continue;
        }
        let key = trimmed.strip_prefix("- ").unwrap_or(trimmed);
        if key.starts_with("run:") {
            run.get_or_insert(number);
        }
        if let Some(value) = key.strip_prefix("shell:") {
            shell = Some(value.trim().to_string());
        }
    }
    if let Some(line) = run {
        steps.push(RunStep { line, shell });
    }
    steps
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

/// What is wrong with a benchmark baseline step in a workflow.
///
/// A benchmark regression check that compares a PR against a baseline must derive
/// that baseline from the target branch (`origin/main`), not from the PR head itself.
/// Falling back to self-comparison turns a regression gate into a vacuous self-check.
fn baseline_comparison_findings(path: &str, text: &str) -> Vec<Finding> {
    let mut findings = Vec::new();
    for (number, line) in crate::gates::scan::numbered(text) {
        let trimmed = line.trim();
        if trimmed.contains("--save-baseline")
            && (trimmed.contains("fallback")
                || trimmed.contains("PR head")
                || trimmed.contains("self-baseline")
                || trimmed.contains("self-comparison"))
        {
            findings.push(Finding::at(
                path,
                number,
                "baseline step permits fallback to self-comparison on PR head",
                "benchmark the target branch implementation directly by materializing the \
                 benchmark target into the main checkout; self-comparison is a vacuous gate",
            ));
        }
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

impl Job {
    /// The status context this job reports, which is the display name when the
    /// job declares one and the job id otherwise.
    ///
    /// Branch protection matches this string and nothing else, so a required
    /// check named by job id where the job declares a `name:` waits for a
    /// context that never arrives.
    fn reported(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.id)
    }
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
        assert!(invokes_sweep(
            "run: ./cargo_full run -p xtask -- gates --subset cat-a"
        ));
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
        assert_eq!(
            workflow_file(".github/workflows/actions/x/action.yml"),
            None
        );

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

    /// WHY: A benchmark regression check that compares a PR against a baseline must
    /// derive that baseline from the target branch (origin/main), not from the PR head
    /// itself. Falling back to self-comparison turns a regression gate into a vacuous check.
    #[test]
    fn a_baseline_step_permitting_self_comparison_is_reported() {
        let bad_workflow = "\
jobs:
  bench:
    steps:
      - name: Save baseline
        run: |
          git checkout -
          ./cargo_full bench -p vyre-foundation --bench optimizer_pipeline -- --save-baseline main fallback to PR head
";
        let findings = baseline_comparison_findings("bench.yml", bad_workflow);
        assert_eq!(findings.len(), 1);
        assert!(findings[0].message.contains("fallback to self-comparison"));

        let passing_workflow = "\
jobs:
  bench:
    steps:
      - name: Save baseline from main
        run: |
          set -euo pipefail
          git fetch origin main
          git checkout origin/main
          git checkout - -- vyre-foundation/benches
          if ! grep -q 'name = \"optimizer_pipeline\"' vyre-foundation/Cargo.toml; then
            printf '\\n[[bench]]\\nname = \"optimizer_pipeline\"\\nharness = false\\n' >> vyre-foundation/Cargo.toml
          fi
          ./cargo_full bench -p vyre-foundation --bench optimizer_pipeline -- --save-baseline main
          git checkout -f -
";
        let clean_findings = baseline_comparison_findings("bench.yml", passing_workflow);
        assert!(clean_findings.is_empty(), "clean workflow must pass");
    }

    /// A workflow whose steps are the four shapes the rule decides between: a
    /// step that declares bash, one that declares another dialect, one that
    /// declares nothing, and a step whose command body contains lines that read
    /// like keys.
    const SHELLS: &str = "\
jobs:
  matrix:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
    steps:
      - name: declared
        shell: bash
        run: ./cargo_full fmt --check

      - name: other dialect
        shell: pwsh
        run: Write-Output 'hello'

      - name: undeclared
        run: ./cargo_full test --workspace

      - name: body that reads like keys
        run: |
          shell: pwsh
          - run: echo not a step
        shell: bash
";

    /// WHY: a `run:` step with no `shell:` key runs under PowerShell on a Windows
    /// runner, so the rule has to see the absence of a key rather than the
    /// presence of a wrong one. This pins all three verdicts the gate acts on and
    /// goes red if the reader stops at the first step or treats a missing key as
    /// bash. What it does not catch is a step that declares bash and then writes
    /// PowerShell in it.
    #[test]
    fn a_step_declares_no_shell_when_no_key_at_its_depth_does() {
        let steps = run_steps(SHELLS);

        let shells: Vec<Option<&str>> = steps.iter().map(|step| step.shell.as_deref()).collect();
        assert_eq!(
            shells,
            vec![Some("bash"), Some("pwsh"), None, Some("bash")],
            "Fix: every step's shell is the key at its own depth, or nothing."
        );
        assert_eq!(
            steps.iter().map(|step| step.line).collect::<Vec<_>>(),
            vec![9, 13, 16, 19],
            "Fix: a finding names the line the command is on."
        );
    }

    /// WHY: the body of a block scalar is shell text, and shell text that reads
    /// like YAML is still shell text. Reading a body line as a key made the last
    /// step of the fixture declare `pwsh` and swallowed the step boundary, which
    /// is how a reader that scans for `shell:` per file reports the wrong step.
    #[test]
    fn a_run_body_line_that_reads_like_a_key_is_script_text() {
        let steps = run_steps(SHELLS);

        assert_eq!(
            steps.len(),
            4,
            "Fix: the fixture has four steps; a body line is not a fifth."
        );
        assert_eq!(
            steps[3].shell.as_deref(),
            Some("bash"),
            "Fix: the last step declares bash after its command, and its body \
             names pwsh only as script text."
        );
    }

    /// WHY: the rule only applies to a workflow a Windows runner executes, and
    /// the word is not the runner. A step named after Windows, and a comment
    /// about it, put no job on one, and reading either would hold every workflow
    /// in the tree to a rule none of them needs.
    #[test]
    fn only_a_runner_label_puts_a_job_on_windows() {
        assert!(
            runs_on_windows(SHELLS),
            "Fix: a matrix axis value is a runner label."
        );
        assert!(
            runs_on_windows("jobs:\n  build:\n    runs-on: windows-latest\n"),
            "Fix: a runs-on value is a runner label."
        );
        assert!(
            !runs_on_windows(
                "jobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n      \
                 - name: windows-latest notes\n        # windows-latest is discussed here\n        \
                 run: echo windows-latest\n"
            ),
            "Fix: a step name, a comment and a command are not runner labels."
        );
    }

    /// WHY: the workflow set is read from the two directories workflows live in,
    /// so a paused workflow is held to the rule it has to satisfy to return, and
    /// a file that is not a workflow is not read at all.
    #[test]
    fn a_workflow_is_a_yaml_file_in_one_of_the_two_workflow_directories() {
        assert!(is_workflow(".github/workflows/ci.yml"));
        assert!(is_workflow(".github/workflows-paused/strict.yaml"));
        assert!(!is_workflow(".github/workflows/nested/ci.yml"));
        assert!(!is_workflow(".github/CI_REQUIRED.md"));
        assert!(!is_workflow("xtask/ci-registry.toml"));
    }

    /// WHY: the rule exempts a lane no push supersedes, and the exemption is
    /// what a stale workflow would hide behind. A `push:` filtered to tags is
    /// exempt; the same key filtered to branches is not, and neither is a
    /// workflow that also answers a pull request.
    #[test]
    fn only_a_lane_a_push_supersedes_is_held_to_a_concurrency_group() {
        let tag_only = "name: Release\non:\n  workflow_dispatch:\n  push:\n    tags: [\"v*\"]\n";
        let branch_push = "name: CI\non:\n  push:\n    branches: [main]\n";
        let schedule_only = "name: Nightly\non:\n  schedule:\n    - cron: '0 0 * * *'\n";
        let pull_request = "name: Docs\non:\n  pull_request:\n";
        assert!(!triggered_by_a_change(tag_only), "a tag is not superseded");
        assert!(
            !triggered_by_a_change(schedule_only),
            "a cron has no ref to supersede"
        );
        assert!(triggered_by_a_change(branch_push));
        assert!(triggered_by_a_change(pull_request));
    }

    /// WHY: `concurrency:` inside a job scopes that job and leaves the run it
    /// superseded holding the runner, which is the exact failure the rule
    /// exists for. Only a key at column zero answers it.
    #[test]
    fn a_job_scoped_concurrency_key_is_not_the_workflow_group() {
        let text = "name: CI\non:\n  pull_request:\njobs:\n  build:\n    \
                    concurrency:\n      group: build\n      cancel-in-progress: true\n";
        assert!(
            concurrency_block(text).is_none(),
            "a job-scoped group is not the workflow's"
        );
    }

    /// WHY: both spellings GitHub accepts have to read the same, or the rule
    /// reports a workflow that already retires its superseded run.
    #[test]
    fn a_group_reads_the_same_written_inline_or_as_a_block() {
        let block = "on:\n  pull_request:\nconcurrency:\n  group: ci-${{ github.ref }}\n  \
                     cancel-in-progress: true\n";
        let inline = "on:\n  pull_request:\nconcurrency: { group: ci-${{ github.ref }}, \
                      cancel-in-progress: true }\n";
        for text in [block, inline] {
            let parsed = concurrency_block(text).expect("the workflow declares a group");
            assert_eq!(parsed.group.as_deref(), Some("ci-${{ github.ref }}"));
            assert_eq!(parsed.cancel_in_progress.as_deref(), Some("true"));
            assert!(varies_by_ref(parsed.group.as_deref().expect("a group")));
        }
    }

    /// WHY: a constant group is worse than none, because a push to one branch
    /// then cancels the run another branch is waiting on.
    #[test]
    fn a_group_that_is_the_same_on_every_ref_does_not_vary_by_ref() {
        assert!(!varies_by_ref("gates"));
        assert!(varies_by_ref("gates-${{ github.ref }}"));
        assert!(varies_by_ref("gates-${{ github.head_ref }}"));
    }

    /// WHY: every one of these lines was tracked in this tree and ran fine on
    /// Linux. `wait -n` is the one that reached CI: the macOS runner printed
    /// `wait: -n: invalid option` once per worker, the shard pool counted each
    /// as a failed shard, and the job reported four failed workers for a run
    /// whose shards never started. A scan that cannot name these lines would
    /// certify a portability claim it never checked.
    #[test]
    fn the_scan_names_every_bash_four_construct_this_tree_has_carried() {
        let script = "#!/usr/bin/env bash\n\
                      if ! wait -n; then failures=1; fi\n\
                      mapfile -t CONTEXTS < <(awk '{print}' \"$DOC\")\n\
                      readarray -t rows <<< \"$output\"\n\
                      declare -A seen\n\
                      if [[ \"${repo_visibility^^}\" != PUBLIC ]]; then exit 2; fi\n\
                      printf '%s' \"${name,}\"\n\
                      command |& tee log\n\
                      command &>> log\n";
        let findings = legacy_bash_findings("scripts/sample.sh", script);
        let lines: Vec<Option<u32>> = findings.iter().map(|finding| finding.line).collect();
        assert_eq!(
            lines,
            vec![
                Some(2),
                Some(3),
                Some(4),
                Some(5),
                Some(6),
                Some(7),
                Some(8),
                Some(9)
            ],
            "Fix: the scan must name one finding per offending line: {findings:#?}"
        );
    }

    /// WHY: a scan that flags the expansions bash 3.2 does have would push
    /// authors to rewrite working scripts, and a gate nobody trusts gets
    /// silenced. Every form here is in the tree today.
    #[test]
    fn the_scan_leaves_expansions_bash_three_already_has() {
        let script = "#!/usr/bin/env bash\n\
                      printf '%s' \"${PUBLISH_ENTRIES[0]}\"\n\
                      printf '%s' \"${#PUBLISH_ENTRIES[@]}\"\n\
                      printf '%s' \"${VYRE_RELEASE_BACKEND:-all}\"\n\
                      printf '%s' \"${BASH_SOURCE[0]}\"\n\
                      printf '%s' \"${path//,/ }\"\n\
                      printf '%s' \"${compared:+$compared,}\"\n\
                      printf '%s' \"${separator:-,}\"\n\
                      printf '%s' \"${prefix%,}\"\n\
                      # mapfile -t stale < <(true)\n\
                      wait \"${worker_pids[joined]}\"\n";
        let findings = legacy_bash_findings("scripts/sample.sh", script);
        assert!(
            findings.is_empty(),
            "Fix: these are bash 3.2 constructs and a comment, and none may be reported: {findings:#?}"
        );
    }

    /// WHY: the operator is the same construct behind a subscript, and a scan
    /// that stops at the name would pass `${row[0]^^}` on to a 3.2 runner.
    #[test]
    fn the_scan_names_a_case_operator_behind_a_subscript() {
        assert_eq!(case_modification_operator("row[0]^^"), Some('^'));
        assert_eq!(case_modification_operator("row[index],"), Some(','));
        assert_eq!(case_modification_operator("row[0]"), None);
        assert_eq!(case_modification_operator("row[0"), None);
    }

    /// WHY: the scan reads shell scripts, and the tree's own entry point
    /// (`cargo_full`) carries no extension at all.
    #[test]
    fn a_shell_script_is_recognized_with_or_without_an_extension() {
        assert!(is_shell_script("scripts/prove-release-shards.sh"));
        assert!(is_shell_script("scripts/lib/toml_reader.sh"));
        assert!(is_shell_script("cargo_full"));
        assert!(!is_shell_script("xtask/src/gates/ci_contract.rs"));
        assert!(!is_shell_script(".github/workflows/ci.yml"));
    }
}
