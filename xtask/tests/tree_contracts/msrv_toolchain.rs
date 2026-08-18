//! The toolchain a workflow installs is the only thing `--print-toolchain`
//! writes.
//!
//! WHY: the gates workflow installs the MSRV toolchain from the first stdout
//! line of `xtask feature-msrv --print-toolchain` and hands that line to
//! `rustup toolchain install`. The coupling is a contract with no declaration:
//! a banner printed by the dispatcher, a note rendered before the early return,
//! or a second line of any kind is fed to rustup, which then installs a
//! toolchain named after a sentence, or installs nothing and leaves the sweep
//! running on the default. This drives the real binary so the whole path is
//! covered, dispatcher included, and reads the expected version out of the root
//! manifest rather than restating it.

#![forbid(unsafe_code)]

use std::process::Command;

use crate::workspace_sources::workspace_root;

#[test]
fn print_toolchain_writes_the_advertised_version_and_nothing_else() {
    let root = workspace_root();
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["feature-msrv", "--print-toolchain"])
        .current_dir(&root)
        .output()
        .expect("Fix: the runner must launch");
    assert!(
        output.status.success(),
        "Fix: the gate must exit zero when it is only asked for the version: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().filter(|line| !line.is_empty()).collect();
    assert_eq!(
        lines.len(),
        1,
        "Fix: the workflow installs the first stdout line, so this mode writes one line: {stdout:?}"
    );

    let manifest = std::fs::read_to_string(root.join("Cargo.toml"))
        .expect("Fix: the root manifest must be readable");
    let table: toml::Table = toml::from_str(&manifest).expect("Fix: the root manifest must parse");
    let advertised = table
        .get("workspace")
        .and_then(|workspace| workspace.get("package"))
        .and_then(|package| package.get("rust-version"))
        .and_then(toml::Value::as_str)
        .expect("Fix: [workspace.package] must declare rust-version");
    assert_eq!(
        stdout.as_ref(),
        format!("{advertised}\n"),
        "Fix: print mode must output exactly the advertised version followed by a newline and nothing else"
    );
    assert_eq!(
        lines[0], advertised,
        "Fix: the printed toolchain is the advertised minimum"
    );
    assert!(
        lines[0]
            .split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit())),
        "Fix: rustup installs a version, and `{}` is not one",
        lines[0]
    );
}

/// WHY: print mode is an early exit that returns only the version; normal
/// execution of feature-msrv without --print-toolchain must still render the
/// complete gate report with notes and finding summary.
#[test]
fn normal_feature_msrv_preserves_standard_gate_report() {
    let root = workspace_root();
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["feature-msrv"])
        .current_dir(&root)
        .output()
        .expect("Fix: the runner must launch");
    assert!(
        output.status.success(),
        "Fix: the gate must exit zero on normal execution: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("feature-msrv: note: advertised rust-version"),
        "Fix: normal gate execution must include report notes: {stdout}"
    );
    assert!(
        stdout.ends_with("feature-msrv: 0 finding(s)\n"),
        "Fix: normal gate execution must end with standard finding count summary: {stdout}"
    );
}

/// WHY: `cargo-fuzz` installs on stable, but the libFuzzer runner passes `-Z`
/// sanitizer flags and therefore requires nightly. The smoke workflow installed
/// and selected stable, so every fuzz target failed before reading one byte.
/// The job set is derived from the workflow so a new fuzz job is covered.
#[test]
fn every_cargo_fuzz_job_selects_nightly() {
    let root = workspace_root();
    let workflow = std::fs::read_to_string(root.join(".github/workflows/fuzz.yml"))
        .expect("Fix: the fuzz workflow must be readable");
    let mut job = "<none>";
    let mut toolchain = None;
    let mut installs = 0usize;
    let mut runs = 0usize;

    for line in workflow.lines() {
        if line.starts_with("  ") && !line.starts_with("    ") && line.trim_end().ends_with(':') {
            job = line.trim().trim_end_matches(':');
            toolchain = None;
        }

        let trimmed = line.trim();
        if let Some(selected) = trimmed.strip_prefix("- uses: dtolnay/rust-toolchain@") {
            toolchain = Some(selected);
        }
        if trimmed.contains("install --locked cargo-fuzz") {
            installs += 1;
            assert_eq!(
                toolchain,
                Some("nightly"),
                "Fix: fuzz job `{job}` must select nightly before installing cargo-fuzz"
            );
            assert_eq!(
                trimmed, "run: cargo +nightly install --locked cargo-fuzz",
                "Fix: fuzz job `{job}` must install cargo-fuzz with the selected nightly toolchain"
            );
        }

        if trimmed.contains("cargo_full") && trimmed.contains("fuzz run") {
            runs += 1;
            assert_eq!(
                toolchain,
                Some("nightly"),
                "Fix: fuzz job `{job}` must select nightly before running cargo-fuzz"
            );
            assert!(
                trimmed.starts_with("../../cargo_full +nightly fuzz run "),
                "Fix: fuzz job `{job}` must invoke cargo-fuzz through the nightly toolchain: {trimmed}"
            );
        }
    }

    assert!(
        installs > 0,
        "Fix: the fuzz workflow contains no cargo-fuzz install, so this contract guards nothing"
    );
    assert_eq!(
        runs, installs,
        "Fix: every cargo-fuzz install must have one nightly cargo-fuzz run"
    );
}

fn workflow_jobs(workflow: &str) -> Vec<(String, String)> {
    let mut jobs = Vec::new();
    let mut current: Option<(String, String)> = None;
    for line in workflow.lines() {
        if line.starts_with("  ") && !line.starts_with("    ") && line.trim_end().ends_with(':') {
            if let Some(job) = current.take() {
                jobs.push(job);
            }
            current = Some((line.trim().trim_end_matches(':').to_string(), String::new()));
        } else if let Some((_, body)) = &mut current {
            body.push_str(line);
            body.push('\n');
        }
    }
    if let Some(job) = current {
        jobs.push(job);
    }
    jobs
}

fn toolchains_selected_at_public_api_installs(
    workflow_job: &str,
) -> Result<Vec<(String, String)>, String> {
    let lines = workflow_job.lines().collect::<Vec<_>>();
    let mut selected = None;
    let mut installs = Vec::new();

    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if let Some(action_ref) = trimmed
            .strip_prefix("- uses: dtolnay/rust-toolchain@")
            .or_else(|| trimmed.strip_prefix("uses: dtolnay/rust-toolchain@"))
        {
            let action_ref = action_ref.trim();
            if action_ref.is_empty() || action_ref.chars().any(char::is_whitespace) {
                return Err(format!(
                    "line {} has a malformed dtolnay/rust-toolchain action ref",
                    index + 1
                ));
            }

            let action_indent = line.len() - line.trim_start().len();
            let step_end = lines[index + 1..]
                .iter()
                .position(|candidate| {
                    let candidate_indent = candidate.len() - candidate.trim_start().len();
                    candidate_indent <= action_indent && candidate.trim_start().starts_with("- ")
                })
                .map_or(lines.len(), |offset| index + 1 + offset);
            let mut inputs = Vec::new();
            let mut with_indent = None;
            for candidate in &lines[index + 1..step_end] {
                let candidate_indent = candidate.len() - candidate.trim_start().len();
                let candidate = candidate.trim();
                if candidate == "with:" {
                    with_indent = Some(candidate_indent);
                    continue;
                }
                let Some(parent_indent) = with_indent else {
                    continue;
                };
                if !candidate.is_empty() && candidate_indent <= parent_indent {
                    with_indent = None;
                    continue;
                }
                if let Some(input) = candidate.strip_prefix("toolchain:").map(str::trim) {
                    inputs.push(input);
                }
            }
            if inputs.len() > 1 || inputs.first().is_some_and(|input| input.is_empty()) {
                return Err(format!(
                    "line {} has malformed or repeated toolchain inputs",
                    index + 1
                ));
            }
            let input = inputs.first().copied();
            if action_ref == "master" && input.is_none() {
                return Err(format!(
                    "line {} selects @master without an explicit toolchain input",
                    index + 1
                ));
            }
            selected = Some(input.unwrap_or(action_ref).to_string());
        }

        if trimmed.contains("install --locked cargo-public-api") {
            let toolchain = selected.clone().ok_or_else(|| {
                format!(
                    "line {} installs cargo-public-api before selecting a Rust toolchain",
                    index + 1
                )
            })?;
            installs.push((toolchain, trimmed.to_string()));
        }
    }

    Ok(installs)
}

/// WHY: dated Rust releases can be expressed either as the action ref or as
/// the `toolchain` input on `@master`. The workflow contract must accept both
/// valid shapes and fail closed when either selector is incomplete.
#[test]
fn rust_toolchain_action_parser_accepts_both_pinned_forms_and_rejects_gaps() {
    let direct = "\
      - uses: dtolnay/rust-toolchain@nightly-2026-08-07
      - run: cargo +nightly-2026-08-07 install --locked cargo-public-api
";
    assert_eq!(
        toolchains_selected_at_public_api_installs(direct).unwrap(),
        vec![(
            "nightly-2026-08-07".to_string(),
            "- run: cargo +nightly-2026-08-07 install --locked cargo-public-api".to_string()
        )]
    );

    let explicit_input = "\
      - name: Install the pinned rustdoc
        uses: dtolnay/rust-toolchain@master
        with:
          toolchain: nightly-2026-08-07
      - run: cargo +nightly-2026-08-07 install --locked cargo-public-api
";
    assert_eq!(
        toolchains_selected_at_public_api_installs(explicit_input)
            .unwrap()
            .first()
            .map(|event| event.0.as_str()),
        Some("nightly-2026-08-07")
    );

    for malformed in [
        "      - uses: dtolnay/rust-toolchain@\n",
        "      - uses: dtolnay/rust-toolchain@master\n",
        "      - run: cargo +nightly install --locked cargo-public-api\n",
        "      - uses: dtolnay/rust-toolchain@master\n        toolchain: nightly-2026-08-07\n",
        "      - uses: dtolnay/rust-toolchain@master\n        with:\n          toolchain:\n",
    ] {
        assert!(
            toolchains_selected_at_public_api_installs(malformed).is_err(),
            "malformed selector must fail closed: {malformed:?}"
        );
    }
}

/// WHY: the public-API snapshot records rustdoc's rendering of every item path,
/// and that rendering moves with the compiler. The release that re-homed
/// `std::io::Error` under `core` rewrote nine committed snapshots with no source
/// change behind them, so a floating `nightly` turns the gate red on a date. The
/// date is declared once by `RUSTDOC_TOOLCHAIN` and exported onto the
/// extraction, which makes it a build requirement of every job that runs the
/// extraction, not only of the job named after it: `tree_contracts` drives the
/// same gate. The job set is derived from the workflows at run time so a new job
/// that reaches the extraction is covered without being listed here.
///
/// It does not catch a job that reaches the extraction through a command spelled
/// in neither form, such as a script that runs the gate binary by path.
#[test]
fn every_job_running_the_public_api_extraction_installs_the_declared_rustdoc() {
    let root = workspace_root();
    let pinned = xtask::gates::public_api::RUSTDOC_TOOLCHAIN;
    let version = xtask::gates::public_api::CARGO_PUBLIC_API_VERSION;

    assert!(
        pinned.starts_with("nightly-") && pinned.len() > "nightly-".len(),
        "Fix: RUSTDOC_TOOLCHAIN must name a dated nightly; a floating channel makes the \
         snapshot a function of the calendar rather than of the tree"
    );

    let directory = root.join(".github/workflows");
    let mut reached = 0usize;
    for entry in
        std::fs::read_dir(&directory).expect("Fix: the workflow directory must be readable")
    {
        let path = entry
            .expect("Fix: every workflow entry must be readable")
            .path();
        if path.extension().and_then(|value| value.to_str()) != Some("yml") {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .expect("Fix: a workflow file name must be UTF-8");
        let workflow =
            std::fs::read_to_string(&path).expect("Fix: every workflow file must be readable");

        for (job, body) in workflow_jobs(&workflow) {
            let runs_extraction = body.lines().any(|line| {
                let trimmed = line.trim();
                trimmed.contains("public-api-snapshot") || trimmed.contains("--test tree_contracts")
            });
            if !runs_extraction {
                continue;
            }
            reached += 1;

            let installs =
                toolchains_selected_at_public_api_installs(&body).unwrap_or_else(|error| {
                    panic!("Fix: {name} job `{job}` has an invalid toolchain setup: {error}")
                });
            assert_eq!(
                installs.len(),
                1,
                "Fix: {name} job `{job}` runs the public-API extraction, so it must install \
                 cargo-public-api exactly once on {pinned}"
            );
            let (selected, install) = &installs[0];
            assert_eq!(
                selected, pinned,
                "Fix: in {name} job `{job}`, select {pinned} through \
                 dtolnay/rust-toolchain before installing cargo-public-api"
            );
            assert!(
                install.contains(&format!("cargo +{pinned} install")),
                "Fix: in {name} job `{job}`, install cargo-public-api with \
                 `cargo +{pinned}`: {install}"
            );
            assert!(
                install.contains(version),
                "Fix: in {name} job `{job}`, install the pinned cargo-public-api version \
                 {version}: {install}"
            );
        }
    }

    assert!(
        reached >= 2,
        "Fix: the workflow scan found {reached} job(s) running the public-API extraction; \
         the snapshot gate and the tree-contract suite both run it, so a lower count means the \
         scan stopped matching and this contract guards nothing"
    );
}
