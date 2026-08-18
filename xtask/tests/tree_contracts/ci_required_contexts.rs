//! A required status context is the string branch protection receives.
//!
//! WHY: branch protection matches a required check by the context GitHub
//! reports, which is the job's `name:` when the job declares one and the job id
//! otherwise. A required list that names a job by id where the job declares a
//! display name waits forever on a context that never arrives, and the pull
//! request stays pending with every gate green. The gate used to accept either
//! string, so the one thing branch protection consumes was the one thing it did
//! not verify.
//!
//! The fixtures drive `ci-required` over a two-job workflow: one job with a
//! display name, one without. Both readings are asserted, so the gate cannot go
//! back to accepting either.

use std::fs;
use std::path::Path;
use std::process::Command;

use tempfile::TempDir;

use xtask::gate::{GateBehavior, GateCtx, Report};
use xtask::gates::ci_contract::CiRequired;

/// A checkout with one workflow and the required-context document.
fn fixture(required: &str) -> TempDir {
    let root = tempfile::tempdir().expect("Fix: temporary fixture directory must be writable");
    fs::create_dir_all(root.path().join(".github/workflows")).unwrap();
    fs::write(
        root.path().join("Cargo.toml"),
        "[workspace]\nresolver = \"2\"\nmembers = []\n",
    )
    .unwrap();
    fs::write(
        root.path().join(".github/workflows/gates.yml"),
        concat!(
            "name: gates\n",
            "on:\n",
            "  push:\n",
            "    branches: [main]\n",
            "  pull_request:\n",
            "jobs:\n",
            "  named-job:\n",
            "    name: Every registered gate\n",
            "    runs-on: ubuntu-latest\n",
            "    steps:\n",
            "      - run: cargo run -q -p xtask --bin xtask -- gates\n",
            "  bare-job:\n",
            "    runs-on: ubuntu-latest\n",
            "    steps:\n",
            "      - run: cargo run -q -p xtask --bin xtask -- op-matrix\n",
        ),
    )
    .unwrap();
    fs::write(root.path().join(".github/CI_REQUIRED.md"), required).unwrap();
    let status = Command::new("git")
        .args(["init", "-q"])
        .current_dir(root.path())
        .status()
        .expect("Fix: git must launch for the fixture tree listing");
    assert!(status.success());
    root
}

fn judge(root: &Path) -> Report {
    CiRequired
        .run(&GateCtx::new(root.to_path_buf(), Vec::new()))
        .expect("Fix: the gate must be able to read the fixture tree")
}

/// Every finding rendered as one line, for substring assertions.
fn rendered(report: &Report) -> String {
    report.finding_messages()
}

/// A required list that names the display name of every job is accepted.
#[test]
fn a_context_named_as_the_job_reports_it_is_accepted() {
    let root = fixture(concat!(
        "# Required checks\n\n",
        "## From `gates.yml`\n\n",
        "- `Every registered gate`\n",
        "- `bare-job`\n",
    ));
    let report = judge(root.path());
    assert_eq!(report.count(), 0, "{}", rendered(&report));
}

/// A job id used as a required context where the job declares a display name is
/// a finding that names the string the job actually reports.
#[test]
fn a_job_id_for_a_job_with_a_display_name_is_a_finding() {
    let root = fixture(concat!(
        "# Required checks\n\n",
        "## From `gates.yml`\n\n",
        "- `named-job`\n",
        "- `bare-job`\n",
    ));
    let report = judge(root.path());
    let rendered = rendered(&report);
    assert_eq!(report.count(), 1, "{rendered}");
    assert!(
        rendered.contains("`named-job` is a job id in `gates.yml` and that job reports as `Every registered gate`"),
        "the finding must name the reported context: {rendered}"
    );
}

/// A context matching no job at all is still a finding.
#[test]
fn a_context_no_job_reports_is_a_finding() {
    let root = fixture(concat!(
        "# Required checks\n\n",
        "## From `gates.yml`\n\n",
        "- `Every registered gate`\n",
        "- `bare-job`\n",
        "- `no-such-job`\n",
    ));
    let report = judge(root.path());
    let rendered = rendered(&report);
    assert_eq!(report.count(), 1, "{rendered}");
    assert!(
        rendered.contains("no job in `gates.yml` reports as `no-such-job`"),
        "the finding must name the missing context: {rendered}"
    );
}
