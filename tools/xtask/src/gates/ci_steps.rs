//! Verify `.github/workflows/*.yml` invariants.

use std::fs;
use std::path::Path;

/// Check workflow invariants under `.github/workflows/`.
pub fn check(root: &Path) -> Result<(), Vec<String>> {
    let wf_dir = root.join(".github/workflows");
    let Ok(entries) = fs::read_dir(&wf_dir) else {
        return Err(vec![format!(
            "missing workflow directory {}",
            wf_dir.display()
        )]);
    };
    let mut errors = Vec::new();
    let mut workflow_count = 0usize;
    let mut saw_coverage_floor = false;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("yml") {
            continue;
        }
        workflow_count += 1;
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let Ok(text) = fs::read_to_string(&path) else {
            errors.push(format!("cannot read {}", path.display()));
            continue;
        };
        if !text.contains("concurrency:") {
            errors.push(format!(
                "{name}: missing top-level `concurrency:` configuration"
            ));
        }
        for (line_no, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with('#') {
                continue;
            }
            if trimmed.contains("CARGO_TARGET_DIR")
                || trimmed.contains("CARGO_BUILD_TARGET_DIR")
                || trimmed.contains("--target-dir")
            {
                errors.push(format!(
                    "{name}:{}: sets Cargo target directory override (`{trimmed}`)",
                    line_no + 1
                ));
            }
        }
        if text.contains("self-hosted")
            && !text.contains("github.event.pull_request.head.repo.full_name == github.repository")
        {
            errors.push(format!(
                "{name}: self-hosted job lacks fork repository check on pull_request"
            ));
        }
        if name == "coverage.yml" && text.contains("COVERAGE_FLOOR_LINES") && text.contains("85") {
            saw_coverage_floor = true;
        }
    }

    if workflow_count == 0 {
        errors.push("no `.yml` workflows found in .github/workflows".to_owned());
    }
    if !saw_coverage_floor {
        errors.push("coverage.yml must enforce COVERAGE_FLOOR_LINES = 85".to_owned());
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
