//! Workspace package/path naming footgun detection.
//!
//! Path dependencies that escape the workspace root and directory-name vs
//! package-name mismatches confuse tooling and break hermetic builds.

use vyre_test_support::monorepo::{vyre_crate_directory, vyre_workspace_root};

#[test]
fn no_path_dependencies_outside_workspace() {
    let workspace_root = vyre_workspace_root();
    let root_toml = workspace_root.join("Cargo.toml");
    let content = std::fs::read_to_string(&root_toml).unwrap();

    let mut violations = Vec::new();
    for (line_no, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if (trimmed.contains("path = \"../") || trimmed.contains("path = \"../../"))
            && !is_adjacent_dataflow_dependency(trimmed)
        {
            violations.push(format!("line {}: {}", line_no + 1, trimmed));
        }
    }

    assert!(
        violations.is_empty(),
        "workspace Cargo.toml must not reference paths outside the workspace (baseline known exceptions). Violations:\n{}",
        violations.join("\n")
    );
}

fn is_adjacent_dataflow_dependency(line: &str) -> bool {
    line.contains("path = \"../../../dataflow/")
}

#[test]
fn meta_crate_directory_naming_is_stable() {
    let workspace_root = vyre_workspace_root();

    // The meta-crate named "vyre" sits in a directory named `vyre`, the
    // basename every member directory shares with its package.
    let directory = vyre_crate_directory("vyre");
    assert_eq!(
        directory.file_name().and_then(|name| name.to_str()),
        Some("vyre"),
        "the meta-crate named 'vyre' must sit in a directory named 'vyre', found {}",
        directory.display()
    );

    // Workspace dependency routing must keep the meta-crate name bound to
    // the directory the roster places it at.
    let relative = directory
        .strip_prefix(&workspace_root)
        .expect("the roster directory resolves under the workspace root")
        .display()
        .to_string();
    let root_toml = workspace_root.join("Cargo.toml");
    let root_content = std::fs::read_to_string(&root_toml).unwrap();
    let route = format!("path = \"{relative}\"");
    assert!(
        root_content
            .lines()
            .any(|line| line.trim().starts_with("vyre = {") && line.contains(&route)),
        "workspace dependency routing must route 'vyre' to its member directory `{relative}`"
    );
}
