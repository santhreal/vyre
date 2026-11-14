//! Contract and mutation-checked negative fixture tests for unified `xtask`.

use std::fs;
use tempfile::TempDir;

#[test]
fn gate_registry_is_sorted_and_non_empty() {
    let names: Vec<&str> = xtask::gates::GATES.iter().map(|g| g.name).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(names, sorted);
    assert_eq!(names.len(), 9);
}

#[test]
fn source_scan_skips_comments_and_raw_strings() {
    let sample =
        "let x = \"todo!()\"; // unimplemented!()\n/* block todo!() */ let y = r#\"todo!()\"#;";
    for offset in xtask::source_scan::code_offsets(sample) {
        let rest = &sample[offset..];
        assert!(
            !(rest.starts_with("todo!(") || rest.starts_with("unimplemented!(")),
            "comment or string interior leaked into code_offsets at {offset}"
        );
    }
}

#[test]
fn cfg_test_extraction_partitions_test_and_production_items() {
    let src =
        "pub fn prod() -> u32 { 1 }\n#[cfg(test)]\nmod tests { #[test] fn t() { todo!(); } }\n";
    let stripped = xtask::cfg_test::strip_cfg_test_items(src);
    assert!(stripped.contains("pub fn prod"));
    assert!(!stripped.contains("todo!"));
    let decls = xtask::cfg_test::cfg_test_module_declarations(src);
    assert!(decls.is_empty());
    let items = xtask::cfg_test::cfg_test_items(src);
    assert!(items.contains("mod tests"));
}

#[test]
fn ci_steps_rejects_target_dir_and_unguarded_self_hosted_runner() {
    let tmp = TempDir::new().unwrap();
    let wf_dir = tmp.path().join(".github/workflows");
    fs::create_dir_all(&wf_dir).unwrap();
    fs::write(
        wf_dir.join("bad.yml"),
        "name: Bad\njobs:\n  gpu:\n    runs-on: [self-hosted, gpu]\n    steps:\n      - run: CARGO_TARGET_DIR=/foo cargo test\n",
    )
    .unwrap();
    let errs = xtask::gates::ci_steps::check(tmp.path()).unwrap_err();
    assert!(errs.iter().any(|e| e.contains("concurrency")));
    assert!(errs.iter().any(|e| e.contains("target directory")));
    assert!(errs.iter().any(|e| e.contains("fork repository check")));
}

#[test]
fn docs_register_rejects_competitor_names_and_excess_prose_docs() {
    let tmp = TempDir::new().unwrap();
    for i in 0..6 {
        fs::write(
            tmp.path().join(format!("DOC{i}.md")),
            "Moondream \u{2014} /tmp/foo\n",
        )
        .unwrap();
    }
    let errs = xtask::gates::docs_register::check(tmp.path()).unwrap_err();
    assert!(errs.iter().any(|e| e.contains("exceed ceiling")));
    assert!(errs.iter().any(|e| e.contains("Moondream")));
    assert!(errs.iter().any(|e| e.contains("/tmp/")));
}

/// WHY: every crate carries exactly one README stating its purpose and
/// boundaries, and those READMEs do not count against the repository prose
/// ceiling. Catches a crate added without a README and a crate README counted
/// as repository prose. Does not judge README content beyond the banned tokens.
#[test]
fn docs_register_requires_one_readme_per_crate_outside_the_prose_ceiling() {
    let tmp = TempDir::new().unwrap();
    for name in ["a", "b", "c", "d", "e", "f"] {
        let dir = tmp.path().join("crates").join(name);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("Cargo.toml"), "[package]\n").unwrap();
        if name != "f" {
            fs::write(dir.join("README.md"), "# crate\n").unwrap();
        }
    }
    fs::write(tmp.path().join("README.md"), "# root\n").unwrap();
    let errs = xtask::gates::docs_register::check(tmp.path()).unwrap_err();
    assert_eq!(
        errs,
        vec![
            "crates/f: crate has no README.md; add one stating its purpose and boundaries"
                .to_owned()
        ]
    );

    fs::write(tmp.path().join("crates/f/README.md"), "# crate\n").unwrap();
    assert_eq!(xtask::gates::docs_register::check(tmp.path()), Ok(()));
}

#[test]
fn workspace_membership_and_layering_reject_undeclared_or_reversed_crates() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("crates/low/src")).unwrap();
    fs::create_dir_all(tmp.path().join("crates/high/src")).unwrap();
    fs::create_dir_all(tmp.path().join("crates/orphan/src")).unwrap();
    fs::create_dir_all(tmp.path().join("docs")).unwrap();

    fs::write(
        tmp.path().join("Cargo.toml"),
        "[workspace]\nresolver = \"2\"\nmembers = [\"crates/low\", \"crates/high\"]\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("crates/low/Cargo.toml"),
        "[package]\nname = \"low\"\nversion = \"0.1.0\"\nedition = \"2021\"\n[lints]\nworkspace = true\n[dependencies]\nhigh = { path = \"../high\" }\n",
    )
    .unwrap();
    fs::write(tmp.path().join("crates/low/src/lib.rs"), "pub fn a() {}\n").unwrap();
    fs::write(
        tmp.path().join("crates/high/Cargo.toml"),
        "[package]\nname = \"high\"\nversion = \"0.1.0\"\nedition = \"2021\"\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    fs::write(tmp.path().join("crates/high/src/lib.rs"), "pub fn b() {}\n").unwrap();
    fs::write(
        tmp.path().join("crates/orphan/Cargo.toml"),
        "[package]\nname = \"orphan\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("docs/CRATE_OWNERSHIP.toml"),
        "schema_version = 4\n[[layer]]\nname = \"ir\"\nrank = 0\n[[layer]]\nname = \"top\"\nrank = 1\n[[crate]]\npackage = \"low\"\nlayer = \"ir\"\n[[crate]]\npackage = \"high\"\nlayer = \"top\"\n",
    )
    .unwrap();

    let mem_errs = xtask::gates::workspace_membership::check(tmp.path()).unwrap_err();
    assert!(mem_errs.iter().any(|e| e.contains("orphan")));

    let layer_errs = xtask::gates::layering::check(tmp.path()).unwrap_err();
    assert!(layer_errs
        .iter()
        .any(|e| e.contains("upward dependency `low` (rank 0) -> `high` (rank 1)")));
}

#[path = "path_modules_stay_inside_their_package.rs"]
mod path_modules_stay_inside_their_package;
