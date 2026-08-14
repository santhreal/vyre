//! Scaffold one launch-rule contract and its truth-data directories.
//!
//! Run via `cargo run -p xtask --bin scaffold_rule -- <slug>`.
//! The command writes a contract plus positive, negative, evasion and
//! cross-file case directories and the CVE replay, property, differential and
//! end-to-end truth manifests, all under `rules/launch/<slug>/` in this
//! repository. `rule_tree` owns the layout and refuses any path outside it.

use std::fs;
use std::path::Path;

#[path = "rule_tree/mod.rs"]
mod rule_tree;

fn fatal(message: &str) -> ! {
    eprintln!("Fix: {message}");
    std::process::exit(1);
}

fn create_dir(path: &Path) {
    rule_tree::require_inside_repository(path);
    if let Err(error) = fs::create_dir_all(path) {
        eprintln!("Fix: failed to create `{}`: {error}", path.display());
        std::process::exit(1);
    }
}

fn write_file(path: &Path, contents: &str) {
    rule_tree::require_inside_repository(path);
    if let Err(error) = fs::write(path, contents) {
        eprintln!("Fix: failed to write `{}`: {error}", path.display());
        std::process::exit(1);
    }
}

fn print_help() {
    println!("Scaffold one launch-rule contract and truth-test suite.");
    println!();
    println!("Usage: scaffold_rule <slug>");
    println!();
    println!("Arguments:");
    println!("  <slug>  launch-rule directory name");
    println!();
    println!("Exit codes:");
    println!("  0  scaffold created");
    println!("  1  input or filesystem failure");
    println!("  2  command-line arguments are invalid");
}

fn is_valid_slug(slug: &str) -> bool {
    let bytes = slug.as_bytes();
    !bytes.is_empty()
        && bytes.first().is_some_and(u8::is_ascii_alphanumeric)
        && bytes.last().is_some_and(u8::is_ascii_alphanumeric)
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'-')
}

fn main() {
    let mut args = std::env::args().skip(1);
    let slug = match args.next() {
        Some(arg) if matches!(arg.as_str(), "-h" | "--help") => {
            if let Some(extra) = args.next() {
                eprintln!(
                    "Fix: unexpected argument `{extra}` after `--help`. Use `scaffold_rule --help`."
                );
                std::process::exit(2);
            }
            print_help();
            return;
        }
        Some(slug) if is_valid_slug(&slug) => slug,
        Some(slug) => {
            eprintln!(
                "Fix: invalid rule slug `{slug}`. Use lowercase letters, digits, and interior hyphens."
            );
            std::process::exit(2);
        }
        None => {
            eprintln!("Fix: expected rule slug. Use `scaffold_rule --help`.");
            std::process::exit(2);
        }
    };
    if let Some(extra) = args.next() {
        eprintln!("Fix: unexpected argument `{extra}`. Use `scaffold_rule --help`.");
        std::process::exit(2);
    }

    let launch_dir = rule_tree::launch_dir().join(&slug);
    create_dir(&launch_dir);

    write_file(&launch_dir.join("CONTRACT.md"), "# Rule Contract\n");

    let truth_dir = rule_tree::truth_dir(&slug);
    create_dir(&truth_dir);

    for case_class in rule_tree::TRUTH_DIRS {
        create_dir(&truth_dir.join(case_class));
    }

    for manifest in rule_tree::TRUTH_FILES {
        write_file(&truth_dir.join(manifest), "");
    }

    println!("Scaffolded rule {}", slug);
}
