//! Scaffold one launch-rule contract and its truth-test directories.
//!
//! Run via `cargo xtaskbin scaffold_rule -- <slug>`.
//! The command writes a contract and positive, negative, evasion, cross-file,
//! CVE replay, property, differential, and end-to-end test placeholders.

use std::fs;
use std::path::Path;

fn fatal(message: &str) -> ! {
    eprintln!("Fix: {message}");
    std::process::exit(1);
}

fn create_dir(path: &Path) {
    if let Err(error) = fs::create_dir_all(path) {
        eprintln!("Fix: failed to create `{}`: {error}", path.display());
        std::process::exit(1);
    }
}

fn write_file(path: &Path, contents: &str) {
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

    let launch_dir = Path::new("../../../../../rules/launch").join(&slug);
    create_dir(&launch_dir);

    write_file(&launch_dir.join("CONTRACT.md"), "# Rule Contract\n");

    let test_dir = Path::new("../../../../../tests/launch_rule_truth").join(&slug);
    create_dir(&test_dir);

    for d in &["positives", "negatives", "evasions", "cross_file"] {
        create_dir(&test_dir.join(d));
    }

    write_file(&test_dir.join("cve_replay.toml"), "");
    write_file(&test_dir.join("property.rs"), "");
    write_file(&test_dir.join("differential.toml"), "");
    write_file(&test_dir.join("e2e_cli.rs"), "");

    println!("Scaffolded rule {}", slug);
}
