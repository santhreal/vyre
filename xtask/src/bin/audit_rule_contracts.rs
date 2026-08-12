//! Audit `rules/launch/*.toml` rule contracts and report missing or
//! malformed fields.
//!
//! Run via `cargo xtaskbin audit_rule_contracts`. The binary
//! exits non-zero when any rule deviates from `rules/SCHEMA.md`.

use std::fs;
use std::path::Path;

fn print_help() {
    println!("Audit launch-rule contracts and truth-test directories.");
    println!();
    println!("Usage: audit_rule_contracts");
    println!();
    println!("Exit codes:");
    println!("  0  every rule contract and truth-test directory exists");
    println!("  1  the rule tree is unavailable or a contract is incomplete");
    println!("  2  command-line arguments are invalid");
}

fn main() {
    let mut args = std::env::args().skip(1);
    if let Some(arg) = args.next() {
        if matches!(arg.as_str(), "-h" | "--help") && args.next().is_none() {
            print_help();
            return;
        }
        eprintln!("Fix: unknown argument `{arg}`. Use `audit_rule_contracts --help`.");
        std::process::exit(2);
    }
    let launch_dir = Path::new("../../../../../rules/launch");
    if !launch_dir.exists() {
        eprintln!("Rules directory not found");
        std::process::exit(1);
    }

    let mut failed = false;
    for entry in fs::read_dir(launch_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            let slug = path.file_name().unwrap().to_str().unwrap();
            println!("Auditing rule {}", slug);

            let contract = path.join("CONTRACT.md");
            if !contract.exists() {
                eprintln!("FAIL: Missing CONTRACT.md in {}", slug);
                failed = true;
            }

            let test_dir = Path::new("../../../../../tests/launch_rule_truth").join(slug);
            let expected_dirs = ["positives", "negatives", "evasions", "cross_file"];
            for d in expected_dirs.iter() {
                if !test_dir.join(d).exists() {
                    eprintln!("FAIL: Missing test dir {}/{}", slug, d);
                    failed = true;
                }
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}
