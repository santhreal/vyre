//! Audit every `rules/launch/<slug>` rule contract and report the missing
//! contract or truth-data directories.
//!
//! Run via `./cargo_full run -p xtask --bin audit_rule_contracts`. The binary exits
//! non-zero when any rule deviates from `rules/SCHEMA.md`. `rule_tree` owns the
//! layout it audits, so the auditor and `scaffold_rule` cannot disagree.

use std::fs;

use xtask::rule_tree;

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
    let launch_dir = rule_tree::launch_dir();
    if !launch_dir.exists() {
        eprintln!(
            "Fix: no rule tree at `{}`. Scaffold one with `cargo run -p xtask --bin \
             scaffold_rule -- <slug>`.",
            launch_dir.display()
        );
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

            let truth_dir = rule_tree::truth_dir(slug);
            for case_class in rule_tree::TRUTH_DIRS {
                if !truth_dir.join(case_class).exists() {
                    eprintln!("FAIL: Missing truth directory {slug}/truth/{case_class}");
                    failed = true;
                }
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}
