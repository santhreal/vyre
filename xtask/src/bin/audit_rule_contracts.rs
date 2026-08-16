//! Audit every `rules/launch/<slug>` rule contract and report the missing
//! contract or truth-data directories.
//!
//! Run via `./cargo_full run -p xtask --bin audit_rule_contracts`. The binary exits
//! non-zero when any rule deviates from `rules/SCHEMA.md`. `rule_tree` owns the
//! layout it audits, so the auditor and `scaffold_rule` cannot disagree.

use std::fs;

use xtask::operator_binary::{help_requested, Usage};
use xtask::rule_tree;

const USAGE: Usage = Usage {
    name: "audit_rule_contracts",
    summary: "Audit launch-rule contracts and truth-test directories.",
    exit_codes: &[
        (0, "every rule contract and truth-test directory exists"),
        (1, "the rule tree is unavailable or a contract is incomplete"),
        (2, "command-line arguments are invalid"),
    ],
};

fn main() {
    if help_requested(&USAGE) {
        return;
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
