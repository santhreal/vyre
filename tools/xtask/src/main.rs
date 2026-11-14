//! Command-line runner for vyre workspace structural verification and maintenance tasks.

use std::process::ExitCode;

fn print_usage() {
    eprintln!("usage: xtask <command>");
    eprintln!("commands:");
    eprintln!("  check | gates            Run all registered structural and contract gates");
    eprintln!("  list-gates               List all registered gates");
    eprintln!("  prune-coverage-objects   Delete unprofiled llvm-cov executables before reporting");
    for gate in xtask::gates::GATES {
        eprintln!("  {:24} {}", gate.name, gate.summary);
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first().map(String::as_str) else {
        print_usage();
        return ExitCode::from(2);
    };

    let root = xtask::workspace_root();
    match cmd {
        "-h" | "--help" | "help" => {
            print_usage();
            ExitCode::SUCCESS
        }
        "list-gates" => {
            for gate in xtask::gates::GATES {
                println!("{}	{}", gate.name, gate.summary);
            }
            ExitCode::SUCCESS
        }
        "prune-coverage-objects" => match xtask::coverage_objects_prune(&root) {
            Ok(_) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        },
        "check" | "gates" => match xtask::gates::run_all(&root) {
            Ok(()) => {
                let dup = xtask::gates::dup_scan::scan(&root);
                println!(
                    "ok: {} gates passed (dup-scan {:.2}%, {}/{} lines)",
                    xtask::gates::GATES.len(),
                    dup.ratio_pct,
                    dup.duplicate_lines,
                    dup.total_lines
                );
                ExitCode::SUCCESS
            }
            Err(failures) => {
                for (gate, errs) in &failures {
                    for err in errs {
                        eprintln!("[{gate}] {err}");
                    }
                }
                eprintln!("failed: {} gate(s) reported errors", failures.len());
                ExitCode::FAILURE
            }
        },
        gate_name => {
            if gate_name == "dup-scan" {
                let dup = xtask::gates::dup_scan::scan(&root);
                println!(
                    "dup-scan: {:.2}% ({}/{} normalized production lines)",
                    dup.ratio_pct, dup.duplicate_lines, dup.total_lines
                );
            }
            match xtask::gates::run_named(&root, gate_name) {
                Ok(()) => ExitCode::SUCCESS,
                Err(errs) => {
                    for err in errs {
                        eprintln!("[{gate_name}] {err}");
                    }
                    ExitCode::FAILURE
                }
            }
        }
    }
}
