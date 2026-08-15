//! xtask dispatcher for the vyre workspace.
//!
//! Every subcommand is a gate, so dispatch is one lookup in the registry. The
//! only name that is not a gate is `gates`, which is the runner.

use std::env;
use std::process;

use xtask::gate::{self, GateCtx};
use xtask::gates::sweep;
use xtask::subcommands;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Fix: missing subcommand. See --help.");
        process::exit(1);
    }

    let name = args[1].as_str();
    if name == "--help" || name == "-h" {
        print!("{}", subcommands::help_text());
        process::exit(0);
    }
    if name == sweep::RUNNER {
        sweep::run(&args[2..]);
        return;
    }
    let Some(gate) = subcommands::find(name) else {
        eprintln!("Fix: unknown subcommand '{name}'. See --help.");
        process::exit(1);
    };
    let ctx = GateCtx::new(xtask::checkout::checkout_root(), args[2..].to_vec());
    match gate.run(&ctx) {
        Err(error) => {
            eprintln!("{error}");
            process::exit(1);
        }
        Ok(report) => {
            print!("{}", gate::render(name, &report));
            // A finding is a failure. There is no informational mode: a gate
            // that reported a problem and exited 0 is how 32 gates judged
            // nothing while reading as coverage.
            if !report.findings.is_empty() {
                process::exit(1);
            }
        }
    }
}
