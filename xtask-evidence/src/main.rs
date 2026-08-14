//! Evidence-provenance xtask subcommands.
//!
//! `xtask` builds and runs this binary for the subcommands it assigns here. It
//! is not meant to be invoked directly, but it accepts the same argument vector
//! so that `cargo run -p xtask-evidence -- <subcommand>` works.

use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.iter().skip(1).any(|arg| arg == "--help" || arg == "-h") {
        help();
        return;
    }
    if args.len() < 2 {
        eprintln!("Fix: missing subcommand. Run `cargo xtask --help`.");
        process::exit(1);
    }
    if !xtask_evidence::dispatch(args[1].as_str(), &args) {
        eprintln!(
            "Fix: `{}` is not implemented in xtask-evidence. Run `cargo xtask --help`.",
            args[1]
        );
        process::exit(1);
    }
}

/// Print the roster this binary can dispatch.
///
/// The list is read from `IMPLEMENTED` rather than written out here, so a
/// subcommand added to that table is documented by adding it and cannot drift.
fn help() {
    println!("USAGE");
    println!("  cargo run -p xtask-evidence -- <subcommand> [options]");
    println!();
    println!("`xtask` assigns these subcommands to this crate because each one reads recorded");
    println!("benchmark or release evidence. Run `cargo xtask --help` for every workspace");
    println!("command, and `cargo xtask <subcommand> --help` for one command's options.");
    println!();
    println!("SUBCOMMANDS:");
    for (name, _) in xtask_evidence::IMPLEMENTED {
        println!("  {name}");
    }
}
