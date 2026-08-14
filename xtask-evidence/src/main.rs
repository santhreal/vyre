//! Evidence-provenance xtask subcommands.
//!
//! `xtask` builds and runs this binary for the subcommands it assigns here. It
//! is not meant to be invoked directly, but it accepts the same argument vector
//! so that `cargo run -p xtask-evidence -- <subcommand>` works.

use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
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
