//! Registry-linked xtask subcommands.
//!
//! `xtask` builds and runs this binary for the subcommands it assigns here. It
//! is not meant to be invoked directly, but it accepts the same argument vector
//! so that `cargo run -p xtask-registry -- <subcommand>` works.

use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args
        .iter()
        .skip(1)
        .any(|arg| arg == "--help" || arg == "-h")
    {
        xtask::delegate::print_dispatch_help(
            "xtask-registry",
            "`xtask` assigns these subcommands here because each one reads the live operation registry.",
            xtask_registry::IMPLEMENTED.iter().map(|(name, _)| *name),
        );
        return;
    }
    if args.len() < 2 {
        eprintln!("Fix: missing subcommand. Run `cargo xtask --help`.");
        process::exit(1);
    }
    if !xtask_registry::dispatch(args[1].as_str(), &args) {
        eprintln!(
            "Fix: `{}` is not implemented in xtask-registry. Run `cargo xtask --help`.",
            args[1]
        );
        process::exit(1);
    }
}
