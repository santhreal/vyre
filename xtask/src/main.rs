//! xtask dispatcher for the vyre workspace.

use std::env;
use std::process;

use xtask::subcommands::{self, Home};

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
    match subcommands::find(name) {
        Some(entry) => match entry.home {
            Home::Local(run) => run(&args),
            Home::Registry => xtask::delegate::run("xtask-registry", &args),
            Home::Evidence => xtask::delegate::run("xtask-evidence", &args),
        },
        None => {
            eprintln!("Fix: unknown subcommand '{name}'. See --help.");
            process::exit(1);
        }
    }
}
