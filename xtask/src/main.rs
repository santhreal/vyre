//! xtask dispatcher for the vyre workspace.

use std::env;
use std::process;

mod artifact_paths;
mod bench;
mod binary;
mod compile;
mod docs;
mod gates;
mod hash;
mod json_output;
mod manifest_walk;
mod output_arg;
mod print_composition;
mod release;
mod shrink;
mod subcommands;
mod text_markers;
mod toml_config;
mod trace_f32;

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
        Some(entry) => (entry.run)(&args),
        None => {
            eprintln!("Fix: unknown subcommand '{name}'. See --help.");
            process::exit(1);
        }
    }
}
