//! Command-line entry point for the workspace structural gate.

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    structure_gate::run(&args);
}
