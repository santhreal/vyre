//! Command-line entry point for the workspace structural gate.

#![forbid(unsafe_code)]

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    structure_gate::run(&args);
}
