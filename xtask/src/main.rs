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
    if name == "lego-audit" {
        let mut runner_args = vec!["--subset".to_string(), "lego-audit".to_string()];
        runner_args.extend(args[2..].iter().cloned());
        sweep::run(&runner_args);
        return;
    }
    let Some(gate) = subcommands::find(name) else {
        eprintln!("Fix: unknown subcommand '{name}'. See --help.");
        process::exit(1);
    };
    let ctx = GateCtx::new(xtask::checkout::checkout_root(), args[2..].to_vec());
    // A delegated gate carries its options in the crate that implements them,
    // so the request travels to the child and comes back as report notes. Every
    // other gate is answered here, before it reads the tree.
    if gate::help_requested(&ctx.args) && gate.package() == "xtask" {
        print!("{}", gate::render(name, &gate::usage_report(&gate)));
        return;
    }
    let Some(descriptor) = xtask::gate_metadata::descriptor(name) else {
        eprintln!("Fix: gate `{name}` has no descriptor in GATE_METADATA");
        process::exit(1);
    };
    let declared_artifacts = descriptor.artifacts;
    let snapshot = xtask::artifact_gate::WorkspaceSnapshot::capture(&ctx.root);
    let result = gate.run(&ctx);
    let mutations =
        snapshot.detect_mutations(&ctx.root, name, declared_artifacts, gate.writes(&ctx));
    if !mutations.is_empty() {
        for mutation in mutations {
            eprintln!("Fix: {mutation}");
        }
        process::exit(1);
    }
    match result {
        Err(error) => {
            eprintln!("{error}");
            process::exit(1);
        }
        Ok(report) => {
            if !gate::help_requested(&ctx.args) {
                let contract_failures = report.contract_failures(descriptor);
                if !contract_failures.is_empty() {
                    for failure in contract_failures {
                        eprintln!("Fix: gate `{name}` {failure}");
                    }
                    process::exit(1);
                }
            }
            if ctx.has("--print-toolchain") && report.findings.is_empty() {
                return;
            }
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
