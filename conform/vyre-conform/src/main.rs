//! `vyre-conform` CLI  -  runs conformance certs for registered ops.

mod artifact_json;
mod backend_selection;
mod certificate_merge;
mod dispatch_command;
mod operation_selection;
mod proof_options;
mod proof_plan;
mod proof_scheduler;
mod proof_timing;
mod prove_command;
mod reference_parity;
mod replay_capsule;
mod witness_fixtures;

use crate::certificate_merge::merge_certificates;
use crate::dispatch_command::dispatch_pairs;
use crate::proof_plan::emit_plan;
use crate::prove_command::{prove, DEFAULT_CERTIFICATE_DIR, DEFAULT_CERTIFICATE_FILE};

fn main() {
    let mut args = std::env::args();
    let _binary = args.next();
    let subcommand = match args.next() {
        Some(arg) => arg,
        None => {
            print_usage();
            return;
        }
    };
    if subcommand == "-h" || subcommand == "--help" {
        print_usage();
        return;
    }
    if subcommand == "prove" {
        if let Err(error) = prove(args) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }
    if subcommand == "plan" {
        if let Err(error) = emit_plan(args) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }
    if subcommand == "merge" {
        if let Err(error) = merge_certificates(args) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }
    if subcommand != "dispatch" {
        eprintln!(
            "unknown subcommand `{}`  -  supported subcommands: dispatch, merge, plan, prove.",
            subcommand
        );
        std::process::exit(2);
    }

    let mut backend_value = None::<String>;
    let mut ops_value = None::<String>;
    let mut it = args;
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--backend" => {
                backend_value = it.next();
            }
            "--ops" => {
                ops_value = it.next();
            }
            other => {
                eprintln!("unknown flag `{other}`");
                std::process::exit(2);
            }
        }
    }

    let backend = backend_value.as_deref().unwrap_or("auto");
    let ops = ops_value.as_deref().unwrap_or("all");
    match dispatch_pairs(backend, ops) {
        Ok(pairs) => {
            let failed = pairs.iter().any(|pair| !pair.passed);
            for pair in pairs {
                let json = match serde_json::to_string(&pair) {
                    Ok(json) => json,
                    Err(error) => {
                        eprintln!("failed to serialize dispatch result: {error}");
                        std::process::exit(1);
                    }
                };
                println!("{json}");
            }
            if failed {
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    println!("usage: vyre-conform dispatch --backend <backend-id|auto> --ops <all|<op_id>>");
    println!("       vyre-conform plan [--out <plan.json>] [--backend <all|backend-id>] [--ops <all|op_id>] [--shard <index>/<count>]");
    println!("       vyre-conform merge --out <merged.json> <prove-shard.json>...");
    println!(
        "       vyre-conform prove [--out <cert.json>] [--certificates <dir>] [--backend <all|backend-id>] [--ops <all|op_id>] [--shard <index>/<count>]  # default: {DEFAULT_CERTIFICATE_DIR}/{DEFAULT_CERTIFICATE_FILE}"
    );
}
