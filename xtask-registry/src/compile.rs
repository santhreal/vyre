//! `cargo xtask compile`  -  authenticated registered-target payload harness.
//!
//! The command decodes one frontend `Program`, adapts it to `ProgramGraph`,
//! compiles one neutral artifact, and invokes the pure compiler facet submitted
//! by each requested target owner. Target names, formats, and emission details
//! never live in this shared tool.
//!
//! Usage:
//!
//! ```sh
//! cargo xtask compile <program.vir> --to <registered-target-id> \
//!     [--to <registered-target-id>] [--output-dir <dir>]
//! ```
//!
//! Every target writes an authenticated `TargetPayload` frame named by the
//! neutral artifact digest and request position. The companion JSON manifest
//! records the full digest and requested target identities.

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process;

use vyre_foundation::ir::{Program, ProgramGraph};
use vyre_megakernel::{
    Artifact, CompileRequest, Digest, ExternalFacts, SearchBudget, TargetPayload,
};

const MAX_XTASK_COMPILE_INPUT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_XTASK_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;
const XTASK_SEARCH_BUDGET: SearchBudget = SearchBudget::new(256, 100_000, 1, 0, 1_000_000_000);

pub(crate) fn run(args: &[String]) {
    let (input_path, targets, output_dir) = parse_args(args).unwrap_or_else(|error| {
        eprintln!("{error}");
        process::exit(2);
    });
    let wire = read_bytes_bounded(&input_path).unwrap_or_else(|error| {
        eprintln!("Fix: cannot read {}: {error}", input_path.display());
        process::exit(1);
    });
    let program = Program::from_wire(&wire).unwrap_or_else(|error| {
        eprintln!("Fix: wire decode failed: {error}");
        process::exit(1);
    });
    let artifact = compile_neutral(program).unwrap_or_else(|error| {
        eprintln!("{error}");
        process::exit(1);
    });
    fs::create_dir_all(&output_dir).unwrap_or_else(|error| {
        eprintln!(
            "Fix: cannot create output directory {}: {error}",
            output_dir.display()
        );
        process::exit(1);
    });

    let digest = digest_hex(artifact.digest());
    let prefix = &digest[..16];
    for (index, target) in targets.iter().enumerate() {
        let payload = compile_registered_target(&artifact, target).unwrap_or_else(|error| {
            eprintln!("{error}");
            process::exit(1);
        });
        let bytes = payload.to_bytes().unwrap_or_else(|error| {
            eprintln!(
                "Fix: authenticated payload for registered target `{target}` could not encode: {error}"
            );
            process::exit(1);
        });
        let path = output_dir.join(format!("{prefix}.{index}.vtp"));
        fs::write(&path, bytes).unwrap_or_else(|error| {
            eprintln!("Fix: cannot write {}: {error}", path.display());
            process::exit(1);
        });
        println!("emitted: {}", path.display());
    }

    let manifest_path = output_dir.join(format!("{prefix}.manifest.json"));
    let manifest = serde_json::to_string_pretty(&serde_json::json!({
        "artifact": digest,
        "targets": targets,
    }))
    .expect("target manifest contains only strings")
        + "\n";
    fs::write(&manifest_path, manifest).unwrap_or_else(|error| {
        eprintln!("Fix: cannot write {}: {error}", manifest_path.display());
        process::exit(1);
    });
    println!("manifest: {}", manifest_path.display());
}

fn parse_args(args: &[String]) -> Result<(PathBuf, Vec<String>, PathBuf), String> {
    let input = args.get(2).ok_or_else(|| {
        "Fix: missing input wire file. Usage: cargo_full run --bin xtask -- compile <program.vir> --to <registered-target-id>".to_string()
    })?;
    let mut targets = Vec::new();
    let mut output_dir = PathBuf::from("target/vyre-compile");
    let mut index = 3;
    while index < args.len() {
        match args[index].as_str() {
            "--to" => {
                index += 1;
                let target = args
                    .get(index)
                    .filter(|target| !target.trim().is_empty())
                    .ok_or_else(|| {
                        "Fix: --to requires a non-empty registered target id".to_string()
                    })?;
                targets.push(target.clone());
                index += 1;
            }
            "--output-dir" => {
                index += 1;
                let path = args
                    .get(index)
                    .filter(|path| !path.trim().is_empty())
                    .ok_or_else(|| "Fix: --output-dir requires a path".to_string())?;
                output_dir = PathBuf::from(path);
                index += 1;
            }
            other => return Err(format!("Fix: unknown compile argument `{other}`")),
        }
    }
    if targets.is_empty() {
        return Err(
            "Fix: no --to target specified. Pass one target id from the linked target registry."
                .to_string(),
        );
    }
    Ok((PathBuf::from(input), targets, output_dir))
}

fn compile_neutral(program: Program) -> Result<Artifact, String> {
    let graph = ProgramGraph::from_program("xtask-compile", program)
        .map_err(|error| format!("Fix: Program cannot enter the canonical graph: {error}"))?;
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        XTASK_SEARCH_BUDGET,
        MAX_XTASK_ARTIFACT_BYTES,
    )
    .validate()
    .map_err(|error| format!("Fix: compile request is invalid: {error}"))?;
    vyre_megakernel::compile(&request)
        .map_err(|error| format!("Fix: neutral artifact compilation failed: {error}"))
}

fn compile_registered_target(
    artifact: &Artifact,
    target_id: &str,
) -> Result<TargetPayload, String> {
    let registration = vyre_registry_link::backend::live_backend_registry()
        .map_err(|error| format!("Fix: backend registry startup failed: {error}"))?
        .iter()
        .find(|registration| registration.target_id.as_str() == target_id)
        .ok_or_else(|| {
            format!(
                "Fix: target `{target_id}` is not linked. Link its concrete driver crate or select a linked target id."
            )
        })?;
    let compiler = registration
        .target_compiler()
        .map_err(|error| format!("Fix: target `{target_id}` has no compiler facet: {error}"))?;
    compiler
        .compile(artifact)
        .map_err(|error| format!("Fix: target `{target_id}` rejected the artifact: {error}"))
}

fn digest_hex(digest: Digest) -> String {
    let mut output = String::with_capacity(digest.as_bytes().len() * 2);
    for byte in digest.as_bytes() {
        use std::fmt::Write;
        write!(&mut output, "{byte:02x}").expect("formatting bytes into a String cannot fail");
    }
    output
}

fn read_bytes_bounded(path: &Path) -> io::Result<Vec<u8>> {
    let mut reader = fs::File::open(path)?.take(MAX_XTASK_COMPILE_INPUT_BYTES.saturating_add(1));
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_XTASK_COMPILE_INPUT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exceeds the {MAX_XTASK_COMPILE_INPUT_BYTES}-byte compile input cap",
                path.display()
            ),
        ));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::{compile_neutral, compile_registered_target};
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

    #[test]
    fn linked_target_compiler_emits_authenticated_payload() {
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
        );
        let artifact = compile_neutral(program).expect("neutral fixture artifact");
        let registration = vyre_registry_link::backend::live_backend_registry()
            .expect("valid backend registry")
            .iter()
            .find(|registration| registration.target_compiler.is_some())
            .expect("xtask must link at least one concrete target compiler");
        let payload = compile_registered_target(&artifact, registration.target_id.as_str())
            .expect("linked target compiler must emit an authenticated payload");

        assert_eq!(payload.neutral_artifact(), artifact.digest());
        assert!(!payload.bytes().is_empty());
        assert!(!payload.format().identity().is_empty());
    }
}
