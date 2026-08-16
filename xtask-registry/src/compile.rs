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

use vyre_foundation::ir::{Program, ProgramGraph};
use vyre_megakernel::{
    Artifact, CompileRequest, DeviceFacts, Digest, ExternalFacts, SearchBudget, TargetPayload,
};
use xtask::gate::{Finding, Gate, GateCtx, GateError, Report};

const MAX_XTASK_COMPILE_INPUT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_XTASK_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;
const XTASK_SEARCH_BUDGET: SearchBudget = SearchBudget::new(256, 100_000, 1, 0, 1_000_000_000);

/// Compiles the registered release corpus, and any caller-named wire file.
pub struct Compile;

impl Gate for Compile {
    fn name(&self) -> &'static str {
        "compile"
    }

    fn help(&self) -> &'static str {
        "Compile the registered release corpus; --program ID narrows to one case, --input PATH compiles one wire file, --to ID also compiles that registered target, --out DIR writes the payloads"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let cases = corpus(ctx)?;
        let targets: Vec<&str> = ctx
            .args
            .iter()
            .zip(ctx.args.iter().skip(1))
            .filter(|(flag, _)| flag.as_str() == "--to")
            .map(|(_, target)| target.as_str())
            .collect();
        let out_dir = ctx.flag("--out").map(PathBuf::from);
        let mut report = Report::clean();
        report.note(format!(
            "{} program(s) compiled, {} registered target(s) requested",
            cases.len(),
            targets.len()
        ));
        for (id, program) in cases {
            let artifact = match compile_neutral(program) {
                Ok(artifact) => artifact,
                Err(error) => {
                    report.find(Finding::new(
                        format!("`{id}` does not compile to a neutral artifact: {error}"),
                        "repair the program or the neutral compiler path it exercises",
                    ));
                    continue;
                }
            };
            let digest = digest_hex(artifact.digest());
            for (index, target) in targets.iter().enumerate() {
                let payload = match compile_registered_target(&artifact, target) {
                    Ok(payload) => payload,
                    Err(error) => {
                        report.find(Finding::new(
                            format!(
                                "`{id}` does not compile for registered target `{target}`: {error}"
                            ),
                            "repair the target lowering, or stop naming that target",
                        ));
                        continue;
                    }
                };
                let bytes = match payload.to_bytes() {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        report.find(Finding::new(
                            format!(
                                "the authenticated payload of `{id}` for `{target}` does not encode: {error}"
                            ),
                            "repair the payload encoder for that target",
                        ));
                        continue;
                    }
                };
                if let Some(dir) = out_dir.as_ref() {
                    fs::create_dir_all(dir).map_err(|error| {
                        GateError::new(
                            format!("cannot create {}: {error}", dir.display()),
                            "pass a writable path after --out",
                        )
                    })?;
                    let path = dir.join(format!("{}.{index}.vtp", &digest[..16]));
                    fs::write(&path, bytes).map_err(|error| {
                        GateError::new(
                            format!("cannot write {}: {error}", path.display()),
                            "pass a writable path after --out",
                        )
                    })?;
                    report.note(format!("emitted {}", path.display()));
                }
            }
        }
        Ok(report)
    }
}

/// The programs this run compiles: one caller-named wire file, one narrowed
/// corpus case, or the whole registered release corpus.
fn corpus(ctx: &GateCtx) -> Result<Vec<(String, Program)>, GateError> {
    if let Some(input) = ctx.flag("--input") {
        let path = PathBuf::from(input);
        let wire = read_bytes_bounded(&path).map_err(|error| {
            GateError::new(
                format!("cannot read {}: {error}", path.display()),
                "pass a readable wire file after --input",
            )
        })?;
        let program = Program::from_wire(&wire).map_err(|error| {
            GateError::new(
                format!("wire decode of {} failed: {error}", path.display()),
                "pass a file this compiler version encoded",
            )
        })?;
        return Ok(vec![(path.display().to_string(), program)]);
    }
    crate::corpus::selected_cases(ctx.flag("--program"), "compile")
}

fn compile_neutral(program: Program) -> Result<Artifact, String> {
    let graph = ProgramGraph::from_program("xtask-compile", program)
        .map_err(|error| format!("Fix: Program cannot enter the canonical graph: {error}"))?;
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
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
    const NIBBLES: &[u8; 16] = b"0123456789abcdef";
    let bytes = digest.as_bytes();
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(NIBBLES[usize::from(byte >> 4)]));
        output.push(char::from(NIBBLES[usize::from(byte & 0x0f)]));
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
