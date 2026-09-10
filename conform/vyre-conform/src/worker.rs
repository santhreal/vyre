//! Disposable conformance worker process execution.
//!
//! A worker runs in its own process, holds one device lease, decodes inputs independently,
//! executes under explicit budgets, and emits an authenticated execution receipt.

use std::io::{Read, Write};
use std::time::Instant;

use vyre::ir::Program;
use vyre_conform_spec::{hash_outputs, WorkerMode, WorkerReceipt, WorkerRequest, WorkerStatus};

use crate::backend_selection::backend_registration;
use crate::oracle::OracleSession;
use crate::production::{ProductionSession, CONFORMANCE_SCHEDULES};

/// Secret used to authenticate worker reports when none is provided via environment.
pub const DEFAULT_WORKER_SECRET: &[u8] = b"vyre.conform.worker.internal.secret.v1";

/// Bound on one worker request read from stdin.
///
/// A request carries one case, its inputs and its budgets. A coordinator that
/// writes more than this is not sending a request, and reading a pipe to the end
/// lets whatever is on the other side decide how much of this process's memory
/// it takes.
pub const MAX_WORKER_REQUEST_BYTES: u64 = 67_108_864;

/// Bound on the binary whose digest identifies this worker.
pub const MAX_WORKER_BINARY_BYTES: u64 = 1_073_741_824;

/// Bound on one `/proc` line read for a memory estimate.
const MAX_PROC_STATM_BYTES: u64 = 4_096;

/// Run the worker stdio loop: read request from stdin, execute, write receipt to stdout.
pub fn run_worker_stdio() {
    let mut stdin_bytes = Vec::new();
    if let Err(e) = std::io::stdin()
        .take(MAX_WORKER_REQUEST_BYTES)
        .read_to_end(&mut stdin_bytes)
    {
        eprintln!("worker failed to read request from stdin: {e}");
        std::process::exit(1);
    }
    let secret = std::env::var("VYRE_CONFORM_WORKER_SECRET")
        .ok()
        .and_then(|hex_str| hex::decode(hex_str).ok())
        .unwrap_or_else(|| DEFAULT_WORKER_SECRET.to_vec());

    let request: WorkerRequest = match serde_json::from_slice(&stdin_bytes) {
        Ok(req) => req,
        Err(e) => {
            let error_receipt = WorkerReceipt {
                receipt_version: WorkerReceipt::SCHEMA_VERSION,
                request_id: "unknown".to_string(),
                case_id: "unknown".to_string(),
                mode: WorkerMode::Reference,
                status: WorkerStatus::ProtocolViolation {
                    message: format!("invalid JSON worker request: {e}"),
                },
                outputs: Vec::new(),
                outputs_blake3: String::new(),
                artifact_blake3: None,
                payload_blake3: None,
                target_facts_blake3: None,
                compiler_binary_blake3: String::new(),
                runner_binary_blake3: String::new(),
                environment_blake3: String::new(),
                device_lease_id: None,
                elapsed_ms: 0,
                peak_memory_bytes: 0,
                auth_tag: String::new(),
            };
            let mut signed = error_receipt;
            signed.auth_tag = signed.compute_auth_tag(&secret);
            if let Ok(json) = serde_json::to_string(&signed) {
                println!("{json}");
            }
            std::process::exit(0);
        }
    };

    let receipt = execute_worker_request(&request, &secret);
    if let Ok(json) = serde_json::to_string(&receipt) {
        let mut stdout = std::io::stdout();
        let _ = stdout.write_all(json.as_bytes());
        let _ = stdout.write_all(b"\n");
        let _ = stdout.flush();
    }
}

/// Check if the worker environment variable is set; if so, run worker stdio and exit immediately.
pub fn run_worker_from_env_or_exit() {
    if std::env::var("VYRE_CONFORM_WORKER").is_ok() {
        run_worker_stdio();
        std::process::exit(0);
    }
}

/// Execute a worker request within this process and return an authenticated receipt.
#[must_use]
pub fn execute_worker_request(request: &WorkerRequest, secret: &[u8]) -> WorkerReceipt {
    let started = Instant::now();
    let compiler_binary = current_binary_digest();
    let runner_binary = current_binary_digest();
    let environment = current_environment_digest();

    let execution_result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match request.mode {
            WorkerMode::Reference => execute_reference(request),
            WorkerMode::Production => execute_production(request),
        }));

    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let peak_memory_bytes = estimate_process_memory();

    let mut receipt = match execution_result {
        Err(panic_payload) => {
            let msg = crate::panic_payload::panic_message(panic_payload);
            WorkerReceipt {
                receipt_version: WorkerReceipt::SCHEMA_VERSION,
                request_id: request.request_id.clone(),
                case_id: request.case.case_id.clone(),
                mode: request.mode,
                status: WorkerStatus::Panicked {
                    message: msg.to_string(),
                },
                outputs: Vec::new(),
                outputs_blake3: String::new(),
                artifact_blake3: None,
                payload_blake3: None,
                target_facts_blake3: None,
                compiler_binary_blake3: compiler_binary,
                runner_binary_blake3: runner_binary,
                environment_blake3: environment,
                device_lease_id: request.device_lease.as_ref().map(|l| l.lease_id.clone()),
                elapsed_ms,
                peak_memory_bytes,
                auth_tag: String::new(),
            }
        }
        Ok(Err(exec_err)) => WorkerReceipt {
            receipt_version: WorkerReceipt::SCHEMA_VERSION,
            request_id: request.request_id.clone(),
            case_id: request.case.case_id.clone(),
            mode: request.mode,
            status: WorkerStatus::ExecutionError { message: exec_err },
            outputs: Vec::new(),
            outputs_blake3: String::new(),
            artifact_blake3: None,
            payload_blake3: None,
            target_facts_blake3: None,
            compiler_binary_blake3: compiler_binary,
            runner_binary_blake3: runner_binary,
            environment_blake3: environment,
            device_lease_id: request.device_lease.as_ref().map(|l| l.lease_id.clone()),
            elapsed_ms,
            peak_memory_bytes,
            auth_tag: String::new(),
        },
        Ok(Ok(success_data)) => {
            let total_output_bytes =
                success_data.outputs.iter().map(Vec::len).sum::<usize>() as u64;
            if total_output_bytes > request.budget.max_output_bytes {
                WorkerReceipt {
                    receipt_version: WorkerReceipt::SCHEMA_VERSION,
                    request_id: request.request_id.clone(),
                    case_id: request.case.case_id.clone(),
                    mode: request.mode,
                    status: WorkerStatus::Leak {
                        bytes_leaked: total_output_bytes,
                        message: format!(
                            "total output size {total_output_bytes} exceeded ceiling {}",
                            request.budget.max_output_bytes
                        ),
                    },
                    outputs: Vec::new(),
                    outputs_blake3: String::new(),
                    artifact_blake3: success_data.artifact_blake3,
                    payload_blake3: success_data.payload_blake3,
                    target_facts_blake3: success_data.target_facts_blake3,
                    compiler_binary_blake3: compiler_binary,
                    runner_binary_blake3: runner_binary,
                    environment_blake3: environment,
                    device_lease_id: request.device_lease.as_ref().map(|l| l.lease_id.clone()),
                    elapsed_ms,
                    peak_memory_bytes,
                    auth_tag: String::new(),
                }
            } else {
                let outputs_blake3 = hash_outputs(&success_data.outputs);
                WorkerReceipt {
                    receipt_version: WorkerReceipt::SCHEMA_VERSION,
                    request_id: request.request_id.clone(),
                    case_id: request.case.case_id.clone(),
                    mode: request.mode,
                    status: WorkerStatus::Success,
                    outputs: success_data.outputs,
                    outputs_blake3,
                    artifact_blake3: success_data.artifact_blake3,
                    payload_blake3: success_data.payload_blake3,
                    target_facts_blake3: success_data.target_facts_blake3,
                    compiler_binary_blake3: compiler_binary,
                    runner_binary_blake3: runner_binary,
                    environment_blake3: environment,
                    device_lease_id: request.device_lease.as_ref().map(|l| l.lease_id.clone()),
                    elapsed_ms,
                    peak_memory_bytes,
                    auth_tag: String::new(),
                }
            }
        }
    };

    receipt.auth_tag = receipt.compute_auth_tag(secret);
    receipt
}

struct WorkerSuccess {
    outputs: Vec<Vec<u8>>,
    artifact_blake3: Option<String>,
    payload_blake3: Option<String>,
    target_facts_blake3: Option<String>,
}

fn execute_reference(request: &WorkerRequest) -> Result<WorkerSuccess, String> {
    let program = Program::from_wire(&request.case.program_wire)
        .map_err(|e| format!("program wire decode failed in reference worker: {e}"))?;
    let session = OracleSession::new(program);
    let inputs_borrowed: Vec<&[u8]> = request.case.inputs.iter().map(Vec::as_slice).collect();
    let outputs = session
        .execute(&inputs_borrowed)
        .map_err(|e| format!("oracle execution failed: {e}"))?;
    Ok(WorkerSuccess {
        outputs,
        artifact_blake3: None,
        payload_blake3: None,
        target_facts_blake3: None,
    })
}

fn execute_production(request: &WorkerRequest) -> Result<WorkerSuccess, String> {
    let program = Program::from_wire(&request.case.program_wire)
        .map_err(|e| format!("program wire decode failed in production worker: {e}"))?;
    let lease = request
        .device_lease
        .as_ref()
        .ok_or_else(|| "missing device lease in production worker request".to_string())?;

    let registration = backend_registration(&lease.backend_id)
        .map_err(|e| format!("backend `{}` registration failed: {e}", lease.backend_id))?;

    let mut session = ProductionSession::from_registration(&program, registration)
        .map_err(|e| format!("production session creation failed: {e}"))?;

    if let Some(ref schedule_name) = request.case.schedule_family {
        if let Some((_, required)) = CONFORMANCE_SCHEDULES
            .iter()
            .find(|(name, _)| *name == schedule_name.as_str())
        {
            session = session.requiring_schedule(*required);
        }
    }

    let inputs_borrowed: Vec<&[u8]> = request.case.inputs.iter().map(Vec::as_slice).collect();
    let execution = session
        .submit(&inputs_borrowed)
        .map_err(|e| format!("production submit failed: {e}"))?;

    let target_facts_blake3 = registration.acquire().ok().map(|handle| {
        let facts = handle.device_profile().compile_facts();
        let debug_str = format!("{facts:?}");
        blake3::hash(debug_str.as_bytes()).to_hex().to_string()
    });
    Ok(WorkerSuccess {
        outputs: execution.outputs,
        artifact_blake3: Some(execution.artifact.to_hex().to_string()),
        payload_blake3: Some(execution.payload.to_hex().to_string()),
        target_facts_blake3,
    })
}

/// Compute a canonical digest of the current binary.
///
/// The binary is hashed in fixed-size chunks, so the digest costs one buffer
/// instead of a second copy of the executable in memory, and a file past
/// [`MAX_WORKER_BINARY_BYTES`] is refused rather than read.
#[must_use]
pub fn current_binary_digest() -> String {
    if let Some(digest) = std::env::current_exe()
        .ok()
        .and_then(|exe_path| hash_file_bounded(&exe_path))
    {
        return digest;
    }
    let fallback = format!("vyre-conform-bin-{}", env!("CARGO_PKG_VERSION"));
    blake3::hash(fallback.as_bytes()).to_hex().to_string()
}

/// Hash one file in fixed-size chunks, refusing anything past the bound.
fn hash_file_bounded(path: &std::path::Path) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 65_536];
    let mut total = 0u64;
    loop {
        let filled = file.read(&mut buffer).ok()?;
        if filled == 0 {
            return Some(hasher.finalize().to_hex().to_string());
        }
        total = total.checked_add(filled as u64)?;
        if total > MAX_WORKER_BINARY_BYTES {
            return None;
        }
        hasher.update(&buffer[..filled]);
    }
}

/// Compute a canonical digest of the execution environment.
#[must_use]
pub fn current_environment_digest() -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vyre.conform.environment.v1");
    hasher.update(std::env::consts::OS.as_bytes());
    hasher.update(std::env::consts::ARCH.as_bytes());
    hasher.update(std::env::consts::FAMILY.as_bytes());
    hasher.finalize().to_hex().to_string()
}

fn estimate_process_memory() -> u64 {
    /// The resident-set estimate when `/proc` does not answer.
    const ASSUMED_RESIDENT_BYTES: u64 = 1_048_576;
    /// Bytes per page in the `statm` counts.
    const PAGE_BYTES: u64 = 4_096;

    #[cfg(target_os = "linux")]
    {
        if let Some(pages) = read_first_statm_count() {
            return pages.saturating_mul(PAGE_BYTES);
        }
    }
    ASSUMED_RESIDENT_BYTES
}

/// The first `/proc/self/statm` count, read under a fixed bound.
#[cfg(target_os = "linux")]
fn read_first_statm_count() -> Option<u64> {
    let file = std::fs::File::open("/proc/self/statm").ok()?;
    let mut statm = String::new();
    file.take(MAX_PROC_STATM_BYTES)
        .read_to_string(&mut statm)
        .ok()?;
    statm.split_whitespace().next()?.parse::<u64>().ok()
}
