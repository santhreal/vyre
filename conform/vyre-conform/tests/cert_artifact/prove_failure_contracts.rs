use super::*;
use vyre_driver_reference::ORACLE_EXECUTOR_ID;

/// The certificate path a refusal case may never create.
fn refused_artifact_path(case: &str) -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let path = dir.join(format!(
        "vyre-conform-prove-refuses-{case}-{}.json",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    path
}

/// Run `prove --backend <id>` and return its stderr, requiring a refusal that
/// wrote no certificate.
fn refusal_for_backend(case: &str, backend: &str) -> String {
    let out_path = refused_artifact_path(case);
    let output = Command::new(conform_binary())
        .args(["prove", "--backend", backend, "--out"])
        .arg(&out_path)
        .output()
        .expect("Fix: the built vyre-conform binary must launch");
    assert!(
        !output.status.success(),
        "TEST-034: prove against `{backend}` must exit non-zero; stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !out_path.exists(),
        "TEST-034: prove must not leave a certificate file on disk when it refuses"
    );
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// TEST-034: a certificate signed against the reference executor alone proves
/// nothing, so `prove` must refuse instead of emitting one.
///
/// The refusal is decided at backend selection. The oracle submits no
/// `BackendRegistration`, so its id resolves to nothing in the registry and
/// would otherwise be reported as a typo. `select_backends` names the oracle
/// ahead of that lookup so the caller is told what it asked for rather than
/// that it misspelled something, and `semantic_execution_backends` still
/// excludes a `reference_oracle` registration for an out-of-tree one.
#[test]
fn prove_refuses_a_certificate_proving_the_reference_against_itself() {
    let stderr = refusal_for_backend("reference", ORACLE_EXECUTOR_ID);
    assert!(
        stderr.contains("refused to emit"),
        "TEST-034: prove must explain why it refused to emit the certificate; stderr={stderr}"
    );
    assert!(
        stderr.contains("is the reference oracle, not a backend"),
        "TEST-034: the refusal must say the id is the oracle rather than a backend; stderr={stderr}"
    );
    assert!(
        stderr.contains(&format!("`{ORACLE_EXECUTOR_ID}`")),
        "TEST-034: the refusal must name the rejected id; stderr={stderr}"
    );
}

/// A backend id nothing registered is a different refusal from the reference
/// oracle, and saying so is the whole value of the message: one is a typo, the
/// other is a proof that would certify nothing.
#[test]
fn prove_separates_an_unregistered_backend_from_the_reference_oracle() {
    let stderr = refusal_for_backend("unknown", "no-such-backend");
    assert!(
        stderr.contains("refused to emit"),
        "prove must state that it refused; stderr={stderr}"
    );
    assert!(
        stderr.contains("unknown backend `no-such-backend`"),
        "an unregistered id must be reported as unknown; stderr={stderr}"
    );
    assert!(
        !stderr.contains("reference oracle"),
        "an unregistered id is not the reference oracle; stderr={stderr}"
    );
}
