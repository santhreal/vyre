use super::*;

/// TEST-034: a certificate signed against the reference executor alone proves
/// nothing, so `prove` must refuse instead of emitting one.
///
/// The refusal used to be proved only in a `not(feature = "gpu")` build, which
/// compiled the test out of every build that ships. The selected backend set is
/// an argument, so the refusal is reachable in any build: naming the reference
/// oracle asks for exactly the set `prove` must reject, and the registry answers
/// from its inventory without acquiring a device.
#[test]
fn prove_refuses_a_certificate_proving_the_reference_against_itself() {
    let out_path = std::env::temp_dir().join(format!(
        "vyre-conform-prove-refuses-{}.json",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&out_path);
    let output = Command::new(conform_binary())
        .args(["prove", "--backend", "cpu-ref", "--out"])
        .arg(&out_path)
        .output()
        .expect("Fix: the built vyre-conform binary must launch");
    assert!(
        !output.status.success(),
        "TEST-034: prove against the reference oracle alone must exit non-zero; stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("refused to emit"),
        "TEST-034: prove must explain why it refused to emit the certificate; stderr={stderr}"
    );
    assert!(
        stderr.contains("reference dispatch backends"),
        "TEST-034: the refusal must name the reference-only backend set as the reason; stderr={stderr}"
    );
    assert!(
        !out_path.exists(),
        "TEST-034: prove must not leave a certificate file on disk when it refuses"
    );
}
