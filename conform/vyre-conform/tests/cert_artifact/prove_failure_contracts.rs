#[cfg(not(feature = "gpu"))]
use super::*;

#[cfg(not(feature = "gpu"))]
#[test]
fn prove_refuses_certificate_when_backend_cannot_dispatch() {
    // A no-default-features build links no dispatch-capable target. `prove`
    // must refuse rather than emit an untested certificate.
    let out_path = std::env::temp_dir().join(format!(
        "vyre-conform-prove-refuses-{}.json",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&out_path);
    let output = Command::new(conform_binary())
        .args(["prove", "--out"])
        .arg(&out_path)
        .output()
        .expect("Fix: the built vyre-conform binary must launch");
    assert!(
        !output.status.success(),
        "TEST-034: prove without a dispatch-capable backend must exit non-zero; stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("refused to emit"),
        "TEST-034: prove must explain why it refused to emit the certificate; stderr={stderr}"
    );
    assert!(
        !out_path.exists(),
        "TEST-034: prove must not leave a certificate file on disk when parity fails"
    );
}

// Drives `cargo run -p vyre-conform --features gpu -- prove`; GPU
