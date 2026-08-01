//! End-to-end contracts for the runtime-sized `ReadWrite` regression harness.

use std::process::{Command, Output};
use std::sync::LazyLock;

static RUN: LazyLock<Output> = LazyLock::new(|| {
    Command::new(env!("CARGO_BIN_EXE_wgpu_readwrite_count_repro"))
        .output()
        .expect("Fix: the runtime-sizing regression binary must launch")
});

fn stdout() -> String {
    assert!(
        RUN.status.success(),
        "Fix: the runtime-sizing regression binary must complete successfully. stderr:\n{}",
        String::from_utf8_lossy(&RUN.stderr)
    );
    String::from_utf8(RUN.stdout.clone())
        .expect("Fix: the regression binary must emit UTF-8 operator diagnostics")
}

/// Locks out the original cross-backend count drift for caller-sized `ReadWrite` storage.
#[test]
fn runtime_sized_readwrite_uses_all_caller_supplied_bytes() {
    let output = stdout();
    for backend in ["reference", "cuda", "wgpu"] {
        assert!(
            output.contains(&format!(
                "[read_write, no count] {backend:<9} = Some([255, 254, 253, 252])"
            )),
            "Fix: {backend} must derive the runtime-sized ReadWrite count from all 16 supplied bytes. Output:\n{output}"
        );
    }
}

/// Proves that a backend-allocated result with no count fails loudly on every execution path.
#[test]
fn countless_backend_allocated_output_is_rejected() {
    let output = stdout();
    for backend in ["reference", "cuda", "wgpu"] {
        assert!(
            output.contains(&format!("[output, no count] {backend:<9} = error:")),
            "Fix: {backend} must reject a countless backend-allocated output. Output:\n{output}"
        );
    }
}

/// Preserves exact four-word parity for both statically sized writable result forms.
#[test]
fn static_readwrite_and_output_counts_match_across_backends() {
    let output = stdout();
    for case in ["read_write, count=4", "output, count=4"] {
        for backend in ["reference", "cuda", "wgpu"] {
            assert!(
                output.contains(&format!(
                    "[{case}] {backend:<9} = Some([255, 254, 253, 252])"
                )),
                "Fix: {backend} must preserve exact four-word output for `{case}`. Output:\n{output}"
            );
        }
    }
}
