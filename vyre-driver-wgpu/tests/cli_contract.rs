//! Operator-facing `vyre-wgpu` command contracts.

#![cfg(feature = "device-tests")]
#![forbid(unsafe_code)]

use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vyre-wgpu"))
        .args(args)
        .output()
        .expect("Fix: vyre-wgpu binary must launch")
}

/// Prevents top-level help from omitting the real GPU command, version route, or exit semantics.
#[test]
fn top_level_help_documents_the_operator_surface() {
    let output = run(&["--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("Fix: help must be UTF-8");
    assert!(stdout.contains("Usage: vyre [--version] <COMMAND>"));
    assert!(stdout.contains("demo  dispatch one u32 write and verify the exact result 42"));
    assert!(stdout.contains("1  invalid arguments, device acquisition, dispatch"));
}

/// Prevents `demo --help` from acquiring a device or dispatching work merely to print usage.
#[test]
fn demo_help_is_device_independent() {
    let output = run(&["demo", "--help"]);
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).expect("Fix: help must be UTF-8");
    assert_eq!(
        stdout,
        concat!(
            "Dispatch one generated Vyre IR program on the local WGPU device.\n",
            "\n",
            "Usage: vyre demo\n",
            "\n",
            "Hardware:\n",
            "  A Vulkan, Metal, DX12, or WebGPU compute device is required.\n",
            "  The command never falls back to CPU.\n",
            "\n",
            "Output:\n",
            "  vyre demo gpu_u32=42\n",
        )
    );
}

/// Prevents ignored trailing arguments from silently running an unintended GPU dispatch.
#[test]
fn demo_rejects_trailing_arguments_before_device_acquisition() {
    let output = run(&["demo", "unexpected"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert_eq!(
        String::from_utf8(output.stderr).expect("Fix: error must be UTF-8"),
        "unexpected demo argument `unexpected`. Fix: use `vyre demo --help`.\n"
    );
}

/// Keeps version output exact for package managers and operator probes.
#[test]
fn version_output_uses_the_package_version() {
    let output = run(&["--version"]);
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).expect("Fix: version must be UTF-8"),
        format!("vyre {}\n", env!("CARGO_PKG_VERSION"))
    );
}
