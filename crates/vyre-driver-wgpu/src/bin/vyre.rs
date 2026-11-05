#![deny(unsafe_code)]
//! Installable `vyre` command-line entry point.
//!
//! The `demo` subcommand builds a minimal vyre IR Program (write the
//! value 42 into an output buffer), dispatches it via the wgpu
//! backend, and prints the resulting u32. This is the canonical
//! "vyre works on this machine" smoke test  -  deliberately uses
//! vyre's IR + Program + VyreBackend surface, NOT raw wgpu. If the
//! demo ever needs raw WGSL to work, that's a failure of vyre's
//! abstraction, not the demo's shape.

use std::process::ExitCode;

use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("--version") | Some("-V") => {
            println!("vyre {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some("demo") => match args.next().as_deref() {
            None => {
                let value = run_demo()?;
                println!("vyre demo gpu_u32={value}");
                Ok(())
            }
            Some("-h" | "--help") => {
                if let Some(extra) = args.next() {
                    Err(format!(
                        "unexpected argument `{extra}` after `--help`. Fix: use `vyre demo --help`."
                    ))
                } else {
                    print_demo_help();
                    Ok(())
                }
            }
            Some(other) => Err(format!(
                "unexpected demo argument `{other}`. Fix: use `vyre demo --help`."
            )),
        },
        Some("--help") | Some("-h") | None => {
            print_help();
            Ok(())
        }
        Some(other) => Err(format!(
            "unknown vyre command `{other}`. Fix: use `vyre --version` or `vyre demo`."
        )),
    }
}

fn print_help() {
    println!("vyre {}", env!("CARGO_PKG_VERSION"));
    println!("Run a minimal Vyre IR program on the local WGPU device.");
    println!();
    println!("Usage: vyre [--version] <COMMAND>");
    println!();
    println!("Commands:");
    println!("  demo  dispatch one u32 write and verify the exact result 42");
    println!();
    println!("Options:");
    println!("  -h, --help     print this help");
    println!("  -V, --version  print the Vyre version");
    println!();
    println!("Exit codes:");
    println!("  0  help, version, or GPU demo completed");
    println!("  1  invalid arguments, device acquisition, dispatch, or output validation failed");
}

fn print_demo_help() {
    println!("Dispatch one generated Vyre IR program on the local WGPU device.");
    println!();
    println!("Usage: vyre demo");
    println!();
    println!("Hardware:");
    println!("  A Vulkan, Metal, DX12, or WebGPU compute device is required.");
    println!("  The command never falls back to CPU.");
    println!();
    println!("Output:");
    println!("  vyre demo gpu_u32=42");
}

fn run_demo() -> Result<u32, String> {
    // Pure vyre IR: one read-write u32 buffer, one Store node writing
    // the literal 42 at index 0. No WGSL, no naga, no hand-written
    // kernel  -  the backend lowers this to a compute pipeline and
    // returns the bytes.
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32)
            .with_count(1)
            .with_full_output_byte_range()],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    );

    let backend = vyre_driver_wgpu::WgpuBackend::acquire().map_err(|error| {
        format!(
            "failed to acquire wgpu backend: {error}. Fix: install a compatible GPU driver \
             (Vulkan / Metal / DX12) or run on a host with GPU access."
        )
    })?;

    // Input buffer list mirrors the program's non-output buffers; our
    // demo has only one read-write output so inputs is empty.
    let outputs = backend
        .dispatch(&program, &[], &DispatchConfig::default())
        .map_err(|error| {
            format!("demo dispatch failed: {error}. Fix: inspect the GPU driver logs.")
        })?;

    validate_demo_outputs(&outputs)
}

fn validate_demo_outputs(outputs: &[Vec<u8>]) -> Result<u32, String> {
    let [bytes] = outputs else {
        return Err(format!(
            "demo returned {} output buffers; expected one.",
            outputs.len()
        ));
    };
    let slice: [u8; 4] = bytes.as_slice().try_into().map_err(|_| {
        format!(
            "demo output buffer contains {} bytes; expected 4.",
            bytes.len()
        )
    })?;
    let value = u32::from_le_bytes(slice);
    if value != 42 {
        return Err(format!("demo returned {value}; expected 42."));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::validate_demo_outputs;

    /// WHY: the demo must reject an incorrect result or output ABI instead of
    /// reporting GPU success or panicking. These tests cover validation, not
    /// device acquisition or dispatch, which require the real GPU demo.
    #[test]
    fn demo_requires_one_output_buffer() {
        for count in 0..=3 {
            let outputs = vec![42_u32.to_le_bytes().to_vec(); count];
            let result = validate_demo_outputs(&outputs);
            if count == 1 {
                assert_eq!(result, Ok(42));
            } else {
                assert_eq!(
                    result,
                    Err(format!(
                        "demo returned {count} output buffers; expected one."
                    ))
                );
            }
        }
    }

    #[test]
    fn demo_requires_exact_output_width() {
        for width in 0..=8 {
            let mut bytes = 42_u32.to_le_bytes().to_vec();
            bytes.resize(width, 0);
            let result = validate_demo_outputs(&[bytes]);
            if width == 4 {
                assert_eq!(result, Ok(42));
            } else {
                assert_eq!(
                    result,
                    Err(format!(
                        "demo output buffer contains {width} bytes; expected 4."
                    ))
                );
            }
        }
    }

    #[test]
    fn demo_requires_the_expected_little_endian_value() {
        for value in [0_u32, 1, 41, 42, 43, u32::MAX, 42_u32.swap_bytes()] {
            let result = validate_demo_outputs(&[value.to_le_bytes().to_vec()]);
            if value == 42 {
                assert_eq!(result, Ok(42));
            } else {
                assert_eq!(result, Err(format!("demo returned {value}; expected 42.")));
            }
        }
    }
}
