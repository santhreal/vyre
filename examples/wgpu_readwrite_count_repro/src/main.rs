//! Regression harness for runtime-sized `ReadWrite` buffers.
//!
//! Run it from this directory:
//!
//! ```sh
//! cargo run
//! ```
//!
//! Every case builds the same element-wise XOR over four `u32` elements. A
//! `ReadWrite` declaration without `.with_count()` is runtime-sized and takes
//! its element count from the caller-supplied bytes. A backend-allocated output
//! has no caller bytes, so it must declare a count or an output byte range.
//!
//! A correct run reports four identical words for both `ReadWrite` cases and
//! the statically sized output case. Every backend rejects the countless
//! backend-allocated output with actionable guidance.

use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};
// `WgpuBackend::dispatch` comes from the `VyreBackend` trait, so the trait must
// be in scope. `CudaBackend::dispatch` is an inherent method and does not need
// it. Importing the trait covers both.
use vyre::{DispatchConfig, VyreBackend};
use vyre_reference::value::Value;

const LEN: u32 = 4;

fn program(out: BufferDecl) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32).with_count(LEN),
            BufferDecl::read("b", 1, DataType::U32).with_count(LEN),
            out,
        ],
        [64, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::store(
                "out",
                Expr::var("idx"),
                Expr::bitxor(
                    Expr::load("a", Expr::var("idx")),
                    Expr::load("b", Expr::var("idx")),
                ),
            ),
        ],
    )
}

fn a_bytes() -> Vec<u8> {
    (0..LEN).flat_map(|i| (0xF0 + i).to_le_bytes()).collect()
}

fn b_bytes() -> Vec<u8> {
    (0..LEN).flat_map(|_| 0x0Fu32.to_le_bytes()).collect()
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Supply one buffer per declaration the backend does not allocate itself.
///
/// The runtime-sized `ReadWrite` result receives 16 caller-owned bytes. Its
/// logical count must therefore be four on every backend.
fn supplied(program: &Program) -> Vec<Vec<u8>> {
    program
        .buffers()
        .iter()
        .filter(|d| !d.is_backend_allocated_output())
        .map(|d| match d.name() {
            "a" => a_bytes(),
            "b" => b_bytes(),
            _ => vec![0u8; (LEN * 4) as usize],
        })
        .collect()
}

/// Decode the first returned buffer, so an empty result stays visible as `[]`
/// rather than being mistaken for a missing buffer.
fn first_words(outputs: Vec<Vec<u8>>) -> Result<Vec<u32>, String> {
    match outputs.into_iter().next() {
        Some(bytes) => Ok(words(&bytes)),
        None => Err("backend returned no output buffers".to_string()),
    }
}

fn report(case: &str, path: &str, result: Result<Vec<u32>, String>) {
    match result {
        Ok(words) => println!("[{case}] {path:<9} = Some({words:?})"),
        Err(error) => println!("[{case}] {path:<9} = error: {error}"),
    }
}

fn main() {
    let cuda = vyre_driver_cuda::CudaBackend::acquire();
    let wgpu = vyre_driver_wgpu::WgpuBackend::acquire();
    if let Err(error) = &cuda {
        println!("note: CUDA unavailable on this host: {error}");
    }
    if let Err(error) = &wgpu {
        println!("note: WGPU unavailable on this host: {error}");
    }

    let cases = [
        (
            "read_write, no count",
            BufferDecl::read_write("out", 2, DataType::U32),
        ),
        (
            "output, no count",
            BufferDecl::output("out", 2, DataType::U32),
        ),
        (
            "read_write, count=4",
            BufferDecl::read_write("out", 2, DataType::U32).with_count(LEN),
        ),
        (
            "output, count=4",
            BufferDecl::output("out", 2, DataType::U32).with_count(LEN),
        ),
    ];

    for (case, out) in cases {
        let program = program(out);
        let inputs = supplied(&program);

        let reference_inputs: Vec<Value> = program
            .buffers()
            .iter()
            .filter(|d| vyre_reference::is_reference_input(d))
            .map(|d| match d.name() {
                "a" => Value::Bytes(a_bytes().into()),
                "b" => Value::Bytes(b_bytes().into()),
                _ => Value::Bytes(vec![0u8; (LEN * 4) as usize].into()),
            })
            .collect();

        report(
            case,
            "reference",
            vyre_reference::reference_eval(&program, &reference_inputs)
                .map_err(|e| e.to_string())
                .and_then(|out| match out.into_iter().next() {
                    Some(Value::Bytes(bytes)) => Ok(words(&bytes)),
                    other => Err(format!("unexpected reference output: {other:?}")),
                }),
        );
        if let Ok(backend) = cuda.as_ref() {
            report(
                case,
                "cuda",
                backend
                    .dispatch(&program, &inputs, &DispatchConfig::default())
                    .map_err(|e| e.to_string())
                    .and_then(first_words),
            );
        }
        if let Ok(backend) = wgpu.as_ref() {
            report(
                case,
                "wgpu",
                backend
                    .dispatch(&program, &inputs, &DispatchConfig::default())
                    .map_err(|e| e.to_string())
                    .and_then(first_words),
            );
        }
        println!();
    }
}
