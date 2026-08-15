//! WGSL dumping: compute entry rendering, line prefixing, and naga validation failures.
use vyre_debug::{dump_wgsl, dump_wgsl_with_lines};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};

#[path = "support/mod.rs"]
mod support;
use support::minimal_program;

#[test]
fn dump_wgsl_minimal_program_returns_compute_entry() {
    let p = minimal_program();
    let dump = dump_wgsl(&p).unwrap();
    assert!(dump.text.contains("@compute @workgroup_size"));
    assert!(dump.text.contains("fn main"));
}

#[test]
fn dump_wgsl_with_lines_prefixes_each_line() {
    let p = minimal_program();
    let dump = dump_wgsl_with_lines(&p).unwrap();
    for line in dump.text.lines() {
        if !line.trim().is_empty() {
            // Regex: ^\s*\d+ \|
            let trimmed = line.trim_start();
            let mut parts = trimmed.splitn(2, " | ");
            let num = parts.next().unwrap();
            assert!(num.parse::<usize>().is_ok());
            assert!(parts.next().is_some());
        }
    }
}

#[test]
fn dump_wgsl_propagates_naga_validation_failure() {
    // A zero workgroup axis passes frontend construction but fails lowering or Naga validation.
    let buffer =
        BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(16);
    let p = Program::wrapped(
        vec![buffer],
        [0, 1, 1], // Workgroup size 0 is valid in vyre but invalid in Naga
        vec![Node::Store {
            buffer: Ident::from("out"),
            index: Expr::InvocationId { axis: 0 },
            value: Expr::u32(7),
        }],
    );
    let err = match dump_wgsl(&p) {
        Err(e) => e,
        Ok(_) => panic!("Expected error"),
    };
    assert!(
        err.to_lowercase().contains("failed") || err.to_lowercase().contains("error"),
        "Expected error message to contain 'failed' or 'error', got: {}",
        err
    );
}
