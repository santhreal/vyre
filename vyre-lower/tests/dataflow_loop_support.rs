//! Shared fixtures for dataflow-aware loop rewrite suites.

use vyre_lower::{KernelBody, KernelOp, KernelOpKind};

pub(crate) fn store_body(index: u32, value: u32) -> KernelBody {
    KernelBody {
        ops: vec![KernelOp {
            kind: KernelOpKind::StoreGlobal,
            operands: vec![0, index, value],
            result: None,
        }],
        child_bodies: Vec::new(),
        literals: Vec::new(),
    }
}
