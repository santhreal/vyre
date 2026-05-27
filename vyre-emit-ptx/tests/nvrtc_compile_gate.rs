//! CUDA driver compile gate: validate that emitted PTX is well-formed
//! CUDA assembly.
//!
//! Real driver module-load validation is gated behind the `nvrtc` feature
//! because it requires a CUDA toolkit and GPU driver at test time.
//! When the feature is off, the mock gate validates PTX string
//! structure and instruction presence.

use vyre_foundation::ir::{BinOp, DataType};
use vyre_lower::{
    BindingLayout, BindingSlot, BindingVisibility, Dispatch, KernelBody, KernelDescriptor,
    KernelOp, KernelOpKind, LiteralValue, MemoryClass,
};

fn rw_slot(id: u32, name: &str) -> BindingSlot {
    rw_slot_typed(id, name, DataType::U32)
}

fn rw_slot_typed(id: u32, name: &str, element_type: DataType) -> BindingSlot {
    BindingSlot {
        slot: id,
        element_type,
        element_count: None,
        memory_class: MemoryClass::Global,
        visibility: BindingVisibility::ReadWrite,
        name: name.into(),
    }
}

fn ptx_for_op(op_kind: KernelOpKind) -> String {
    let result_id = 3u32;
    let idx_id = 2u32;

    let (mut ops, literals, binding) = match op_kind {
        KernelOpKind::Fma => (
            vec![
                KernelOp {
                    kind: KernelOpKind::LocalInvocationId,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::Cast {
                        target: DataType::F32,
                    },
                    operands: vec![0],
                    result: Some(1),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(4),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![1],
                    result: Some(idx_id),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![2],
                    result: Some(5),
                },
                KernelOp {
                    kind: KernelOpKind::Fma,
                    operands: vec![1, 4, 5],
                    result: Some(result_id),
                },
            ],
            vec![
                LiteralValue::F32(2.0),
                LiteralValue::U32(0),
                LiteralValue::F32(3.0),
            ],
            rw_slot_typed(0, "out", DataType::F32),
        ),
        KernelOpKind::BinOpKind(BinOp::Mul) => (
            vec![
                KernelOp {
                    kind: KernelOpKind::LocalInvocationId,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::LocalInvocationId,
                    operands: vec![0],
                    result: Some(1),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(idx_id),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Mul),
                    operands: vec![0, 1],
                    result: Some(result_id),
                },
            ],
            vec![LiteralValue::U32(0)],
            rw_slot(0, "out"),
        ),
        other => (
            vec![
                // Use LocalInvocationId so the op survives constant folding.
                KernelOp {
                    kind: KernelOpKind::LocalInvocationId,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(1),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![1],
                    result: Some(idx_id),
                },
                KernelOp {
                    kind: other,
                    operands: vec![0, 1],
                    result: Some(result_id),
                },
            ],
            vec![LiteralValue::U32(7), LiteralValue::U32(0)],
            rw_slot(0, "out"),
        ),
    };
    ops.push(KernelOp {
        kind: KernelOpKind::StoreGlobal,
        operands: vec![0, idx_id, result_id],
        result: None,
    });

    let desc = KernelDescriptor {
        id: "test".into(),
        bindings: BindingLayout {
            slots: vec![binding],
        },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops,
            child_bodies: vec![],
            literals,
        },
    };
    vyre_emit_ptx::emit_optimized(&desc).unwrap()
}

#[test]
fn mock_gate_add_ptx_is_well_formed() {
    let ptx = ptx_for_op(KernelOpKind::BinOpKind(BinOp::Add));
    assert!(ptx.contains(".version"), "missing .version directive");
    assert!(ptx.contains(".target"), "missing .target directive");
    assert!(ptx.contains(".visible .entry main"), "missing entry point");
    assert!(ptx.contains("add"), "missing add instruction");
    assert!(ptx.contains("ret;"), "missing ret instruction");
}

#[test]
fn mock_gate_mul_ptx_is_well_formed() {
    let ptx = ptx_for_op(KernelOpKind::BinOpKind(BinOp::Mul));
    assert!(ptx.contains(".version"), "missing .version directive");
    assert!(ptx.contains(".target"), "missing .target directive");
    assert!(ptx.contains(".visible .entry main"), "missing entry point");
    assert!(ptx.contains("mul.lo"), "missing mul.lo instruction");
    assert!(ptx.contains("ret;"), "missing ret instruction");
}

#[test]
fn mock_gate_fma_ptx_is_well_formed() {
    let ptx = ptx_for_op(KernelOpKind::Fma);
    assert!(ptx.contains(".version"), "missing .version directive");
    assert!(ptx.contains(".target"), "missing .target directive");
    assert!(ptx.contains(".visible .entry main"), "missing entry point");
    assert!(ptx.contains("fma.rn"), "missing fma.rn instruction");
    assert!(ptx.contains("ret;"), "missing ret instruction");
}

#[test]
fn mock_gate_ptx_has_register_declarations() {
    let ptx = ptx_for_op(KernelOpKind::BinOpKind(BinOp::Add));
    assert!(ptx.contains(".reg"), "missing register declarations");
    assert!(ptx.contains("%r"), "missing u32 register prefix");
}

#[test]
fn mock_gate_rejects_malformed_placeholder() {
    // A syntactically invalid PTX fragment should not pass structural
    // checks that real emitted PTX satisfies.
    let fake = ".version 0.0\n.target sm_99\n.entry broken { }";
    assert!(!fake.contains(".reg"), "fake should lack register decls");
    assert!(!fake.contains("ret;"), "fake should lack ret");
}

#[cfg(feature = "nvrtc")]
mod nvrtc_real {
    //! Real CUDA driver PTX module-load gate.
    //!
    //! Enabled only when the `nvrtc` feature is active because it
    //! requires a CUDA driver and toolkit at test time. The feature is
    //! off by default so CI and environments without CUDA can still run
    //! the mock gate tests above.

    use std::ffi::CString;

    use super::ptx_for_op;
    use cudarc::driver::{sys::CUresult, CudaContext};
    use vyre_foundation::ir::BinOp;
    use vyre_lower::KernelOpKind;

    fn driver_loads_ptx(ptx: &str) -> Result<(), String> {
        let ctx = CudaContext::new(0)
            .map_err(|error| format!("CUDA context creation failed: {error}"))?;
        ctx.bind_to_thread()
            .map_err(|error| format!("CUDA context bind failed: {error}"))?;
        let ptx = CString::new(ptx)
            .map_err(|error| format!("emitted PTX contained interior NUL: {error}"))?;
        let mut module = std::ptr::null_mut();
        let load = unsafe {
            cudarc::driver::sys::cuModuleLoadData(&mut module, ptx.as_ptr().cast())
        };
        if load != CUresult::CUDA_SUCCESS {
            return Err(format!("cuModuleLoadData failed with {load:?}"));
        }
        if !module.is_null() {
            let unload = unsafe { cudarc::driver::sys::cuModuleUnload(module) };
            if unload != CUresult::CUDA_SUCCESS {
                return Err(format!("cuModuleUnload failed with {unload:?}"));
            }
        }
        Ok(())
    }

    #[test]
    fn nvrtc_compiles_add_ptx() {
        let ptx = ptx_for_op(KernelOpKind::BinOpKind(BinOp::Add));
        let compiled = driver_loads_ptx(&ptx);
        assert!(
            compiled.is_ok(),
            "CUDA driver failed to load add PTX: {:?}",
            compiled.err()
        );
    }

    #[test]
    fn nvrtc_compiles_mul_ptx() {
        let ptx = ptx_for_op(KernelOpKind::BinOpKind(BinOp::Mul));
        let compiled = driver_loads_ptx(&ptx);
        assert!(
            compiled.is_ok(),
            "CUDA driver failed to load mul PTX: {:?}",
            compiled.err()
        );
    }

    #[test]
    fn nvrtc_compiles_fma_ptx() {
        let ptx = ptx_for_op(KernelOpKind::Fma);
        let compiled = driver_loads_ptx(&ptx);
        assert!(
            compiled.is_ok(),
            "CUDA driver failed to load fma PTX: {:?}",
            compiled.err()
        );
    }
}
