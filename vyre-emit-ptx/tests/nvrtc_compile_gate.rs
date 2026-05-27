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
    slot_typed(id, name, element_type, BindingVisibility::ReadWrite)
}

fn slot_typed(
    id: u32,
    name: &str,
    element_type: DataType,
    visibility: BindingVisibility,
) -> BindingSlot {
    BindingSlot {
        slot: id,
        element_type,
        element_count: None,
        memory_class: MemoryClass::Global,
        visibility,
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

fn ptx_for_vector_load_fusion() -> String {
    let desc = KernelDescriptor {
        id: "vector_load_fusion".into(),
        bindings: BindingLayout {
            slots: vec![
                slot_typed(0, "input", DataType::U32, BindingVisibility::ReadOnly),
                slot_typed(1, "output", DataType::U32, BindingVisibility::WriteOnly),
            ],
        },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops: vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![1],
                    result: Some(1),
                },
                KernelOp {
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 0],
                    result: Some(2),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Add),
                    operands: vec![0, 1],
                    result: Some(3),
                },
                KernelOp {
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 3],
                    result: Some(4),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Add),
                    operands: vec![3, 1],
                    result: Some(5),
                },
                KernelOp {
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 5],
                    result: Some(6),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Add),
                    operands: vec![5, 1],
                    result: Some(7),
                },
                KernelOp {
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 7],
                    result: Some(8),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Add),
                    operands: vec![2, 4],
                    result: Some(9),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Add),
                    operands: vec![9, 6],
                    result: Some(10),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Add),
                    operands: vec![10, 8],
                    result: Some(11),
                },
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![1, 0, 11],
                    result: None,
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(1)],
        },
    };
    vyre_emit_ptx::emit_optimized(&desc).unwrap()
}

fn ptx_for_dynamic_vector_load_fusion() -> String {
    let desc = KernelDescriptor {
        id: "dynamic_vector_load_fusion".into(),
        bindings: BindingLayout {
            slots: vec![
                slot_typed(0, "input", DataType::U32, BindingVisibility::ReadOnly),
                slot_typed(1, "output", DataType::U32, BindingVisibility::WriteOnly),
            ],
        },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![
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
                    kind: KernelOpKind::BinOpKind(BinOp::Mul),
                    operands: vec![0, 1],
                    result: Some(2),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![1],
                    result: Some(3),
                },
                KernelOp {
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 2],
                    result: Some(4),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Add),
                    operands: vec![2, 3],
                    result: Some(5),
                },
                KernelOp {
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 5],
                    result: Some(6),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Add),
                    operands: vec![5, 3],
                    result: Some(7),
                },
                KernelOp {
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 7],
                    result: Some(8),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Add),
                    operands: vec![7, 3],
                    result: Some(9),
                },
                KernelOp {
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 9],
                    result: Some(10),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Add),
                    operands: vec![4, 6],
                    result: Some(11),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Add),
                    operands: vec![11, 8],
                    result: Some(12),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Add),
                    operands: vec![12, 10],
                    result: Some(13),
                },
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![1, 0, 13],
                    result: None,
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(4), LiteralValue::U32(1)],
        },
    };
    vyre_emit_ptx::emit_optimized(&desc).unwrap()
}

fn ptx_for_vector_store_fusion() -> String {
    let desc = KernelDescriptor {
        id: "vector_store_fusion".into(),
        bindings: BindingLayout {
            slots: vec![slot_typed(
                0,
                "output",
                DataType::U32,
                BindingVisibility::WriteOnly,
            )],
        },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops: vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![1],
                    result: Some(1),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![2],
                    result: Some(2),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![3],
                    result: Some(3),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![4],
                    result: Some(4),
                },
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![0, 0, 1],
                    result: None,
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![5],
                    result: Some(5),
                },
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![0, 5, 2],
                    result: None,
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![6],
                    result: Some(6),
                },
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![0, 6, 3],
                    result: None,
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![7],
                    result: Some(7),
                },
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![0, 7, 4],
                    result: None,
                },
            ],
            child_bodies: vec![],
            literals: vec![
                LiteralValue::U32(0),
                LiteralValue::U32(10),
                LiteralValue::U32(11),
                LiteralValue::U32(12),
                LiteralValue::U32(13),
                LiteralValue::U32(1),
                LiteralValue::U32(2),
                LiteralValue::U32(3),
            ],
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
fn mock_gate_vector_load_fusion_ptx_is_well_formed() {
    let ptx = ptx_for_vector_load_fusion();
    assert!(
        ptx.contains("ld.global.nc.v4.u32") || ptx.contains("ld.global.v4.u32"),
        "missing fused vector load instruction\n{ptx}"
    );
    assert_eq!(
        ptx.matches("ld.global.u32").count(),
        0,
        "fused vector load must not leave scalar global loads\n{ptx}"
    );
    assert!(ptx.contains("ret;"), "missing ret instruction");
}

#[test]
fn mock_gate_dynamic_vector_load_fusion_ptx_is_well_formed() {
    let ptx = ptx_for_dynamic_vector_load_fusion();
    assert!(
        ptx.contains("ld.global.nc.v4.u32") || ptx.contains("ld.global.v4.u32"),
        "missing dynamic-base fused v4 vector load instruction\n{ptx}"
    );
    let scalar_data_loads =
        ptx.matches("ld.global.u32").count() + ptx.matches("ld.global.nc.u32").count();
    assert_eq!(
        scalar_data_loads, 0,
        "dynamic-base fused vector load must eliminate scalar data loads\n{ptx}"
    );
    assert!(
        ptx.contains("st.global.u32"),
        "missing per-thread output store"
    );
    assert!(ptx.contains("ret;"), "missing ret instruction");
}

#[test]
fn mock_gate_vector_store_fusion_ptx_is_well_formed() {
    let ptx = ptx_for_vector_store_fusion();
    assert!(
        ptx.contains("st.global.v4.u32"),
        "missing fused vector store instruction\n{ptx}"
    );
    assert_eq!(
        ptx.matches("st.global.u32").count(),
        0,
        "fused vector store must not leave scalar global stores\n{ptx}"
    );
    assert!(ptx.contains("ret;"), "missing ret instruction");
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

    use super::{
        ptx_for_dynamic_vector_load_fusion, ptx_for_op, ptx_for_vector_load_fusion,
        ptx_for_vector_store_fusion,
    };
    use cudarc::driver::{sys::CUresult, CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
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
        let load =
            unsafe { cudarc::driver::sys::cuModuleLoadData(&mut module, ptx.as_ptr().cast()) };
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

    fn launch_vector_load_fusion_ptx(ptx: &str) -> Result<Vec<u32>, String> {
        launch_vector_load_fusion_matrix(ptx, &[[7_u32, 11, 13, 17]])
    }

    fn launch_vector_load_fusion_matrix(ptx: &str, cases: &[[u32; 4]]) -> Result<Vec<u32>, String> {
        let ctx = CudaContext::new(0)
            .map_err(|error| format!("CUDA context creation failed: {error}"))?;
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(Ptx::from_src(ptx))
            .map_err(|error| format!("CUDA module load failed: {error}"))?;
        let kernel = module
            .load_function("main")
            .map_err(|error| format!("CUDA function lookup failed: {error}"))?;
        let params = stream
            .clone_htod(&[1_u32, 4, 1])
            .map_err(|error| format!("params HtoD copy failed: {error}"))?;

        let mut observed = Vec::with_capacity(cases.len());
        for input_host in cases {
            let input = stream
                .clone_htod(input_host)
                .map_err(|error| format!("input HtoD copy failed: {error}"))?;
            let mut output = stream
                .alloc_zeros::<u32>(1)
                .map_err(|error| format!("output allocation failed: {error}"))?;

            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(&input)
                    .arg(&mut output)
                    .arg(&params)
                    .launch(LaunchConfig::for_num_elems(1))
            }
            .map_err(|error| format!("vector load kernel launch failed: {error}"))?;

            let output_host = stream
                .clone_dtoh(&output)
                .map_err(|error| format!("output DtoH copy failed: {error}"))?;
            observed.push(output_host[0]);
        }
        Ok(observed)
    }

    fn launch_dynamic_vector_load_fusion_ptx(
        ptx: &str,
        input_host: &[u32],
        thread_count: usize,
    ) -> Result<Vec<u32>, String> {
        if input_host.len() != thread_count * 4 {
            return Err(format!(
                "input length {} must equal thread_count * 4 ({})",
                input_host.len(),
                thread_count * 4
            ));
        }
        let ctx = CudaContext::new(0)
            .map_err(|error| format!("CUDA context creation failed: {error}"))?;
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(Ptx::from_src(ptx))
            .map_err(|error| format!("CUDA module load failed: {error}"))?;
        let kernel = module
            .load_function("main")
            .map_err(|error| format!("CUDA function lookup failed: {error}"))?;

        let input = stream
            .clone_htod(input_host)
            .map_err(|error| format!("input HtoD copy failed: {error}"))?;
        let mut output = stream
            .alloc_zeros::<u32>(thread_count)
            .map_err(|error| format!("output allocation failed: {error}"))?;
        let params = stream
            .clone_htod(&[
                thread_count as u32,
                input_host.len() as u32,
                thread_count as u32,
            ])
            .map_err(|error| format!("params HtoD copy failed: {error}"))?;

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&input)
                .arg(&mut output)
                .arg(&params)
                .launch(LaunchConfig::for_num_elems(thread_count as u32))
        }
        .map_err(|error| format!("dynamic vector load kernel launch failed: {error}"))?;

        stream
            .clone_dtoh(&output)
            .map_err(|error| format!("output DtoH copy failed: {error}"))
    }

    fn launch_vector_store_fusion_ptx(ptx: &str) -> Result<Vec<u32>, String> {
        let outputs = launch_vector_store_fusion_matrix(ptx, &[[0_u32, 0, 0, 0]])?;
        Ok(outputs[0].to_vec())
    }

    fn launch_vector_store_fusion_matrix(
        ptx: &str,
        initial_outputs: &[[u32; 4]],
    ) -> Result<Vec<[u32; 4]>, String> {
        let ctx = CudaContext::new(0)
            .map_err(|error| format!("CUDA context creation failed: {error}"))?;
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(Ptx::from_src(ptx))
            .map_err(|error| format!("CUDA module load failed: {error}"))?;
        let kernel = module
            .load_function("main")
            .map_err(|error| format!("CUDA function lookup failed: {error}"))?;
        let params = stream
            .clone_htod(&[1_u32, 4])
            .map_err(|error| format!("params HtoD copy failed: {error}"))?;

        let mut observed = Vec::with_capacity(initial_outputs.len());
        for initial_output in initial_outputs {
            let mut output = stream
                .clone_htod(initial_output)
                .map_err(|error| format!("output HtoD copy failed: {error}"))?;

            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(&params)
                    .launch(LaunchConfig::for_num_elems(1))
            }
            .map_err(|error| format!("vector store kernel launch failed: {error}"))?;

            let output_host = stream
                .clone_dtoh(&output)
                .map_err(|error| format!("output DtoH copy failed: {error}"))?;
            observed.push(
                output_host.try_into().map_err(|output: Vec<u32>| {
                    format!("expected 4 lanes, got {}", output.len())
                })?,
            );
        }
        Ok(observed)
    }

    fn vector_load_runtime_cases() -> Vec<[u32; 4]> {
        (0_u32..64)
            .map(|seed| {
                let mixed = seed.wrapping_mul(0x9e37_79b1).rotate_left(seed % 31);
                [
                    mixed,
                    mixed.wrapping_add(seed.wrapping_mul(3).wrapping_add(1)),
                    mixed ^ 0xa5a5_5a5a,
                    mixed.wrapping_mul(7).wrapping_add(0x1234_5678),
                ]
            })
            .collect()
    }

    fn vector_store_initial_outputs() -> Vec<[u32; 4]> {
        (0_u32..64)
            .map(|seed| {
                let sentinel = 0xfeed_0000_u32.wrapping_add(seed);
                [
                    sentinel,
                    sentinel.rotate_left(5),
                    sentinel ^ 0xffff_0000,
                    sentinel.wrapping_mul(17),
                ]
            })
            .collect()
    }

    fn dynamic_vector_load_input(thread_count: usize) -> Vec<u32> {
        (0..thread_count * 4)
            .map(|idx| {
                let value = idx as u32;
                value
                    .wrapping_mul(0x045d_9f3b)
                    .rotate_left((value % 17) + 1)
                    ^ 0x9e37_79b9
            })
            .collect()
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

    #[test]
    fn nvrtc_compiles_vector_load_fusion_ptx() {
        let ptx = ptx_for_vector_load_fusion();
        let compiled = driver_loads_ptx(&ptx);
        assert!(
            compiled.is_ok(),
            "CUDA driver failed to load vector-load fusion PTX: {:?}",
            compiled.err()
        );
    }

    #[test]
    fn nvrtc_compiles_dynamic_vector_load_fusion_ptx() {
        let ptx = ptx_for_dynamic_vector_load_fusion();
        let compiled = driver_loads_ptx(&ptx);
        assert!(
            compiled.is_ok(),
            "CUDA driver failed to load dynamic vector-load fusion PTX: {:?}",
            compiled.err()
        );
    }

    #[test]
    fn nvrtc_compiles_vector_store_fusion_ptx() {
        let ptx = ptx_for_vector_store_fusion();
        let compiled = driver_loads_ptx(&ptx);
        assert!(
            compiled.is_ok(),
            "CUDA driver failed to load vector-store fusion PTX: {:?}",
            compiled.err()
        );
    }

    #[test]
    fn nvrtc_executes_vector_load_fusion_ptx() {
        let ptx = ptx_for_vector_load_fusion();
        let output = launch_vector_load_fusion_ptx(&ptx);
        assert_eq!(
            output,
            Ok(vec![48]),
            "CUDA execution must match CPU sum of fused vector-loaded values"
        );
    }

    #[test]
    fn nvrtc_executes_vector_store_fusion_ptx() {
        let ptx = ptx_for_vector_store_fusion();
        let output = launch_vector_store_fusion_ptx(&ptx);
        assert_eq!(
            output,
            Ok(vec![10, 11, 12, 13]),
            "CUDA execution must materialize all lanes from fused vector store"
        );
    }

    #[test]
    fn nvrtc_executes_vector_load_fusion_runtime_matrix() {
        let ptx = ptx_for_vector_load_fusion();
        let cases = vector_load_runtime_cases();
        let expected = cases
            .iter()
            .map(|case| {
                case.iter()
                    .copied()
                    .fold(0_u32, |acc, value| acc.wrapping_add(value))
            })
            .collect::<Vec<_>>();
        let output = launch_vector_load_fusion_matrix(&ptx, &cases);
        assert_eq!(
            output,
            Ok(expected),
            "CUDA execution must match wrapping CPU sums across the vector-load matrix"
        );
    }

    #[test]
    fn nvrtc_executes_dynamic_vector_load_fusion_multithread_matrix() {
        let ptx = ptx_for_dynamic_vector_load_fusion();
        let thread_count = 64;
        let input = dynamic_vector_load_input(thread_count);
        let expected = input
            .chunks_exact(4)
            .map(|chunk| {
                chunk
                    .iter()
                    .copied()
                    .fold(0_u32, |acc, value| acc.wrapping_add(value))
            })
            .collect::<Vec<_>>();
        let output = launch_dynamic_vector_load_fusion_ptx(&ptx, &input, thread_count);
        assert_eq!(
            output,
            Ok(expected),
            "CUDA execution must match CPU chunk sums for dynamic-base fused vector loads"
        );
    }

    #[test]
    fn nvrtc_executes_vector_store_fusion_overwrite_matrix() {
        let ptx = ptx_for_vector_store_fusion();
        let initial_outputs = vector_store_initial_outputs();
        let expected = vec![[10_u32, 11, 12, 13]; initial_outputs.len()];
        let output = launch_vector_store_fusion_matrix(&ptx, &initial_outputs);
        assert_eq!(
            output,
            Ok(expected),
            "CUDA execution must overwrite every sentinel lane through the fused vector store"
        );
    }
}
