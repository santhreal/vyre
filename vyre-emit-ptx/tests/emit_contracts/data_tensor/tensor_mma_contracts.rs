use super::*;
use vyre_lower::descriptor_builder::{body, descriptor, effect, global_wo, lit, SlotCount};

#[test]
fn matrix_mma_emits_real_mma_sync_and_binds_all_four_results() {
    let mut ops = Vec::new();
    let mut literals = Vec::new();
    for id in 0..6 {
        literals.push(LiteralValue::U32(id));
        ops.push(lit(id, id));
    }
    for id in 6..10 {
        literals.push(LiteralValue::F32(0.0));
        ops.push(lit(id, id));
    }
    ops.push(KernelOp {
        kind: mma_f16_m16n8k16(),
        operands: (0..10).collect(),
        result: Some(10),
    });
    ops.push(lit(10, 14));
    literals.push(LiteralValue::U32(0));
    ops.push(effect(KernelOpKind::StoreGlobal, [0, 14, 13]));

    let kernel = descriptor("mma")
        .slot(global_wo(0, DataType::F32, "out").with_count(1))
        .dispatch(32, 1, 1)
        .body(body().ops(ops).literals(literals))
        .build();

    vyre_lower::verify(&kernel)
        .expect("MatrixMma must publish result ids base..base+4 to verifier");
    let s = emit_with_target(&kernel, ComputeCapability::SM_70).unwrap();
    assert!(s.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
    assert!(
        s.contains("st.global.f32"),
        "fourth MatrixMma result id must be usable by later ops"
    );
}

/// Build a matrix multiply kernel from `spec`, feeding it the operand words the
/// specification derives and storing one accumulator word.
fn mma_kernel(id: &str, spec: MatrixMmaSpec) -> KernelDescriptor {
    let words = spec
        .operand_count()
        .expect("Fix: the fixture spec must be carryable");
    let mut ops = Vec::new();
    let mut literals = Vec::new();
    let inputs = words - spec.result_count().unwrap();
    for value in 0..words {
        literals.push(if value < inputs {
            LiteralValue::U32(value)
        } else {
            LiteralValue::F32(0.0)
        });
        ops.push(lit(value, value));
    }
    let result = words;
    ops.push(KernelOp {
        kind: KernelOpKind::MatrixMma(Box::new(spec)),
        operands: (0..words).collect(),
        result: Some(result),
    });
    let index = result + spec.result_count().unwrap();
    ops.push(lit(u32::try_from(literals.len()).unwrap(), index));
    literals.push(LiteralValue::U32(0));
    ops.push(effect(KernelOpKind::StoreGlobal, [0, index, result]));

    descriptor(id)
        .slot(global_wo(0, DataType::F32, "out").with_count(1))
        .dispatch(32, 1, 1)
        .body(body().ops(ops).literals(literals))
        .build()
}

fn f16_spec(tile: MatrixTileShape) -> MatrixMmaSpec {
    MatrixMmaSpec {
        tile,
        left: FragmentValue::in_registers(MatrixMmaElement::F16, MatrixMmaLayout::RowMajor, 32),
        right: FragmentValue::in_registers(MatrixMmaElement::F16, MatrixMmaLayout::ColMajor, 32),
        accumulator: FragmentValue::in_registers(
            MatrixMmaElement::F32,
            MatrixMmaLayout::RowMajor,
            32,
        ),
    }
}

#[test]
fn a_declared_form_this_target_has_no_instruction_for_is_rejected() {
    // Every one of these verifies: the descriptor can state them, and only the
    // target decides which one it has an instruction for.
    let native = f16_spec(MatrixTileShape { m: 16, n: 8, k: 16 });
    let wider_tile = f16_spec(MatrixTileShape { m: 32, n: 8, k: 16 });
    let mut swapped_layout = native;
    swapped_layout.right =
        FragmentValue::in_registers(MatrixMmaElement::F16, MatrixMmaLayout::RowMajor, 32);
    let mut narrow_subgroup = native;
    narrow_subgroup.left =
        FragmentValue::in_registers(MatrixMmaElement::F16, MatrixMmaLayout::RowMajor, 16);
    narrow_subgroup.right =
        FragmentValue::in_registers(MatrixMmaElement::F16, MatrixMmaLayout::ColMajor, 16);
    narrow_subgroup.accumulator =
        FragmentValue::in_registers(MatrixMmaElement::F32, MatrixMmaLayout::RowMajor, 16);
    let mut bf16_inputs = native;
    bf16_inputs.left =
        FragmentValue::in_registers(MatrixMmaElement::BF16, MatrixMmaLayout::RowMajor, 32);
    bf16_inputs.right =
        FragmentValue::in_registers(MatrixMmaElement::BF16, MatrixMmaLayout::ColMajor, 32);
    let mut staged_accumulator = native;
    staged_accumulator.accumulator.access = Some(TensorAccessMap {
        storage: MemoryClass::Scratch,
        row_stride: 0,
        alignment: 16,
    });

    for (name, spec) in [
        ("wider_tile", wider_tile),
        ("swapped_layout", swapped_layout),
        ("narrow_subgroup", narrow_subgroup),
        ("bf16_inputs", bf16_inputs),
        ("staged_accumulator", staged_accumulator),
    ] {
        let kernel = mma_kernel(name, spec);
        vyre_lower::verify(&kernel).unwrap_or_else(|errors| {
            panic!("Fix: {name} must be a statable descriptor: {errors:?}")
        });
        let error = emit_with_target(&kernel, ComputeCapability::SM_70).expect_err(&format!(
            "Fix: {name} has no native form and must be rejected"
        ));
        assert!(
            matches!(error, EmitError::UnsupportedOp(_)),
            "Fix: {name} must be reported as unsupported, not mislowered; got {error:?}"
        );
    }

    // The one form the target does have still emits, so the rejection above is
    // selection and not a blanket refusal.
    let kernel = mma_kernel("native", native);
    let text = emit_with_target(&kernel, ComputeCapability::SM_70).unwrap();
    assert!(text.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
}

#[test]
fn an_operand_list_that_disagrees_with_the_declaration_is_rejected() {
    let mut kernel = mma_kernel(
        "short_operands",
        f16_spec(MatrixTileShape { m: 16, n: 8, k: 16 }),
    );
    let mma = kernel
        .body
        .ops
        .iter_mut()
        .find(|op| matches!(op.kind, KernelOpKind::MatrixMma(_)))
        .expect("Fix: the fixture must carry the matrix op");
    mma.operands.pop();
    let error = emit_with_target(&kernel, ComputeCapability::SM_70)
        .expect_err("Fix: an operand list shorter than the declaration must be rejected");
    match error {
        EmitError::InvalidDescriptor(message) => assert!(
            message.contains("declares 10 operand words"),
            "Fix: the diagnostic must state the declared arity; got {message}"
        ),
        other => panic!("Fix: expected an invalid-descriptor rejection; got {other:?}"),
    }
}
