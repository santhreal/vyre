//! Generated coverage for operation signatures that carry execution contracts.
//!
//! `OpSignature` is the frozen handoff between spec, conformance runners, and
//! backend vendors. This matrix pins that attaching `OperationContract`
//! metadata never changes byte accounting, never drops capability ordering, and
//! remains serde-stable across thousands of generated type/contract shapes.

mod spec_variants;

use smallvec::smallvec;
use vyre_spec::SignatureParam;
use vyre_spec::{
    CapabilityId, CostHint, DataType, DeterminismClass, OpSignature, OperationContract,
    QuantizationScale, QuantizationZeroPoint, SideEffectClass, TypeId,
};

use spec_variants::{QUANTIZED_STORAGE_TYPES, SCALAR_LEAF_TYPES};

#[test]
fn generated_op_signatures_with_contracts_round_trip_for_8192_cases() {
    let mut checked = 0usize;
    for seed in 0u64..8192 {
        let inputs = generated_inputs(seed);
        let output = generated_type(seed ^ 0xfeed_face_cafe_babe);
        let input_params = generated_params(seed, &inputs);
        let output_params = generated_params(seed.rotate_left(17), std::slice::from_ref(&output));
        let contract = generated_contract(seed);
        let expected_min_input_bytes = inputs.iter().map(DataType::min_bytes).sum::<usize>();
        let expected_caps = contract
            .capability_requirements
            .as_ref()
            .map(|caps| {
                caps.iter()
                    .map(|capability| capability.as_str().to_owned())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let signature = OpSignature {
            inputs,
            output,
            input_params: Some(input_params),
            output_params: Some(output_params),
            contract: Some(contract),
        };

        assert_eq!(
            signature.min_input_bytes(),
            expected_min_input_bytes,
            "case {seed}: contract metadata must not affect input byte accounting"
        );

        let json = serde_json::to_string(&signature)
            .expect("Fix: contracted OpSignature must serialize through the frozen spec API.");
        let decoded: OpSignature = serde_json::from_str(&json).expect(
            "Fix: contracted OpSignature JSON must deserialize through the frozen spec API.",
        );

        assert_eq!(decoded, signature, "case {seed}: serde round-trip drift");
        assert_eq!(
            decoded.min_input_bytes(),
            expected_min_input_bytes,
            "case {seed}: decoded byte accounting drift"
        );
        let decoded_caps = decoded
            .contract
            .as_ref()
            .and_then(|contract| contract.capability_requirements.as_ref())
            .map(|caps| {
                caps.iter()
                    .map(|capability| capability.as_str().to_owned())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        assert_eq!(
            decoded_caps, expected_caps,
            "case {seed}: capability order is part of the frozen contract"
        );
        checked += 1;
    }
    assert_eq!(checked, 8192);
}

#[test]
fn operation_contract_missing_json_fields_default_to_none() {
    let decoded: OperationContract = serde_json::from_str("{}")
        .expect("Fix: empty OperationContract JSON must deserialize with default fields.");
    assert_eq!(decoded, OperationContract::none());

    let partial: OperationContract = serde_json::from_str(r#"{"capability_requirements":[]}"#)
        .expect("Fix: explicit empty capability list must deserialize.");
    assert!(partial
        .capability_requirements
        .as_ref()
        .is_some_and(|capabilities| capabilities.is_empty()));
    assert_eq!(partial.determinism, None);
    assert_eq!(partial.side_effect, None);
    assert_eq!(partial.cost_hint, None);
}

fn generated_inputs(seed: u64) -> Vec<DataType> {
    let len = (seed as usize % 8) + 1;
    (0..len)
        .map(|index| generated_type(next_state(seed ^ index as u64)))
        .collect()
}

fn generated_params(seed: u64, types: &[DataType]) -> Vec<SignatureParam> {
    types
        .iter()
        .enumerate()
        .map(|(index, ty)| SignatureParam {
            name: format!("p{index}_{}", seed.rotate_left((index & 63) as u32) & 0xff),
            ty: ty.clone(),
            metadata: ((seed >> (index & 15)) & 1 == 1)
                .then(|| format!("generated role {index} seed {seed}")),
        })
        .collect()
}

fn generated_contract(seed: u64) -> OperationContract {
    OperationContract {
        capability_requirements: Some(match seed % 5 {
            0 => smallvec![],
            1 => smallvec![CapabilityId::new("cuda")],
            2 => smallvec![
                CapabilityId::new("cuda"),
                CapabilityId::new("resident_dispatch"),
            ],
            3 => smallvec![
                CapabilityId::new("cuda_graph"),
                CapabilityId::new("cooperative_launch"),
                CapabilityId::new("async_copy"),
            ],
            _ => smallvec![
                CapabilityId::new("multi_gpu"),
                CapabilityId::new("collectives"),
                CapabilityId::new("quantized_tensor_cores"),
                CapabilityId::new("gpudirect_storage"),
            ],
        }),
        determinism: Some(match (seed >> 3) % 3 {
            0 => DeterminismClass::Deterministic,
            1 => DeterminismClass::DeterministicModuloRounding,
            _ => DeterminismClass::NonDeterministic,
        }),
        side_effect: Some(match (seed >> 7) % 5 {
            0 => SideEffectClass::Pure,
            1 => SideEffectClass::ReadsMemory,
            2 => SideEffectClass::WritesMemory,
            3 => SideEffectClass::Synchronizing,
            _ => SideEffectClass::Atomic,
        }),
        cost_hint: Some(match (seed >> 11) % 4 {
            0 => CostHint::Cheap,
            1 => CostHint::Medium,
            2 => CostHint::Expensive,
            _ => CostHint::Unknown,
        }),
    }
}

fn generated_type(seed: u64) -> DataType {
    let leaves = SCALAR_LEAF_TYPES.len() as u64;
    let idx = seed % (leaves + 6);
    if idx < leaves {
        return SCALAR_LEAF_TYPES[idx as usize].clone();
    }

    match idx - leaves {
        0 => DataType::Handle(TypeId((seed >> 8) as u32)),
        1 => DataType::Array {
            element_size: ((seed >> 13) as usize % 64) + 1,
        },
        2 => DataType::Vec {
            element: Box::new(DataType::U32),
            count: ((seed >> 17) as u8 % 16) + 1,
        },
        3 => DataType::SparseCsr {
            element: Box::new(DataType::F32),
        },
        4 => DataType::DeviceMesh {
            axes: [((seed >> 19) as u32 % 8) + 1, ((seed >> 23) as u32 % 8) + 1]
                .as_slice()
                .into(),
        },
        _ => DataType::Quantized {
            storage: Box::new(
                QUANTIZED_STORAGE_TYPES[(seed >> 29) as usize % QUANTIZED_STORAGE_TYPES.len()]
                    .clone(),
            ),
            scale: QuantizationScale::PerGroup {
                group_size: ((seed >> 31) as u32 % 256) + 1,
            },
            zero_point: match (seed >> 39) % 3 {
                0 => QuantizationZeroPoint::Absent,
                1 => QuantizationZeroPoint::PerTensor,
                _ => QuantizationZeroPoint::PerGroup {
                    group_size: ((seed >> 41) as u32 % 256) + 1,
                },
            },
        },
    }
}

fn next_state(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}
