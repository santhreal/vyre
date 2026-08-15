//! Generated property coverage for operation signature byte accounting.

mod spec_variants;

use proptest::prelude::*;
use spec_variants::data_type_strategy;
use vyre_spec::SignatureParam;
use vyre_spec::{DataType, OpSignature};

fn signature_param_strategy() -> impl Strategy<Value = SignatureParam> {
    (
        "[a-z][a-z0-9_]{0,24}",
        data_type_strategy(),
        prop::option::of("[a-zA-Z0-9 _./:-]{0,48}"),
    )
        .prop_map(|(name, ty, metadata)| SignatureParam { name, ty, metadata })
}

fn optional_params_strategy() -> impl Strategy<Value = Option<Vec<SignatureParam>>> {
    prop::option::of(prop::collection::vec(signature_param_strategy(), 0..=6))
}

proptest! {
    #[test]
    fn generated_signatures_round_trip_and_preserve_byte_accounting(
        inputs in prop::collection::vec(data_type_strategy(), 0..=8),
        output in data_type_strategy(),
        input_params in optional_params_strategy(),
        output_params in optional_params_strategy(),
    ) {
        let expected_min_input_bytes = inputs.iter().map(DataType::min_bytes).sum::<usize>();
        let signature = OpSignature {
            inputs,
            output,
            input_params,
            output_params,
            contract: None,
        };

        prop_assert_eq!(signature.min_input_bytes(), expected_min_input_bytes);

        let encoded = serde_json::to_string(&signature)
            .expect("Fix: generated OpSignature must serialize through the frozen spec contract");
        let decoded: OpSignature = serde_json::from_str(&encoded)
            .expect("Fix: generated OpSignature JSON must deserialize through the frozen spec contract");

        prop_assert_eq!(decoded.min_input_bytes(), expected_min_input_bytes);
        prop_assert_eq!(decoded, signature);
    }
}
