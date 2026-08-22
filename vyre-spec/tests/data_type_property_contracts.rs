//! Generated property coverage for `DataType` layout and serde contracts.

mod spec_variants;

use proptest::prelude::*;
use spec_variants::{data_type_strategy, quantized_storage_strategy};
use vyre_spec::{DataType, QuantizationScale, QuantizationZeroPoint};

proptest! {
    #[test]
    fn generated_data_types_round_trip_through_json(ty in data_type_strategy()) {
        let encoded = serde_json::to_string(&ty)
            .expect("Fix: generated DataType must serialize through the frozen spec contract");
        let decoded: DataType = serde_json::from_str(&encoded)
            .expect("Fix: generated DataType JSON must deserialize through the frozen spec contract");

        prop_assert_eq!(decoded, ty);
    }

    #[test]
    fn generated_data_type_layout_bounds_are_coherent(ty in data_type_strategy()) {
        if let Some(max_bytes) = ty.max_bytes() {
            prop_assert!(
                ty.min_bytes() <= max_bytes,
                "Fix: min_bytes must never exceed max_bytes for {ty}: min={} max={max_bytes}",
                ty.min_bytes()
            );
        }

        if let (Some(bit_width), Some(size_bytes)) = (ty.bit_width(), ty.size_bytes()) {
            prop_assert!(
                size_bytes.saturating_mul(8) >= bit_width,
                "Fix: size_bytes must have enough bits for {ty}: size={size_bytes}, bits={bit_width}"
            );
        }

        prop_assert!(!ty.to_string().is_empty(), "Fix: DataType display must never be empty");
        prop_assert!(
            ty.validate_layout().is_ok(),
            "Fix: generated valid DataType strategy produced malformed layout metadata for {ty}"
        );
    }

    #[test]
    fn generated_quantized_datatypes_preserve_storage_width(
        storage in quantized_storage_strategy(),
        group_size in 1u32..=512,
    ) {
        let ty = DataType::Quantized {
            storage: Box::new(storage.clone()),
            scale: QuantizationScale::PerGroup { group_size },
            zero_point: QuantizationZeroPoint::PerGroup { group_size },
        };

        prop_assert!(ty.is_quantized());
        prop_assert!(storage.is_quantized_storage());
        prop_assert_eq!(ty.bit_width(), storage.bit_width());
        prop_assert_eq!(ty.size_bytes(), storage.size_bytes());
        prop_assert!(!ty.is_float_family());
    }
}
