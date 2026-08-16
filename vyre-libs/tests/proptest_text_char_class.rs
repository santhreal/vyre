//! Property gates for `vyre_libs::text::reference_char_class`.

#![cfg(all(feature = "text", feature = "cpu-parity"))]
mod text_char_class_runner;

use proptest::prelude::*;
use text_char_class_runner::run_packed_u8_program;
use vyre_foundation::ir::DataType;
use vyre_libs::text::{build_char_class_table, char_class_u8, reference_char_class};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn every_ascii_byte_maps_via_table(
        bytes in proptest::collection::vec(any::<u8>(), 0..=256),
    ) {
        let table = build_char_class_table();
        let result = reference_char_class(&bytes, &table);
        let expected: Vec<u32> = bytes.iter().map(|b| table[usize::from(*b)]).collect();
        prop_assert_eq!(result, expected);
    }

    #[test]
    fn table_is_deterministic(_dummy in 0u32..1) {
        let t1 = build_char_class_table();
        let t2 = build_char_class_table();
        prop_assert_eq!(t1, t2);
    }

    #[test]
    fn packed_u8_builder_keeps_byte_source(
        n in 0u32..=4096,
    ) {
        let program = char_class_u8("source", "classified", n);
        let has_u8_source = program.buffers().iter().any(|buffer| {
            buffer.name() == "source"
                && buffer.element() == DataType::U8
                && buffer.count() == n
        });
        let has_u32_table = program.buffers().iter().any(|buffer| {
            buffer.name() == "table"
                && buffer.element() == DataType::U32
                && buffer.count() == 256
        });
        let has_u32_classified = program.buffers().iter().any(|buffer| {
            buffer.name() == "classified"
                && buffer.element() == DataType::U32
                && buffer.count() == n.max(1)
                && buffer.output_byte_range()
                    == Some(0..usize::try_from(n).unwrap_or(usize::MAX).saturating_mul(4))
                && buffer.is_output()
        });

        prop_assert!(has_u8_source, "char_class_u8 source must be packed U8 for n={n}");
        prop_assert!(has_u32_table, "char_class_u8 table must remain a 256-entry U32 lookup table for n={n}");
        prop_assert!(has_u32_classified, "char_class_u8 output must remain one U32 class per source byte for n={n}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2_048))]

    #[test]
    fn packed_u8_program_matches_table_reference(
        source in proptest::collection::vec(any::<u8>(), 0..=256),
        table_values in proptest::collection::vec(any::<u32>(), 256..=256),
    ) {
        let mut table = [0u32; 256];
        table.copy_from_slice(&table_values);
        prop_assert_eq!(
            run_packed_u8_program(&source, &table),
            reference_char_class(&source, &table)
        );
    }
}
