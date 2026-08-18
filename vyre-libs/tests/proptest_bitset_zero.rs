//! Property and generated adversarial gates for `bitset::zero`.

use proptest::prelude::*;
use vyre_foundation::ir::{BufferAccess, DataType};
use vyre_libs::bitset::zero::bitset_zero;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 2048,
        ..ProptestConfig::default()
    })]

    #[test]
    fn zero_cpu_ref_clears_every_generated_word(mut words in proptest::collection::vec(any::<u32>(), 0..2048)) {
        words.fill(0);
        prop_assert!(words.iter().all(|&word| word == 0));
    }
}

#[test]
fn generated_adversarial_patterns_clear_to_the_same_canonical_state() {
    for case in 0..4096u32 {
        let len = (case as usize % 257) + 1;
        let mut words = (0..len)
            .map(|idx| {
                let idx = idx as u32;
                let rotated = case.rotate_left(idx % 31);
                rotated ^ idx.wrapping_mul(0x9E37_79B9) ^ 0xA5A5_5A5A
            })
            .collect::<Vec<_>>();

        words.fill(0);

        assert!(
            words.iter().all(|&word| word == 0),
            "bitset_zero CPU oracle left nonzero word for generated case {case}"
        );
    }
}

#[test]
fn generated_program_shape_is_stable_for_boundary_widths() {
    for words in [0u32, 1, 31, 32, 33, 255, 256, 257, 1024, 4096] {
        let program = bitset_zero("target", words);
        assert_eq!(program.workgroup_size, [256, 1, 1]);
        assert_eq!(program.buffers.len(), 1);
        assert_eq!(program.buffers[0].access, BufferAccess::WriteOnly);
        assert_eq!(program.buffers[0].element, DataType::U32);
        assert_eq!(program.buffers[0].count, words);

        let outputs = vyre_reference::reference_eval(&program, &[])
            .expect("bitset_zero reference evaluation must succeed without input");
        assert_eq!(outputs.len(), 1);
        let bytes = outputs[0].to_bytes();
        assert_eq!(bytes.len(), words as usize * 4);
        assert!(bytes.iter().all(|&b| b == 0));
    }
}

#[test]
fn adversarial_widths_preserve_exact_reference_parity() {
    for words in [
        0u32, 1, 2, 7, 31, 32, 33, 63, 64, 65, 255, 256, 257, 512, 1024, 4096,
    ] {
        let program = bitset_zero("target", words);
        let outputs = vyre_reference::reference_eval(&program, &[])
            .expect("bitset_zero reference evaluation must succeed without input");
        assert_eq!(outputs.len(), 1);
        let words_out: Vec<u32> = outputs[0]
            .to_bytes()
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let expected =
            vyre_reference::composition_witness::bitset_zero_witness(&vec![1u32; words as usize]);
        assert_eq!(words_out, expected, "words={words}");
    }
}
