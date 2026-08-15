//! GPU LZ4 literal-extraction composition.
//!
//! `vyre-primitives::decode::ziftsieve` owns the reusable indexed literal-copy
//! kernel and host oracle. This module keeps the libs-level composition API:
//! scoped buffer names, fixture registration, and stable public exports for
//! decode-to-scan pipelines.

use vyre_foundation::ir::Program;
use vyre_primitives::decode::ziftsieve::{
    ziftsieve_literal_copy_with_op_id, ZiftsieveBuffers, ZiftsieveExtents,
};

// CPU parity oracle: re-exported only for parity tests / the `cpu-parity`
// feature, never as a production decode surface (matches the vyre-primitives
// gating of the underlying helper).
#[cfg(any(test, feature = "cpu-parity"))]
pub use vyre_primitives::decode::ziftsieve::{
    ziftsieve_reference_extract_literals, ZiftsieveExtract,
};

use crate::decode::buffers::{scoped_decode_input_buffer, scoped_decode_output_buffer};
#[cfg(test)]
use vyre_primitives::wire::pack_u32_slice as pack_words;

const OP_ID: &str = "vyre-libs::decode::ziftsieve";
const FAMILY_PREFIX: &str = "decode_ziftsieve";

/// Build a Program that copies LZ4 literals in parallel given a pre-built
/// sequence index.
///
/// The reusable IR and CPU oracle live in `vyre-primitives`; this composition
/// adds exactly two things, and they are what its tests assert: the generic
/// `input` and `output` names are rewritten to family-scoped ones so a fused
/// kernel cannot collide them with another decoder's buffers, and the program
/// carries the libs op id rather than the primitive's.
#[must_use]
pub fn ziftsieve_gpu(buffers: ZiftsieveBuffers<'_>, extents: ZiftsieveExtents) -> Program {
    let input = scoped_decode_input_buffer(FAMILY_PREFIX, buffers.input);
    let output = scoped_decode_output_buffer(
        FAMILY_PREFIX,
        "output",
        buffers.output,
        &["output", "decoded"],
    );
    ziftsieve_literal_copy_with_op_id(
        OP_ID,
        ZiftsieveBuffers {
            input: &input,
            output: &output,
            ..buffers
        },
        extents,
    )
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use vyre_reference::value::Value;

    fn run(input: &[u8], seq_starts: &[u32], seq_lens: &[u32], seq_offsets: &[u32]) -> Vec<u32> {
        let seq_count = seq_starts.len() as u32;
        let max_output = seq_lens.iter().copied().sum::<u32>();
        let input_words = input.iter().map(|&b| u32::from(b)).collect::<Vec<_>>();
        let program = ziftsieve_gpu(
            ZiftsieveBuffers::CANONICAL,
            ZiftsieveExtents {
                input_len: input.len() as u32,
                seq_count,
                max_output,
            },
        );
        let inputs = vec![
            Value::from(pack_words(&input_words)),
            Value::from(pack_words(seq_starts)),
            Value::from(pack_words(seq_lens)),
            Value::from(pack_words(seq_offsets)),
            Value::from(vec![0u8; (max_output.max(1) as usize) * 4]),
        ];
        let outputs = vyre_reference::reference_eval(&program, &inputs)
            .expect("Fix: ziftsieve_gpu wrapper must run.");
        let words = vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[0].to_bytes());
        words.into_iter().take(max_output as usize).collect()
    }

    /// The composition must produce a program that runs end to end. The decode
    /// semantics themselves are the primitive's contract and are proven there,
    /// including the hostile out-of-contract cases; restating them here proved
    /// the primitive twice and the composition not at all.
    #[test]
    fn the_composition_decodes_through_the_primitive() {
        assert_eq!(
            run(&[0x10, b'A', 0x20, b'B', b'C'], &[1, 3], &[1, 2], &[0, 1]),
            vec![b'A' as u32, b'B' as u32, b'C' as u32]
        );
    }

    /// WHY: the generic names are what a fused kernel collides on, so the
    /// rewrite is this module's whole reason to exist and nothing asserted it.
    /// An explicit caller name must survive, or composition by name breaks.
    #[test]
    fn generic_binding_names_are_family_scoped_and_explicit_ones_are_kept() {
        let scoped = ziftsieve_gpu(ZiftsieveBuffers::CANONICAL, ZiftsieveExtents::default());
        let names: Vec<&str> = scoped
            .buffers()
            .iter()
            .map(|buffer| buffer.name())
            .collect();
        assert!(
            names.contains(&"__vyre_decode_ziftsieve_input")
                && names.contains(&"__vyre_decode_ziftsieve_output"),
            "Fix: generic `input`/`output` must be rewritten to family-scoped names, got {names:?}"
        );
        let explicit = ziftsieve_gpu(
            ZiftsieveBuffers {
                input: "block_words",
                output: "literals",
                ..ZiftsieveBuffers::CANONICAL
            },
            ZiftsieveExtents::default(),
        );
        let names: Vec<&str> = explicit
            .buffers()
            .iter()
            .map(|buffer| buffer.name())
            .collect();
        assert!(
            names.contains(&"block_words") && names.contains(&"literals"),
            "Fix: an explicit caller name must be preserved, got {names:?}"
        );
    }

    /// WHY: the libs op id is the other thing this module adds. It reaches the
    /// program identity, so a composition that lost it would be indistinguishable
    /// from the bare primitive in the registry and in every duplicate report.
    #[test]
    fn the_program_carries_the_libs_op_id() {
        let program = ziftsieve_gpu(ZiftsieveBuffers::CANONICAL, ZiftsieveExtents::default());
        assert!(
            !program.structural_eq(&vyre_primitives::decode::ziftsieve::ziftsieve_literal_copy(
                ZiftsieveBuffers::CANONICAL,
                ZiftsieveExtents::default(),
            )),
            "Fix: `{OP_ID}` must reach the program identity instead of the primitive id"
        );
    }

    #[test]
    fn wrapper_reexports_primitive_reference() {
        let result = ziftsieve_reference_extract_literals(&[0x10, b'A'], 1024).unwrap();
        assert_eq!(result.literals, b"A");
        assert!(!result.truncated());
    }
}
