//! A declared shared extent covers every byte the zero prologue stores.
//!
//! The prologue zeroes shared memory with `st.shared.u32`, so a declaration
//! whose byte length is not a whole number of words either leaves its last
//! bytes holding whatever the previous CTA wrote, or has the loop store past
//! the symbol. `workgroup_zero_init.rs` pins five hand-picked byte counts of
//! one `u8` buffer. The arithmetic that produces the padded length runs for
//! every element type and every count, and the element type is what decides
//! whether the product is already word-aligned, so the input space is a
//! product of two axes rather than a list.
//!
//! The type axis is read from `DataType::SCALAR_LEAVES` at run time. Adding a
//! scalar type puts it under this property without editing the file.
//!
//! What this does not catch: whether the zeroed bytes are visible to the lanes
//! that did not store them. The barrier that makes them visible is pinned in
//! `workgroup_zero_init.rs`.

use proptest::prelude::*;
use vyre_foundation::ir::DataType;
use vyre_lower::descriptor_builder::{body, descriptor, shared_rw};

/// `.shared .align 4 .b8 <symbol>[<bytes>];`
fn shared_declarations(ptx: &str) -> Vec<(String, u32)> {
    ptx.lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix(".shared .align 4 .b8 ")?;
            let (symbol, rest) = rest.split_once('[')?;
            let bytes = rest.strip_suffix("];")?;
            Some((
                symbol.to_string(),
                bytes
                    .parse()
                    .expect("Fix: emit a decimal byte length in a shared declaration"),
            ))
        })
        .collect()
}

/// Text from the zero prologue up to the barrier that closes it.
fn zero_prologue(ptx: &str) -> Option<&str> {
    let start = ptx.find("    // Workgroup memory holds zero at entry.\n")?;
    let tail = &ptx[start..];
    let end = tail.find("    bar.sync 0;\n")?;
    Some(&tail[..end])
}

/// The scalar types this emitter can size, with their element widths.
fn sizable_scalars() -> Vec<(DataType, u32)> {
    DataType::SCALAR_LEAVES
        .iter()
        .filter_map(|data_type| {
            let size = u32::try_from(data_type.size_bytes()?).ok()?;
            (size > 0).then_some((data_type.clone(), size))
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1_500))]

    /// WHY: the padded byte length is what the zero loop's word count is
    /// derived from, so an extent that rounds the wrong way either leaves the
    /// tail of the buffer holding the previous CTA's bytes or stores past the
    /// symbol. Both failures need a type whose width does not divide four and
    /// a count that lands mid-word, which no fixed case list reaches for every
    /// type in the workspace.
    #[test]
    fn a_shared_extent_is_declared_and_zeroed_at_word_granularity(
        type_index in any::<prop::sample::Index>(),
        count in 1u32..=4_096,
    ) {
        let scalars = sizable_scalars();
        prop_assert!(
            !scalars.is_empty(),
            "Fix: keep at least one sizable scalar in DataType::SCALAR_LEAVES; an empty type axis makes this property vacuous"
        );
        let (data_type, element_size) = type_index.get(&scalars).clone();
        let kernel = descriptor("shared_extent")
            .slot(shared_rw(0, data_type.clone(), count, "s"))
            .dispatch(64, 1, 1)
            .body(body())
            .build();

        let Ok(ptx) = vyre_emit_ptx::emit(&kernel) else {
            // A refused descriptor declares nothing, so there is no extent to
            // get wrong. Refusal is an outcome; a wrong declaration is not.
            return Ok(());
        };

        let Some(bytes) = count.checked_mul(element_size) else {
            return Ok(());
        };
        let expected = bytes
            .checked_next_multiple_of(4)
            .expect("Fix: keep the padded shared extent inside u32");

        prop_assert_eq!(
            shared_declarations(&ptx),
            vec![("shared_buf_0".to_string(), expected)],
            "Fix: declare {} element(s) of {:?} as {} byte(s) so the word store loop covers the extent",
            count,
            data_type,
            expected
        );

        let prologue = zero_prologue(&ptx)
            .expect("Fix: emit a zero prologue for a kernel that declares shared memory");
        prop_assert!(
            prologue.contains(&format!(", {};\n", expected / 4)),
            "Fix: zero {} word(s) for a {}-byte shared extent",
            expected / 4,
            expected
        );
    }

    /// WHY: the prologue emits one bounded loop per declared buffer, and the
    /// bound is per-buffer. A prologue that reused one word count across
    /// buffers would zero the shortest of them and leave the rest holding the
    /// previous CTA's bytes, which a single-buffer case cannot observe.
    #[test]
    fn every_declared_buffer_gets_its_own_bounded_zero_loop(
        counts in prop::collection::vec(1u32..=512, 1..=4),
    ) {
        let mut kernel = descriptor("many_shared");
        for (slot, count) in counts.iter().enumerate() {
            let index = u32::try_from(slot).expect("Fix: keep the slot count inside u32");
            kernel = kernel.slot(shared_rw(index, DataType::U32, *count, &format!("s{slot}")));
        }
        let kernel = kernel.dispatch(64, 1, 1).body(body()).build();

        let Ok(ptx) = vyre_emit_ptx::emit(&kernel) else {
            return Ok(());
        };
        let declarations = shared_declarations(&ptx);
        prop_assert_eq!(
            declarations.len(),
            counts.len(),
            "Fix: declare one shared symbol per shared slot"
        );

        let prologue = zero_prologue(&ptx)
            .expect("Fix: emit a zero prologue for a kernel that declares shared memory");
        prop_assert_eq!(
            prologue.matches("setp.ge.u32").count(),
            declarations.len(),
            "Fix: bound one zero loop per declared shared buffer"
        );
        prop_assert_eq!(
            prologue.matches("bra $L_zero_shared").count(),
            declarations.len() * 2,
            "Fix: both exit and repeat each zero loop"
        );
        for (_, bytes) in &declarations {
            prop_assert!(
                prologue.contains(&format!(", {};\n", bytes / 4)),
                "Fix: bound the zero loop of a {bytes}-byte buffer at {} word(s)",
                bytes / 4
            );
        }
    }
}
