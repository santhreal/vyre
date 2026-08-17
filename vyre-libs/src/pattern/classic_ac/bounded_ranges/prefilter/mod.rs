//! The candidate gate in front of the bounded-ranges AC replay, and the one
//! program assembler every gated shape in the family is built from.
//!
//! A bounded-ranges program is a gate, a result buffer and a replay. The gate
//! differs by WIDTH: how many trailing haystack bytes it inspects before it
//! admits a position. That width is the only thing that separated the three
//! shipped match-emitting builders, and each of them used to respell the whole
//! input ABI, the scan-node call and the builder quintet to express it. The
//! width is data here ([`PrefilterWidth`]) and the assembly is one function
//! ([`gated_ranges_program`]), so a binding added or resized reaches every shape
//! at once. The bounded walk itself, the range-bound arithmetic and the
//! fail-closed rejection path stay with their owner in the parent module; this
//! module calls them and restates neither.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::pattern::CompiledDfa;

use super::super::count_program::{
    count_suffix2_prefilter_body, suffix3_prefilter_match_nodes, CLASSIC_AC_SUFFIX2_MASK_WORDS,
    CLASSIC_AC_SUFFIX3_BLOOM_WORDS,
};
use super::{
    ac_ranges_output_records_len, ac_ranges_program_or_fail_closed, bounded_ranges_scan_nodes,
    candidate_end_gate_nodes, AcInputBindings,
};

mod suffix3;

pub use suffix3::{
    build_ac_bounded_ranges_suffix3_prefilter_program,
    build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce,
    presence_bitmap_words, presence_by_region_words,
    try_build_ac_bounded_ranges_suffix3_prefilter_program,
    try_build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce,
    try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program,
    try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program_filtered,
    try_build_ac_bounded_ranges_suffix3_presence_by_region_program,
    try_build_ac_bounded_ranges_suffix3_presence_program,
};
/// The canonical buffer names every bounded-ranges dispatch binds, in binding
/// order: inputs 0-5, the match counter at 6, the gate masks, the match sink.
/// The `try_build_*` entrypoints all bind these, so they are spelled once.
const RANGES_BUFFER_NAMES: [&str; 11] = [
    "haystack",
    "transitions",
    "output_offsets",
    "output_records",
    "pattern_lengths",
    "haystack_len",
    "match_count",
    "candidate_end_mask",
    "candidate_suffix2_mask",
    "candidate_suffix3_bloom",
    "matches",
];

/// The binding the first gate mask occupies. Bindings 0-5 are the shared inputs
/// and 6 is the per-shape result buffer, so a gate always starts at 7.
pub(in crate::pattern) const FIRST_GATE_BINDING: u32 = 7;

/// How deep a bounded-ranges candidate gate looks before the DFA replay runs.
///
/// THE width table for the family. Each variant is one shipped gate shape, and
/// every column that differs between shapes is a method here rather than a
/// separate copy of the builder: how many mask buffers the gate binds, which IR
/// it emits, the region generator the program carries, the dispatch-shape name
/// the ABI error and the fail-closed panic quote, and the fallible entrypoint a
/// caller that must recover calls instead. Adding a width means adding a row,
/// which the exhaustive matches below force and
/// `tests/scan_prefilter_width_closure.rs` proves.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::pattern) enum PrefilterWidth {
    /// No candidate mask: every in-bounds position replays the bounded window.
    Unfiltered,
    /// The 8-word end-byte mask alone: a position whose byte cannot terminate
    /// any accepted pattern skips the replay.
    EndByte,
    /// End byte, then the 64Ki-bit bigram mask, then the hashed trigram bloom.
    /// Each stage only narrows, so a false positive still takes the safe replay
    /// and a true match cannot be filtered out.
    Suffix3,
}

impl PrefilterWidth {
    /// Number of mask buffers the gate binds, starting at
    /// [`FIRST_GATE_BINDING`]. The match sink follows them.
    pub(in crate::pattern) const fn mask_count(self) -> u32 {
        match self {
            Self::Unfiltered => 0,
            Self::EndByte => 1,
            Self::Suffix3 => 3,
        }
    }

    /// The region generator the assembled match-emitting program carries.
    const fn scan_generator(self) -> &'static str {
        match self {
            Self::Unfiltered => "vyre-libs::matching::classic_ac_bounded_ranges",
            Self::EndByte => "vyre-libs::matching::classic_ac_bounded_ranges_prefilter",
            Self::Suffix3 => "vyre-libs::matching::classic_ac_bounded_ranges_suffix3_prefilter",
        }
    }

    /// The dispatch shape named in the ABI error and the fail-closed panic.
    const fn dispatch_shape(self) -> &'static str {
        match self {
            Self::Unfiltered => "bounded-ranges",
            Self::EndByte => "bounded-ranges prefilter",
            Self::Suffix3 => "bounded-ranges suffix3 prefilter",
        }
    }

    /// The fallible entrypoint the fail-closed panic tells a caller to use.
    const fn fallible_entrypoint(self) -> &'static str {
        match self {
            Self::Unfiltered => "try_build_ac_bounded_ranges_program_with_subgroup_coalesce",
            Self::EndByte => "try_build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce",
            Self::Suffix3 => {
                "try_build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce"
            }
        }
    }
}

/// A gate of one width, bound to concrete mask buffer names starting at
/// `first_binding`.
///
/// The mask list is a PREFIX of `[end byte, bigram, trigram bloom]`, which is
/// what lets one array serve every width: a narrower gate reads fewer entries
/// and never emits the names it does not bind. The constructors are the only way
/// to build one, so an unbound slot cannot leak into a declaration.
#[derive(Clone, Copy)]
pub(in crate::pattern) struct PrefilterGate<'a> {
    width: PrefilterWidth,
    first_binding: u32,
    masks: [&'a str; 3],
}

impl<'a> PrefilterGate<'a> {
    /// No candidate mask. The program still bounds the invocation against the
    /// live `haystack_len`; it just admits every in-bounds position.
    pub(in crate::pattern) const fn unfiltered() -> Self {
        Self {
            width: PrefilterWidth::Unfiltered,
            first_binding: FIRST_GATE_BINDING,
            masks: ["", "", ""],
        }
    }

    /// The end-byte mask alone, bound at [`FIRST_GATE_BINDING`].
    pub(in crate::pattern) const fn end_byte(end_mask: &'a str) -> Self {
        Self {
            width: PrefilterWidth::EndByte,
            first_binding: FIRST_GATE_BINDING,
            masks: [end_mask, "", ""],
        }
    }

    /// End byte, bigram mask and trigram bloom, bound consecutively from
    /// [`FIRST_GATE_BINDING`].
    pub(in crate::pattern) const fn suffix3(
        end_mask: &'a str,
        suffix2_mask: &'a str,
        suffix3_bloom: &'a str,
    ) -> Self {
        Self {
            width: PrefilterWidth::Suffix3,
            first_binding: FIRST_GATE_BINDING,
            masks: [end_mask, suffix2_mask, suffix3_bloom],
        }
    }

    /// The mask declarations, in binding order. Empty for
    /// [`PrefilterWidth::Unfiltered`].
    pub(in crate::pattern) fn decls(&self) -> Vec<BufferDecl> {
        let counts = [
            8,
            CLASSIC_AC_SUFFIX2_MASK_WORDS as u32,
            CLASSIC_AC_SUFFIX3_BLOOM_WORDS as u32,
        ];
        (0..self.width.mask_count() as usize)
            .map(|mask| {
                BufferDecl::storage(
                    self.masks[mask],
                    self.first_binding + mask as u32,
                    BufferAccess::ReadOnly,
                    DataType::U32,
                )
                .with_count(counts[mask])
            })
            .collect()
    }

    /// Wrap `replay` in this width's candidate gate.
    ///
    /// Every width binds the invocation index `i` and bounds it against the live
    /// `haystack_len` first, so a lane past the end of the haystack does no work
    /// under any width. The deeper widths reuse the `candidate_byte` the outer
    /// byte gate already unpacked instead of loading it again.
    fn gate_nodes(&self, haystack: &str, haystack_len: &str, replay: Vec<Node>) -> Vec<Node> {
        match self.width {
            PrefilterWidth::Unfiltered => vec![
                Node::let_bind("i", Expr::InvocationId { axis: 0 }),
                Node::if_then(
                    Expr::lt(Expr::var("i"), Expr::load(haystack_len, Expr::u32(0))),
                    replay,
                ),
            ],
            PrefilterWidth::EndByte => {
                candidate_end_gate_nodes(haystack, haystack_len, self.masks[0], replay)
            }
            PrefilterWidth::Suffix3 => {
                let suffix3_match_nodes =
                    suffix3_prefilter_match_nodes(haystack, self.masks[2], replay.clone());
                count_suffix2_prefilter_body(
                    haystack,
                    self.masks[0],
                    self.masks[1],
                    haystack_len,
                    replay,
                    suffix3_match_nodes,
                )
            }
        }
    }
}

/// Assemble a gated bounded-ranges program: the shared inputs at bindings 0-5,
/// the per-shape result buffer at 6, this width's mask buffers, then `trailing`.
///
/// THE assembler for the family. The match-emitting scan at every width, the
/// global presence bitmap, the per-region bitmap and the fused
/// presence-and-positions program are all one call to this with a different
/// result buffer, trailing sink and replay, so none of them can drift in the
/// shared input ABI or in what its gate admits.
///
/// The gate leads the parameter list because it is the only argument that
/// differs between shapes; everything behind it is the ABI they share.
pub(in crate::pattern) fn gated_ranges_program(
    gate: PrefilterGate<'_>,
    inputs: AcInputBindings<'_>,
    result: BufferDecl,
    trailing: Vec<BufferDecl>,
    generator: &'static str,
    replay: Vec<Node>,
) -> Program {
    let body = gate.gate_nodes(inputs.haystack, inputs.haystack_len, replay);
    let mut buffers = inputs.decls();
    buffers.reserve(1 + gate.width.mask_count() as usize + trailing.len());
    buffers.push(result);
    buffers.extend(gate.decls());
    buffers.extend(trailing);
    Program::wrapped(
        buffers,
        [128, 1, 1],
        vec![wrap_anonymous_region(generator, body)],
    )
}

/// The match-emitting bounded-ranges scan at one gate width: match counter at
/// binding 6, mask buffers next, `(pattern_id, start, end)` triples in the sink
/// immediately after them.
#[allow(clippy::too_many_arguments)]
pub(in crate::pattern) fn ranges_scan_program(
    gate: PrefilterGate<'_>,
    inputs: AcInputBindings<'_>,
    match_count: &str,
    matches: &str,
    max_matches: u32,
    max_pattern_len: u32,
    use_subgroup_coalesce: bool,
) -> Program {
    let replay = bounded_ranges_scan_nodes(
        inputs.haystack,
        inputs.transitions,
        inputs.output_offsets,
        inputs.output_records,
        inputs.pattern_lengths,
        match_count,
        matches,
        max_pattern_len,
        use_subgroup_coalesce,
    );
    gated_ranges_program(
        gate,
        inputs,
        BufferDecl::read_write(match_count, 6, DataType::U32).with_count(1),
        vec![BufferDecl::output(
            matches,
            FIRST_GATE_BINDING + gate.width.mask_count(),
            DataType::U32,
        )
        .with_count(max_matches.saturating_mul(3))],
        gate.width.scan_generator(),
        replay,
    )
}

/// The gate a width binds under the canonical mask names.
fn canonical_gate(width: PrefilterWidth) -> PrefilterGate<'static> {
    let [.., end_mask, suffix2_mask, suffix3_bloom, _] = RANGES_BUFFER_NAMES;
    match width {
        PrefilterWidth::Unfiltered => PrefilterGate::unfiltered(),
        PrefilterWidth::EndByte => PrefilterGate::end_byte(end_mask),
        PrefilterWidth::Suffix3 => PrefilterGate::suffix3(end_mask, suffix2_mask, suffix3_bloom),
    }
}

/// Build the match-emitting bounded-ranges scan of one width for a compiled DFA
/// under the canonical buffer names.
///
/// The ONE place the input ABI, the DFA metadata narrowing and the gate width
/// meet, so the six public `try_build_*` entrypoints are each a single call and
/// cannot disagree about what they bind.
///
/// # Errors
///
/// Returns an actionable error, naming this width's dispatch shape, when the DFA
/// metadata cannot fit the GPU program's u32 buffer-count ABI.
pub(in crate::pattern) fn try_build_ranges_scan(
    width: PrefilterWidth,
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_matches: u32,
    use_subgroup_coalesce: bool,
) -> Result<Program, String> {
    let [input_names @ .., match_count, _, _, _, matches] = RANGES_BUFFER_NAMES;
    let output_records_len = ac_ranges_output_records_len(dfa, width.dispatch_shape())?;
    Ok(ranges_scan_program(
        canonical_gate(width),
        AcInputBindings::new(
            input_names,
            dfa.state_count,
            output_records_len,
            pattern_count,
        ),
        match_count,
        matches,
        max_matches,
        dfa.max_pattern_len,
        use_subgroup_coalesce,
    ))
}

/// [`try_build_ranges_scan`] for the infallible entrypoints, failing closed
/// through the parent module's shared wrapper rather than substituting a
/// dispatchable program.
pub(in crate::pattern) fn build_ranges_scan(
    width: PrefilterWidth,
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_matches: u32,
    use_subgroup_coalesce: bool,
) -> Program {
    ac_ranges_program_or_fail_closed(
        try_build_ranges_scan(
            width,
            dfa,
            pattern_count,
            max_matches,
            use_subgroup_coalesce,
        ),
        width.dispatch_shape(),
        width.fallible_entrypoint(),
    )
}

/// Build a bounded-window AC ranges program with an exact candidate-end-byte
/// prefilter.
///
/// `candidate_end_mask` is an 8-word bitset where bit `b` is set when byte `b`
/// can terminate at least one accepted DFA state. Non-candidate lanes skip the
/// bounded replay window and match append path entirely.
#[must_use]
#[allow(clippy::too_many_arguments)]
fn classic_ac_bounded_ranges_prefilter_program(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    pattern_lengths: &str,
    haystack_len: &str,
    match_count: &str,
    candidate_end_mask: &str,
    matches: &str,
    state_count: u32,
    output_records_len: u32,
    pattern_count: u32,
    max_matches: u32,
    max_pattern_len: u32,
) -> Program {
    classic_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
        haystack,
        transitions,
        output_offsets,
        output_records,
        pattern_lengths,
        haystack_len,
        match_count,
        candidate_end_mask,
        matches,
        state_count,
        output_records_len,
        pattern_count,
        max_matches,
        max_pattern_len,
        true,
    )
}

/// Variant of [`classic_ac_bounded_ranges_prefilter_program`] with explicit
/// control over subgroup match-append coalescing.
#[must_use]
#[allow(clippy::too_many_arguments)]
fn classic_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    pattern_lengths: &str,
    haystack_len: &str,
    match_count: &str,
    candidate_end_mask: &str,
    matches: &str,
    state_count: u32,
    output_records_len: u32,
    pattern_count: u32,
    max_matches: u32,
    max_pattern_len: u32,
    use_subgroup_coalesce: bool,
) -> Program {
    ranges_scan_program(
        PrefilterGate::end_byte(candidate_end_mask),
        AcInputBindings::new(
            [
                haystack,
                transitions,
                output_offsets,
                output_records,
                pattern_lengths,
                haystack_len,
            ],
            state_count,
            output_records_len,
            pattern_count,
        ),
        match_count,
        matches,
        max_matches,
        max_pattern_len,
        use_subgroup_coalesce,
    )
}

/// Build the candidate-end prefiltered bounded-ranges AC scan for a compiled
/// DFA.
#[must_use]
pub fn build_ac_bounded_ranges_prefilter_program(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_matches: u32,
) -> Program {
    build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
        dfa,
        pattern_count,
        max_matches,
        true,
    )
}

/// Variant of [`build_ac_bounded_ranges_prefilter_program`] that exposes the
/// match-append coalescing selector.
///
/// # Panics
/// Panics when the prefilter program exceeds the GPU ABI limits, through the
/// crate's shared fail-closed wrapper. Callers that must recover use
/// [`try_build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce`].
#[must_use]
pub fn build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_matches: u32,
    use_subgroup_coalesce: bool,
) -> Program {
    build_ranges_scan(
        PrefilterWidth::EndByte,
        dfa,
        pattern_count,
        max_matches,
        use_subgroup_coalesce,
    )
}

/// Fallible variant of [`build_ac_bounded_ranges_prefilter_program`].
///
/// # Errors
///
/// Returns an actionable error when DFA metadata cannot fit the GPU program's
/// u32 buffer-count ABI.
pub fn try_build_ac_bounded_ranges_prefilter_program(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_matches: u32,
) -> Result<Program, String> {
    try_build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
        dfa,
        pattern_count,
        max_matches,
        true,
    )
}

/// Fallible variant of [`build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce`].
///
/// # Errors
///
/// Returns an actionable error when DFA metadata cannot fit the GPU program's
/// u32 buffer-count ABI.
pub fn try_build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_matches: u32,
    use_subgroup_coalesce: bool,
) -> Result<Program, String> {
    try_build_ranges_scan(
        PrefilterWidth::EndByte,
        dfa,
        pattern_count,
        max_matches,
        use_subgroup_coalesce,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern::classic_ac::test_dispatch_and_decode::{
        ac_ranges_inputs, assert_infallible_matches_try, decode_match_triples, pattern_lengths,
        u32_input,
    };
    use crate::pattern::classic_ac::{
        classic_ac_bounded_ranges_scan, classic_ac_candidate_end_byte_mask_words,
        classic_ac_compile,
    };

    #[test]
    fn bounded_ranges_prefilter_reference_eval_matches_reference_oracle() {
        let patterns: [&[u8]; 5] = [b"a", b"bc", b"abcd", b"BEGIN", b"token"];
        let haystack = b"zabcd BEGIN token abcdbc";
        let ac = classic_ac_compile(&patterns);
        let lengths = pattern_lengths(&patterns);
        let mut expected = classic_ac_bounded_ranges_scan(&ac, &lengths, haystack);
        expected.sort_unstable();
        let program = build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
            &ac.dfa,
            patterns.len() as u32,
            128,
            false,
        );
        let mut inputs = ac_ranges_inputs(&ac.dfa, haystack, &lengths);
        inputs.push(u32_input(&[0]));
        inputs.push(u32_input(&classic_ac_candidate_end_byte_mask_words(
            &ac.dfa,
        )));
        let outputs = vyre_reference::reference_eval(&program, &inputs).expect(
            "Fix: prefiltered AC bounded-ranges program should evaluate in reference backend.",
        );
        let mut actual = decode_match_triples(&outputs);
        actual.sort_unstable();

        assert_eq!(actual, expected);
    }

    /// Behavioral regression guard: the infallible prefilter builder must wire the
    /// REAL DFA (delegating to the `try_` Ok program), never a degenerate empty-mask
    /// program (state_count=1, output_records_len=0) that suppresses every candidate.
    #[test]
    fn infallible_prefilter_uses_real_dfa_not_empty_fallback() {
        let ac = classic_ac_compile(&[b"abc", b"de", b"abcd"]);
        let via_infallible = build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
            &ac.dfa, 3, 128, false,
        );
        let via_try = try_build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
            &ac.dfa, 3, 128, false,
        )
        .expect("valid DFA must build");
        assert_infallible_matches_try(
            "bounded-ranges prefilter",
            &via_infallible,
            &via_try,
            &ac.dfa,
        );
    }

    #[test]
    fn bounded_ranges_prefilter_program_has_compact_stable_shape() {
        let ac = classic_ac_compile(&[b"Authorization: Bearer ", b"token", b"tok"]);
        let program = build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
            &ac.dfa, 3, 1024, false,
        );

        assert_eq!(program.workgroup_size(), [128, 1, 1]);
        assert_eq!(program.buffers().len(), 9);
        assert_eq!(program.buffers()[6].name(), "match_count");
        assert_eq!(program.buffers()[6].count, 1);
        assert_eq!(program.buffers()[7].name(), "candidate_end_mask");
        assert_eq!(program.buffers()[7].count, 8);
        assert_eq!(program.buffers()[8].name(), "matches");
        assert_eq!(program.buffers()[8].count, 1024 * 3);
    }

    /// The mask prefix property the one-array gate depends on: a narrower width
    /// binds exactly the leading masks of a wider one, at the same bindings and
    /// the same word counts. If it ever stopped holding, `PrefilterGate::decls`
    /// would silently size a mask against the wrong stage.
    #[test]
    fn narrower_gate_binds_the_leading_masks_of_a_wider_one() {
        let [.., end_mask, suffix2_mask, suffix3_bloom, _] = RANGES_BUFFER_NAMES;
        let wide = PrefilterGate::suffix3(end_mask, suffix2_mask, suffix3_bloom).decls();
        let narrow = PrefilterGate::end_byte(end_mask).decls();
        let unfiltered = PrefilterGate::unfiltered().decls();

        assert!(unfiltered.is_empty());
        assert_eq!(narrow.len(), 1);
        assert_eq!(wide.len(), 3);
        for (mask, (narrow_decl, wide_decl)) in narrow.iter().zip(&wide).enumerate() {
            assert_eq!(
                (narrow_decl.name(), narrow_decl.count),
                (wide_decl.name(), wide_decl.count),
                "gate mask {mask} must be identical across widths"
            );
        }
    }
}
