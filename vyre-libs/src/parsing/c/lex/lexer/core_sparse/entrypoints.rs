use super::*;

/// Shared field set for the entry points below: every one lexes into the same
/// three token columns and differs only in haystack layout and which state
/// replays it performs.
fn spec<'a>(
    haystack: &'a str,
    out_tok_types: &'a str,
    out_tok_starts: &'a str,
    out_tok_lens: &'a str,
    out_counts: &'a str,
    haystack_len: u32,
    layout: SparseHaystackLayout,
) -> SparseLexerSpec<'a> {
    SparseLexerSpec {
        haystack,
        out_tok_types,
        out_tok_starts,
        out_tok_lens,
        out_counts,
        haystack_len,
        suppress_span_readback: false,
        emit_flags: true,
        layout,
        track_preproc_lines: true,
        track_literals: true,
        block_totals: None,
    }
}

pub fn c11_lexer_regular_sparse_packed_haystack_with_flags(
    haystack: &str,
    out_tok_types: &str,
    out_tok_starts: &str,
    out_tok_lens: &str,
    out_counts: &str,
    haystack_len: u32,
) -> Program {
    c11_lexer_regular_sparse_impl(&spec(
        haystack,
        out_tok_types,
        out_tok_starts,
        out_tok_lens,
        out_counts,
        haystack_len,
        SparseHaystackLayout::PackedU32,
    ))
}

pub fn c11_lexer_regular_sparse_u8_haystack_with_flags(
    haystack: &str,
    out_tok_types: &str,
    out_tok_starts: &str,
    out_tok_lens: &str,
    out_counts: &str,
    haystack_len: u32,
) -> Program {
    c11_lexer_regular_sparse_impl(&spec(
        haystack,
        out_tok_types,
        out_tok_starts,
        out_tok_lens,
        out_counts,
        haystack_len,
        SparseHaystackLayout::RawU8,
    ))
}

pub fn c11_lexer_regular_sparse_packed_haystack_with_flags_no_directives(
    haystack: &str,
    out_tok_types: &str,
    out_tok_starts: &str,
    out_tok_lens: &str,
    out_counts: &str,
    haystack_len: u32,
) -> Program {
    c11_lexer_regular_sparse_impl(&SparseLexerSpec {
        track_preproc_lines: false,
        ..spec(
            haystack,
            out_tok_types,
            out_tok_starts,
            out_tok_lens,
            out_counts,
            haystack_len,
            SparseHaystackLayout::PackedU32,
        )
    })
}

pub fn c11_lexer_regular_sparse_packed_haystack_with_flags_no_directives_no_backscan(
    haystack: &str,
    out_tok_types: &str,
    out_tok_starts: &str,
    out_tok_lens: &str,
    out_counts: &str,
    haystack_len: u32,
) -> Program {
    c11_lexer_regular_sparse_impl(&SparseLexerSpec {
        track_preproc_lines: false,
        track_literals: false,
        ..spec(
            haystack,
            out_tok_types,
            out_tok_starts,
            out_tok_lens,
            out_counts,
            haystack_len,
            SparseHaystackLayout::PackedU32,
        )
    })
}

pub fn c11_lexer_regular_sparse_no_directives_no_backscan(
    haystack: &str,
    out_tok_types: &str,
    out_tok_starts: &str,
    out_tok_lens: &str,
    out_counts: &str,
    haystack_len: u32,
) -> Program {
    c11_lexer_regular_sparse_impl(&SparseLexerSpec {
        suppress_span_readback: true,
        emit_flags: false,
        track_preproc_lines: false,
        track_literals: false,
        ..spec(
            haystack,
            out_tok_types,
            out_tok_starts,
            out_tok_lens,
            out_counts,
            haystack_len,
            SparseHaystackLayout::ExpandedU32,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::DataType;

    #[test]
    fn u8_haystack_entrypoint_declares_runtime_sized_raw_byte_source() {
        let program = c11_lexer_regular_sparse_u8_haystack_with_flags(
            "haystack", "types", "starts", "lens", "flags", 17,
        );
        let haystack = program
            .buffers()
            .iter()
            .find(|buffer| buffer.name() == "haystack")
            .expect("sparse lexer must declare haystack input");

        assert_eq!(haystack.element(), DataType::U8);
        assert_eq!(haystack.count(), 0);
    }

    #[test]
    fn packed_haystack_entrypoint_keeps_u32_word_source() {
        let program = c11_lexer_regular_sparse_packed_haystack_with_flags(
            "haystack", "types", "starts", "lens", "flags", 17,
        );
        let haystack = program
            .buffers()
            .iter()
            .find(|buffer| buffer.name() == "haystack")
            .expect("sparse lexer must declare haystack input");

        assert_eq!(haystack.element(), DataType::U32);
        assert_eq!(haystack.count(), 5);
    }

    #[test]
    fn expanded_haystack_entrypoint_keeps_u32_per_byte_source() {
        let program = c11_lexer_regular_sparse_no_directives_no_backscan(
            "haystack", "types", "starts", "lens", "count", 17,
        );
        let haystack = program
            .buffers()
            .iter()
            .find(|buffer| buffer.name() == "haystack")
            .expect("sparse lexer must declare haystack input");

        assert_eq!(haystack.element(), DataType::U32);
        assert_eq!(haystack.count(), 17);
    }
}
