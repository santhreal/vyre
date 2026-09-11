//! Brute-force substring search  -  each invocation checks whether the
//! needle matches at its starting byte offset, writes `1` to the
//! match bitmap at that offset on hit.
//!
//! Category A composition. Sufficient for short needles; long
//! needles should compile to a DFA via the future `dfa_compile`
//! function and use that as a prefilter.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Canonical scan op id.
pub const SCAN_SUBSTRING_OP_ID: &str = "vyre-libs::pattern::substring_search";

/// Build a Program that writes `1` to `matches[i]` when `haystack[i..]`
/// starts with `needle`, else `0`. Both buffers are u32 byte arrays
/// packed one byte per u32 for simplicity (a future packed-u8 version
/// is Category A over `DataType::U8`).
#[must_use]
pub fn substring_search(
    haystack: &str,
    needle: &str,
    matches: &str,
    haystack_len: u32,
    needle_len: u32,
) -> Program {
    build_substring_program(haystack, needle, matches, haystack_len, needle_len)
}

fn build_substring_program(
    haystack: &str,
    needle: &str,
    matches: &str,
    haystack_len: u32,
    needle_len: u32,
) -> Program {
    let counted_storage = |name: &str, binding, count| {
        let decl = BufferDecl::storage(name, binding, BufferAccess::ReadOnly, DataType::U32);
        if count == 0 {
            decl
        } else {
            decl.with_count(count)
        }
    };
    let output_count = haystack_len.max(1);
    let visible_output_bytes = u64::from(haystack_len) * 4;
    let output = BufferDecl::output(matches, 2, DataType::U32)
        .with_count(output_count)
        .with_output_byte_range(0..visible_output_bytes);

    let i = Expr::var("i");
    // ok accumulates AND of per-byte equality checks. Start at 1; each
    // byte mismatch AND-s in 0 and latches the match bit off.
    let mut check_body: Vec<Node> = vec![Node::let_bind("ok", Expr::u32(1))];
    // Walk the needle one byte at a time. bytes are packed u32/byte for
    // simplicity  -  a packed-u8 variant is Category A over DataType::U8.
    check_body.push(Node::loop_for(
        "k",
        Expr::u32(0),
        Expr::u32(needle_len),
        vec![Node::assign(
            "ok",
            Expr::bitand(
                Expr::var("ok"),
                // Select turns the bool comparison into u32 {0,1} so
                // the accumulator stays in integer arithmetic.
                Expr::select(
                    Expr::eq(
                        Expr::load(haystack, Expr::add(i.clone(), Expr::var("k"))),
                        Expr::load(needle, Expr::var("k")),
                    ),
                    Expr::u32(1),
                    Expr::u32(0),
                ),
            ),
        )],
    ));
    check_body.push(Node::Store {
        buffer: matches.into(),
        index: i.clone(),
        value: Expr::var("ok"),
    });

    // The guard admits a start offset only when it indexes the match bitmap
    // and leaves room for the whole needle:
    //
    //   i < haystack_len  ∧  needle_len <= haystack_len
    //                     ∧  i <= haystack_len - needle_len
    //
    // The bitmap holds one slot per haystack byte. An empty needle also
    // matches at offset `haystack_len`, one past the last slot, so the first
    // conjunct bounds the store. The second keeps the subtraction in the
    // third from underflowing when a compile-time needle is longer than the
    // runtime haystack. `i + needle_len <= haystack_len` is not used: that
    // add wraps near `u32::MAX` and admits the last few offsets.
    let body = vec![
        Node::let_bind("i", Expr::LogicalIndex { axis: 0 }),
        Node::let_bind("haystack_len", Expr::buf_len(haystack)),
        Node::if_then(
            Expr::and(
                Expr::lt(i.clone(), Expr::var("haystack_len")),
                Expr::and(
                    Expr::le(Expr::u32(needle_len), Expr::var("haystack_len")),
                    Expr::le(
                        i,
                        Expr::sub(Expr::var("haystack_len"), Expr::u32(needle_len)),
                    ),
                ),
            ),
            check_body,
        ),
    ];
    Program::wrapped(
        vec![
            counted_storage(haystack, 0, haystack_len),
            counted_storage(needle, 1, needle_len),
            output,
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(SCAN_SUBSTRING_OP_ID, body)],
    )
}

const EXPECTED_SUBSTRING_MATCHES_BYTES: [u8; 32] = [
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        SCAN_SUBSTRING_OP_ID,
        || substring_search("haystack", "needle", "matches", 8, 3),
        Some(|| {
            let to_u32_vec = |s: &str| s.bytes().map(u32::from).collect::<Vec<_>>();
            vec![
                vec![
                    vyre_primitives::wire::pack_u32_slice(&to_u32_vec("abcabc++")),
                    vyre_primitives::wire::pack_u32_slice(&to_u32_vec("abc")),
                ],
                vec![
                    vyre_primitives::wire::pack_u32_slice(&to_u32_vec("xyzxyzxy")),
                    vyre_primitives::wire::pack_u32_slice(&to_u32_vec("xyz")),
                ]
            ]
        }),
        Some(|| {
            vec![
                vec![EXPECTED_SUBSTRING_MATCHES_BYTES.to_vec()],
                vec![EXPECTED_SUBSTRING_MATCHES_BYTES.to_vec()],
            ]
        }),
    )
    .with_category("scan")
    .with_uncharacterized()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_uses_canonical_scan_op_id() {
        let program = substring_search("haystack", "needle", "matches", 8, 3);
        let [Node::Region { generator, .. }] = program.entry() else {
            panic!("expected substring search to emit one scan region");
        };

        assert_eq!(generator.as_str(), SCAN_SUBSTRING_OP_ID);
    }
}
