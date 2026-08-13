//! Cross-entry-point IR identity guard for the C11 lexer builders.
//!
//! Every builder below shares one parameterized token-classification walk
//! (`parsing::c::lex::lexer::core::helpers`). A refactor of that shared walk is
//! only a rehome if each public entry point still emits byte-identical IR, so
//! this pins the canonical wire fingerprint of every entry point across a set of
//! haystack lengths that exercise the packed / expanded / raw-u8 / contiguous
//! haystack layouts and their buffer-count arithmetic.
//!
//! The fingerprints were recorded against the pre-merge tree. A change here is
//! never a test bug: it means a builder's generated IR moved, and the diff must
//! be justified before the constant is touched.
//!
//! Mutation gate: perturbing any shared classifier stage (a token constant, a
//! scan bound, a byte accessor, a variable name) changes the fingerprint of
//! every entry point that composes that stage, so this test goes red.

#![cfg(feature = "c-parser")]

use vyre::ir::Program;
use vyre_libs::parsing::c::lex::lexer::{
    c11_lex_regular_single_pass, c11_lex_single_pass, c11_lexer, c11_lexer_regular,
    c11_lexer_regular_ranked, c11_lexer_regular_sparse,
    c11_lexer_regular_sparse_no_directives_no_backscan,
    c11_lexer_regular_sparse_packed_haystack_with_block_totals,
    c11_lexer_regular_sparse_packed_haystack_with_flags,
    c11_lexer_regular_sparse_packed_haystack_with_flags_no_directives,
    c11_lexer_regular_sparse_packed_haystack_with_flags_no_directives_no_backscan,
    c11_lexer_regular_sparse_u8_haystack_with_flags,
};

/// Lengths chosen to hit the layout arithmetic corners: sub-word, unaligned,
/// prime, and a full workgroup-sized run.
const LENGTHS: [u32; 4] = [1, 7, 17, 64];

fn hex(program: &Program) -> String {
    program
        .fingerprint()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

type Builder = fn(u32) -> Program;

fn builders() -> Vec<(&'static str, Builder)> {
    vec![
        ("c11_lexer", |n| {
            c11_lexer("haystack", "types", "starts", "lens", "counts", n)
        }),
        ("c11_lexer_regular", |n| {
            c11_lexer_regular("haystack", "types", "starts", "lens", "counts", n)
        }),
        ("c11_lexer_regular_ranked", |n| {
            c11_lexer_regular_ranked("haystack", "types", "starts", "lens", "counts", n)
        }),
        ("c11_lexer_regular_sparse", |n| {
            c11_lexer_regular_sparse("haystack", "types", "starts", "lens", "counts", n)
        }),
        ("sparse_packed_with_flags", |n| {
            c11_lexer_regular_sparse_packed_haystack_with_flags(
                "haystack", "types", "starts", "lens", "counts", n,
            )
        }),
        ("sparse_u8_with_flags", |n| {
            c11_lexer_regular_sparse_u8_haystack_with_flags(
                "haystack", "types", "starts", "lens", "counts", n,
            )
        }),
        ("sparse_packed_no_directives", |n| {
            c11_lexer_regular_sparse_packed_haystack_with_flags_no_directives(
                "haystack", "types", "starts", "lens", "counts", n,
            )
        }),
        ("sparse_packed_no_directives_no_backscan", |n| {
            c11_lexer_regular_sparse_packed_haystack_with_flags_no_directives_no_backscan(
                "haystack", "types", "starts", "lens", "counts", n,
            )
        }),
        ("sparse_no_directives_no_backscan", |n| {
            c11_lexer_regular_sparse_no_directives_no_backscan(
                "haystack", "types", "starts", "lens", "counts", n,
            )
        }),
        ("sparse_packed_with_block_totals", |n| {
            c11_lexer_regular_sparse_packed_haystack_with_block_totals(
                "haystack", "types", "starts", "lens", "scratch", "block_totals", n,
            )
        }),
        ("c11_lex_single_pass", |n| {
            c11_lex_single_pass("haystack", "types", "starts", "lens", "counts", n, 8)
        }),
        ("c11_lex_regular_single_pass", |n| {
            c11_lex_regular_single_pass("haystack", "types", "starts", "lens", "counts", n, 8)
        }),
    ]
}

/// `(entry point, haystack_len, canonical wire fingerprint)`.
const GOLDEN: &[(&str, u32, &str)] = &[
    ("c11_lexer", 1, "581fce4870bfd03fa5d1e3930d54dd683baa0734905691ee7583bef4836374b5"),
    ("c11_lexer", 7, "4d2db8e3e2e5dcc7d553005c9ab4ed388e97b733b5267586d14ec981a9bb35ce"),
    ("c11_lexer", 17, "d14f027d8991538eedee90e91158b61f3510253ce160ad5bbf4a5d43ee6f4fdd"),
    ("c11_lexer", 64, "2e09261a4535dc8fcdb09b164d4b6c187f94c6a9c28eab2014bb7af9fb31a568"),
    ("c11_lexer_regular", 1, "0ddeb05ab5e1d58c98d334068c12c529fc0eea73422c1908ed034fcf71f6dc54"),
    ("c11_lexer_regular", 7, "af45e76406097ed84e9467182367b32a8c4c6b1667d715bb59b9c12dd66531de"),
    ("c11_lexer_regular", 17, "a266260824d3195b9d1bea8b8da9e1aec9e06c94013e7bcc0767b584b315c696"),
    ("c11_lexer_regular", 64, "29316bf529c99053f80364a94f100b7771b40d0633c1dc6e56fbde0eda9baf27"),
    ("c11_lexer_regular_ranked", 1, "db8d3b2e33402f7ed59856dec71e16e75c1090bec52aec901faebf88b732273e"),
    ("c11_lexer_regular_ranked", 7, "6f701fb483a0ec803ce0aa79e44c1ae2c3fa6dfe277a52a4309ab8e01cdabf1d"),
    ("c11_lexer_regular_ranked", 17, "917755f5229a6a3e0889fa6ca5e2742daa262139eae295aa24bee04f241a4212"),
    ("c11_lexer_regular_ranked", 64, "99c448eb436ff5468746ae44102c390f45817a40ed61230d34196aed45677ca0"),
    ("c11_lexer_regular_sparse", 1, "36e0e4f58b58f2cc510c2b0eaf0f99db48befc2436385ae15a34e9c38a65b25d"),
    ("c11_lexer_regular_sparse", 7, "810a40eb0f1d17f422ebeaa90781e6a27b13df236cf618da882db3e61295b8be"),
    ("c11_lexer_regular_sparse", 17, "689616193a1169470f2cacb5a8888fabc7952c23a54591009694ceab7dffd185"),
    ("c11_lexer_regular_sparse", 64, "a153e4336120739e029671f8ce0477e7fd5e9f178c8233feb8a41ef59b92458d"),
    ("sparse_packed_with_flags", 1, "a48a35d21767dab4b0d68f91032aca1eef0c509af8374eeb37e05b48095afed4"),
    ("sparse_packed_with_flags", 7, "215be2c28ff0d6034d35e060689f9a1c8d31f9c77fb07583a4d16d706595721a"),
    ("sparse_packed_with_flags", 17, "b188dc7297a1758f02aae7a1e6cfb6717b0be0f1d2bde4c21cdfadd687fcf2f6"),
    ("sparse_packed_with_flags", 64, "6761e928f97f185bb546fe3efc2b3b7a9c17fe5178f9ab6325dab534234537d2"),
    ("sparse_u8_with_flags", 1, "1ca44f1e778ae2a3898033eef9323c4841feb8be9245a39c8df5b30c869820f8"),
    ("sparse_u8_with_flags", 7, "07c8c6aa18ec2e968cc7c123f04f6928823c3ba8a4e62a53fe97f6be729194f8"),
    ("sparse_u8_with_flags", 17, "cbe0fc455ac9eba35113ec5772a8c74106d1ec3c493dcf7142b7ababe321b0ef"),
    ("sparse_u8_with_flags", 64, "c41fcc638283eefe7fd411b259c8dc7475d9f52abafcac43c73f311cca13abb3"),
    ("sparse_packed_no_directives", 1, "6a0a0cc051191c9753afef316f96d91cb1281ab50ba8a8ab906758b8dbb15508"),
    ("sparse_packed_no_directives", 7, "fc78d840d399e58dca43febb3e8bb504d4119879b57bd72bab5ac554893d7d96"),
    ("sparse_packed_no_directives", 17, "ab83e5c2a12c7fb2664e4e617a203023d07ed7aa681f94fd47b703fd1e861d6d"),
    ("sparse_packed_no_directives", 64, "2cafed97080f8b9fbed32c2bb65436c325ae75893794f332856d166cf7b7c063"),
    ("sparse_packed_no_directives_no_backscan", 1, "7d668ff2bbd2639b8759bef28ef910c5fca492f5bc1c28699ef5fec339f49a4f"),
    ("sparse_packed_no_directives_no_backscan", 7, "e5f44de8d04588a4b10c05e3f44854af127f950afd68c174dc3d1fc571470e06"),
    ("sparse_packed_no_directives_no_backscan", 17, "a9a2d0d1ef9eed6ee9d192260cd164e834151dd4a0fa94c14ee2cd2b9abaa823"),
    ("sparse_packed_no_directives_no_backscan", 64, "079bdc713523eb890f24748b10a188b36955bd1d68ba13369a2cc808f76f6d17"),
    ("sparse_no_directives_no_backscan", 1, "50acdd023fd7509fd2ccd969c53d9ca0b2607d4100941f7841a730ad8e919fc3"),
    ("sparse_no_directives_no_backscan", 7, "aee6e0d545eb971fa19959d7ee725fe08400bd4611ac709c73f54b8229abacac"),
    ("sparse_no_directives_no_backscan", 17, "e6d3a41f3dc8402b47f24956c6eef6c930dfb2ae27b940d7d879ec62bad87f64"),
    ("sparse_no_directives_no_backscan", 64, "9ac8b3b9956865d30819f26eb94f3edc77b32e7ed48cdd560d20d644dfeeee49"),
    ("sparse_packed_with_block_totals", 1, "b02126e2776152b8bb2f063abd5a5b2e5c4d0645c8006e9ac54516a6de6290b8"),
    ("sparse_packed_with_block_totals", 7, "8abfafd5916f16e3a81df06af719f53619293295e98ea8af50b8b5d02fd4d744"),
    ("sparse_packed_with_block_totals", 17, "dfab4f889a8d852629d2c9d08aabdce9a1ef5f6957b231235f4e0415322a86fa"),
    ("sparse_packed_with_block_totals", 64, "ace52656d739083806d7f443dcc9662eca3e41108fac828113321d89a322d9c7"),
    ("c11_lex_single_pass", 1, "2f7b7fba202b23ead19fde11542a02bf27c922570b8ee4a656836c8230fd1d06"),
    ("c11_lex_single_pass", 7, "25f52f5a896e226f27f5f7708b8c05126ecd19f71fa4f7fa7f66fc9a62357c42"),
    ("c11_lex_single_pass", 17, "c8bc40368f1fd33a706a1d864eaeec0341965b8add05ca303c74c3cfa159e36e"),
    ("c11_lex_single_pass", 64, "646ab9dbd3920eac9f1f38ea818dcf21dfb29cb64041d3005bbb6c4ef316eca8"),
    ("c11_lex_regular_single_pass", 1, "3c8139cf61d66b2077a47f8dc66f8648128bdffabadeef9d456e297455ac264a"),
    ("c11_lex_regular_single_pass", 7, "e6a2d4cbfe4bb67349f081712c9baeef5e78a0a2bdb7ea474069841446d113ef"),
    ("c11_lex_regular_single_pass", 17, "9a516e86e83bfd966883f386a55fe504e72dbccf123d05e8dbb47ef6f4b05c3f"),
    ("c11_lex_regular_single_pass", 64, "2b92fc3e92f1cffdbdd1eb12c0e6ce14fae721b468f5d975a1f0f374d5683d16"),
];

#[test]
fn every_lexer_entry_point_keeps_its_recorded_ir_fingerprint() {
    let mut observed = Vec::new();
    for (name, build) in builders() {
        for len in LENGTHS {
            observed.push((name, len, hex(&build(len))));
        }
    }

    let expected: Vec<(&str, u32, String)> = GOLDEN
        .iter()
        .map(|(name, len, digest)| (*name, *len, (*digest).to_string()))
        .collect();
    assert_eq!(
        observed, expected,
        "lexer IR fingerprints moved; the shared classification walk changed generated IR"
    );
}

/// The shared walk must not have collapsed distinct builders onto one another:
/// a parameterization bug that made every entry point emit the same IR would
/// still satisfy the golden test if the goldens were re-recorded, so pin the
/// distinctness separately.
#[test]
fn lexer_entry_points_stay_distinct_programs() {
    let len = 64;
    let mut digests: Vec<(&str, String)> = builders()
        .into_iter()
        .map(|(name, build)| (name, hex(&build(len))))
        .collect();
    let total = digests.len();
    digests.sort_by(|a, b| a.1.cmp(&b.1));
    digests.dedup_by(|a, b| a.1 == b.1);
    assert_eq!(
        digests.len(),
        total,
        "distinct lexer entry points must emit distinct IR; collapsed set: {digests:?}"
    );
}
