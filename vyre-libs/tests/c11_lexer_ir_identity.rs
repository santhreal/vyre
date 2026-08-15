//! Cross-entry-point IR identity guard for the C11 lexer builders.
//!
//! Every builder below shares one parameterized token-classification walk
//! (`parsing::c::lex::lexer::classify::stages`). A refactor of that shared walk is
//! only a rehome if each public entry point still emits byte-identical IR, so
//! this pins the canonical wire fingerprint of every entry point across a set of
//! haystack lengths that exercise the packed / expanded / raw-u8 / contiguous
//! haystack layouts and their buffer-count arithmetic.
//!
//! A change here is never a test bug: it means a builder's generated IR moved,
//! and the diff must be justified before the constant is touched. The current
//! values were recorded when the seven classifier phase generators were renamed
//! to the `anonymous::` prefix, because a phase boundary inside one operation is
//! not an operation id and every audit that reads a generator as one was being
//! told otherwise.
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
                "haystack",
                "types",
                "starts",
                "lens",
                "scratch",
                "block_totals",
                n,
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
    (
        "c11_lexer",
        1,
        "c32b99406f85114f8dda62ec3e5af4d1d324a844c79d799bbdfbf5301c62f62b",
    ),
    (
        "c11_lexer",
        7,
        "0fb740af23b5ae14ba0eafbd0a7c82973b64cf246a4b3ad0737558f57ee36284",
    ),
    (
        "c11_lexer",
        17,
        "ee71072d6a3f3fc1da8fc17d27e8d6bc23496754555e4f78a834d648096eaaa5",
    ),
    (
        "c11_lexer",
        64,
        "78029aa21844168b19e68f2b71689b35f1b7e0b066c7d1a68e5af8a48127d5d5",
    ),
    (
        "c11_lexer_regular",
        1,
        "725bcd2c807e061faddbf780500229c35b14dfc404fd516f029f1413d4a16205",
    ),
    (
        "c11_lexer_regular",
        7,
        "643fd69f39e5602e1f2fb7215d2a25372385a046f750cac62fb5bad0a4cdc4bd",
    ),
    (
        "c11_lexer_regular",
        17,
        "23f4180808182f9b4e54260eb3b910b141d3bf9b3a8ae841dbc18e27ea3e71c6",
    ),
    (
        "c11_lexer_regular",
        64,
        "d4a7b7b3a06388a619145304bdc0e03652e77047256fa376432565c6e63aedcc",
    ),
    (
        "c11_lexer_regular_ranked",
        1,
        "db8d3b2e33402f7ed59856dec71e16e75c1090bec52aec901faebf88b732273e",
    ),
    (
        "c11_lexer_regular_ranked",
        7,
        "6f701fb483a0ec803ce0aa79e44c1ae2c3fa6dfe277a52a4309ab8e01cdabf1d",
    ),
    (
        "c11_lexer_regular_ranked",
        17,
        "917755f5229a6a3e0889fa6ca5e2742daa262139eae295aa24bee04f241a4212",
    ),
    (
        "c11_lexer_regular_ranked",
        64,
        "99c448eb436ff5468746ae44102c390f45817a40ed61230d34196aed45677ca0",
    ),
    (
        "c11_lexer_regular_sparse",
        1,
        "36e0e4f58b58f2cc510c2b0eaf0f99db48befc2436385ae15a34e9c38a65b25d",
    ),
    (
        "c11_lexer_regular_sparse",
        7,
        "810a40eb0f1d17f422ebeaa90781e6a27b13df236cf618da882db3e61295b8be",
    ),
    (
        "c11_lexer_regular_sparse",
        17,
        "689616193a1169470f2cacb5a8888fabc7952c23a54591009694ceab7dffd185",
    ),
    (
        "c11_lexer_regular_sparse",
        64,
        "a153e4336120739e029671f8ce0477e7fd5e9f178c8233feb8a41ef59b92458d",
    ),
    (
        "sparse_packed_with_flags",
        1,
        "a48a35d21767dab4b0d68f91032aca1eef0c509af8374eeb37e05b48095afed4",
    ),
    (
        "sparse_packed_with_flags",
        7,
        "215be2c28ff0d6034d35e060689f9a1c8d31f9c77fb07583a4d16d706595721a",
    ),
    (
        "sparse_packed_with_flags",
        17,
        "b188dc7297a1758f02aae7a1e6cfb6717b0be0f1d2bde4c21cdfadd687fcf2f6",
    ),
    (
        "sparse_packed_with_flags",
        64,
        "6761e928f97f185bb546fe3efc2b3b7a9c17fe5178f9ab6325dab534234537d2",
    ),
    (
        "sparse_u8_with_flags",
        1,
        "1ca44f1e778ae2a3898033eef9323c4841feb8be9245a39c8df5b30c869820f8",
    ),
    (
        "sparse_u8_with_flags",
        7,
        "07c8c6aa18ec2e968cc7c123f04f6928823c3ba8a4e62a53fe97f6be729194f8",
    ),
    (
        "sparse_u8_with_flags",
        17,
        "cbe0fc455ac9eba35113ec5772a8c74106d1ec3c493dcf7142b7ababe321b0ef",
    ),
    (
        "sparse_u8_with_flags",
        64,
        "c41fcc638283eefe7fd411b259c8dc7475d9f52abafcac43c73f311cca13abb3",
    ),
    (
        "sparse_packed_no_directives",
        1,
        "6a0a0cc051191c9753afef316f96d91cb1281ab50ba8a8ab906758b8dbb15508",
    ),
    (
        "sparse_packed_no_directives",
        7,
        "fc78d840d399e58dca43febb3e8bb504d4119879b57bd72bab5ac554893d7d96",
    ),
    (
        "sparse_packed_no_directives",
        17,
        "ab83e5c2a12c7fb2664e4e617a203023d07ed7aa681f94fd47b703fd1e861d6d",
    ),
    (
        "sparse_packed_no_directives",
        64,
        "2cafed97080f8b9fbed32c2bb65436c325ae75893794f332856d166cf7b7c063",
    ),
    (
        "sparse_packed_no_directives_no_backscan",
        1,
        "7d668ff2bbd2639b8759bef28ef910c5fca492f5bc1c28699ef5fec339f49a4f",
    ),
    (
        "sparse_packed_no_directives_no_backscan",
        7,
        "e5f44de8d04588a4b10c05e3f44854af127f950afd68c174dc3d1fc571470e06",
    ),
    (
        "sparse_packed_no_directives_no_backscan",
        17,
        "a9a2d0d1ef9eed6ee9d192260cd164e834151dd4a0fa94c14ee2cd2b9abaa823",
    ),
    (
        "sparse_packed_no_directives_no_backscan",
        64,
        "079bdc713523eb890f24748b10a188b36955bd1d68ba13369a2cc808f76f6d17",
    ),
    (
        "sparse_no_directives_no_backscan",
        1,
        "50acdd023fd7509fd2ccd969c53d9ca0b2607d4100941f7841a730ad8e919fc3",
    ),
    (
        "sparse_no_directives_no_backscan",
        7,
        "aee6e0d545eb971fa19959d7ee725fe08400bd4611ac709c73f54b8229abacac",
    ),
    (
        "sparse_no_directives_no_backscan",
        17,
        "e6d3a41f3dc8402b47f24956c6eef6c930dfb2ae27b940d7d879ec62bad87f64",
    ),
    (
        "sparse_no_directives_no_backscan",
        64,
        "9ac8b3b9956865d30819f26eb94f3edc77b32e7ed48cdd560d20d644dfeeee49",
    ),
    (
        "sparse_packed_with_block_totals",
        1,
        "b02126e2776152b8bb2f063abd5a5b2e5c4d0645c8006e9ac54516a6de6290b8",
    ),
    (
        "sparse_packed_with_block_totals",
        7,
        "8abfafd5916f16e3a81df06af719f53619293295e98ea8af50b8b5d02fd4d744",
    ),
    (
        "sparse_packed_with_block_totals",
        17,
        "dfab4f889a8d852629d2c9d08aabdce9a1ef5f6957b231235f4e0415322a86fa",
    ),
    (
        "sparse_packed_with_block_totals",
        64,
        "ace52656d739083806d7f443dcc9662eca3e41108fac828113321d89a322d9c7",
    ),
    (
        "c11_lex_single_pass",
        1,
        "4675e39ec0ff48abb8bf1a668c0151b43beb3e07052a0d5368522a656eb56af0",
    ),
    (
        "c11_lex_single_pass",
        7,
        "3e540faddeee3922364c002f8359fdd884cfa21e4a2605fefe9ca5346a5e47e6",
    ),
    (
        "c11_lex_single_pass",
        17,
        "de74137a4dccbf5ddf92f34844792cb2d7f616e848c6297bcbf24998d20041b2",
    ),
    (
        "c11_lex_single_pass",
        64,
        "7167e4d7c6676cd19cdade90694f33722f3078d1d05d05bf5a11ad0bb197a661",
    ),
    (
        "c11_lex_regular_single_pass",
        1,
        "2b868b142e70ac106d32b1bfd6cc7395814cd8e87ebd964d858c39f412b597f5",
    ),
    (
        "c11_lex_regular_single_pass",
        7,
        "71f19ad098c3fe6a765209a86d358c03d869b47bbf6669be9391f8fcacea7b4c",
    ),
    (
        "c11_lex_regular_single_pass",
        17,
        "9a3fee4b5349f1e9687b60abd2a50e987ac2e4e4a344f683715516dee0f35489",
    ),
    (
        "c11_lex_regular_single_pass",
        64,
        "de660c5cae18ae89c94bef2030bb3310ab94adcf920ddef6d8f00ac43fec9ee8",
    ),
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
