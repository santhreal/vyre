# vyre-libs::scan SKILL

Byte/text scan primitives: substring search, DFA / Aho-Corasick, NFA /
regex pipelines. One ingredient inside larger vyre programs.
DFA compilation produces a transition table as a u32 buffer; the runtime
Program walks the table one byte per step.

## Coverage (shipped)

- `substring_search` (`matching-substring`): single-pattern brute-force
  match, one invocation per haystack offset.
- `aho_corasick` / `cooperative_dfa_scan` / `dfa_compile` /
  `dfa_compile_with_budget` (`matching-dfa`): multi-pattern scanners and
  CPU-side Aho-Corasick table builders with size-budget enforcement.
- `ScanProgram` and NFA tables (`matching-nfa`).
- `compile_regex_set` / `regex_compile` / `RegexDfaPipeline` and related
  admission helpers (`matching-regex`, often combined with DFA features).
- Hit packing, post-process, and fused-region evidence helpers on the
  always-on scan surface (unconditional `pub use` re-exports in `mod.rs`).

## Witness sources

- Substring search: simple corpus + edge cases (empty haystack,
  needle-larger-than-haystack, all-zeros, Unicode multi-byte). See
  `tests/cat_a_conform.rs` and `tests/aho_corasick_kat.rs`.
- Aho-Corasick: the 1975 paper's "ushers / he she his hers" example,
  hand-picked regression vectors, and the `aho-corasick` crate corpus.
- DFA budget: `tests` under the DFA compile path exercise
  `DfaCompileError::TooLarge`.

## Benchmark targets (criterion)

- Substring search 4 KiB haystack × 3-byte needle: ≤ 50 µs CPU ref;
  dispatch backends ≤ 10 µs on current high-end fleet hardware.
- Aho-Corasick with 100 patterns × 4 KiB haystack: ≤ 1 ms CPU ref;
  dispatch backends ≤ 50 µs.

## DFA size contract

`dfa_compile` panics when the default 16 MiB budget is exceeded.
Structured-error callers use `dfa_compile_with_budget` and match on
`DfaCompileError::TooLarge`. See `tests/cat_a_conform.rs` for the
budget witness corpus.

## Overflow contract

The substring-search length guard (`needle_len <= haystack_len` and
`i + needle_len <= haystack_len`) is overflow-safe; see
`tests/cat_a_conform.rs` edge cases for needle-larger-than-haystack.
