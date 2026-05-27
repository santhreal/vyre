# IGNORE_AUDIT_2026-04-22

Scope: `libs/performance/matching/vyre/`, `libs/tools/surgec/`, `libs/surge/`

## Session start inventory

| File | Line | Test name | Ignore reason |
|---|---|---|---|
| `surgec-grammar-gen/tests/gen_lex_hash.rs` | 9 | `print_hello_max_munch_blake3` | `run locally to refresh goldens/hello_max_munch_kinds.blake3` |

## Final state

| File | Line | Test name | Final state |
|---|---|---|---|
| `surgec-grammar-gen/tests/gen_lex_hash.rs` | 8 | `hello_max_munch_blake3` | **blocker-implemented-and-lifted** |

### Blocker detail  -  `hello_max_munch_blake3`

**Blocker:** The test was a stale one-off helper that referenced the removed `max_munch_cpu::lex_max_munch_kinds` API, had zero assertions, and was ignored because it would have failed at runtime.

**Fix applied:**
1. Removed `#[ignore]`.
2. Updated imports to the current `lex_c11_max_munch_kinds` + `kinds_blake3` surface.
3. Renamed test from `print_hello_max_munch_blake3` to `hello_max_munch_blake3` to reflect it is now a regression test, not a printer.
4. Added golden assertions for the blake3 hash and the token-kind vector, turning the helper into a real regression test.

**Verification:** `cargo test -p surgec-grammar-gen --test gen_lex_hash hello_max_munch_blake3` passes.

## Escalations

None.

## Session-end summary

- **Total ignores at start:** 1
- **Lifted-and-passing:** 0
- **Blocker-implemented-and-lifted:** 1
- **Escalated:** 0
- **Remaining ignores:** 0
