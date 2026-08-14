//! Shared C-frontend test harness, owned by the workspace rather than a crate.
//!
//! The C frontend's compute lives in `vyre-libs`, so its CPU contract tests live
//! in `vyre-libs/tests`; the parity tests that drive the same passes through a
//! backend live in that driver's `tests`. Both halves build the same fixtures
//! and read the same packed rows. Copying that setup per crate is how the six
//! `c_ast_sema_scope_*` files came to share seventy eight-line shingles each, so
//! it has one owner here and both crates include it with `#[path]`, the same way
//! `tests/support/preferred_dispatch_backend_contract.rs` is shared.
//!
//! A consumer imports only the submodules it needs. `rows` and `token_fixture`
//! are the common base; `expression_pipeline` and `scope_fixture` are the two
//! fixture layouts the C tests use, and they intentionally do not glob together
//! because each owns an `assert_pg_preserves_row` for its own row carrier.
//! `semantic_graph` reads the property graph's node and edge buffers, and
//! `reference_lexer` runs real source bytes through the lexer kernel for the
//! contracts that start from source rather than from a hand-built stream.
//!
//! `fixtures/` holds the per-family token streams that a family's CPU arm and
//! its backend parity arm both build. Those are not declared here: each root
//! that needs one includes it with `#[path]`, so a target compiles only the
//! fixtures it uses.
//!
//! `spelling` owns the compact token-stream spellings every fixture is written
//! in. A stream spelled one token per line reads as a copy of any other stream
//! that happens to share eight consecutive tokens, which is what the fixture
//! streams here were measured as.

pub(crate) mod expression_pipeline;
pub(crate) mod reference_lexer;
pub(crate) mod rows;
pub(crate) mod scope_fixture;
pub(crate) mod semantic_graph;
pub(crate) mod spelling;
pub(crate) mod token_fixture;
