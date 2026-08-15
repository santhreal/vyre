//! GPU-resident preprocessor pipeline orchestration.
//!
//! Replaces the CPU translation-unit host helpers with a chain
//! of GPU dispatches. Host-side responsibilities are limited to:
//!
//! - File I/O initiation (`fs::read`)  -  the kernel-mode VFS work that
//!   has no GPU equivalent.
//! - Recursive include scheduling  -  graph-traversal bookkeeping over
//!   file paths after GPU directive classification emits the include
//!   requests.
//! - Macro / conditional-frame bookkeeping between dispatches. The
//!   directive parsing, conditional evaluation, and replacement-token
//!   work stay in GPU kernels; host state only carries compact metadata
//!   from one dispatch frontier to the next.
//!
//! All actual byte-level / token-level / expression-level computation
//! runs on GPU via the kernels in
//! `vyre_libs::parsing::c::preprocess::*`.
//!
//! ## Phase split (this module ships in chunks)
//!
//! - **18a (this commit):** `gpu_filter_source_bytes`  -  runs
//!   `line_splice_classify` + `comment_strip_mask` + element-wise AND
//!   + prefix-scan + scatter-compact to produce the post-phase-2,
//!   comment-free byte stream that the lexer consumes. Foundational
//!   brick that every later stage builds on.
//! - **18b:** Lex + directive-classify + ifdef/if value evaluation
//!   batch.
//! - **18c:** `#define` / `#include` row parsing + macro-table
//!   maintenance.
//! - **18d:** Recursive include graph driver + macro expansion.
//! - **18e:** production callers route through this GPU pipeline; host
//!   preprocessor code remains only as explicit reference/test infrastructure.

mod filter;
pub use filter::{gpu_filter_source_bytes, FilteredBytes};
mod buffers;
mod byte_lru_cache;
mod cache;
#[cfg(test)]
mod cache_tests;
mod classified_size;
#[cfg(test)]
mod conditional_eval;
mod conditional_events;
mod conditional_stack;
mod directives;
mod dispatch;
mod driver;
mod expansion_events;
mod header_reuse;
mod include_acceleration;
mod include_events;
mod include_loader;
mod live_conditional_cache;
mod live_state;
mod lru_index;
mod macro_events;
mod macro_expansion;
mod macro_table;
mod macro_values;
mod payload_size;
mod preprocessed_source;
mod scan;
mod segments;
mod source_spans;
mod token_provenance;
mod tokenization;
fn trim_ascii(bytes: &[u8]) -> &[u8] {
    let mut start = 0_usize;
    let mut end = bytes.len();
    while start < end && bytes[start].is_ascii_whitespace() {
        start += 1;
    }
    while end > start && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    &bytes[start..end]
}

fn classified_token_bytes_opt(classified: &ClassifiedTokens, idx: usize) -> Option<&[u8]> {
    let start = *classified.tok_starts.get(idx)? as usize;
    let len = *classified.tok_lens.get(idx)? as usize;
    classified.source.get(start..start.checked_add(len)?)
}

pub use buffers::bucket_pow2;
pub use conditional_events::{ConditionalEvent, ConditionalEventKind, ConditionalEventResidency};
pub use directives::{gpu_extract_directive_payloads, DirectivePayload};
pub use dispatch::ProgramOracle;
pub use expansion_events::MacroExpansionEvent;
pub use header_reuse::{HeaderReuseEvent, HeaderReuseKey};
pub use include_acceleration::{IncludeAccelerationEvent, IncludeAccelerationKind};
pub use include_events::{IncludeByteCacheStats, IncludeEvent, IncludeEventResidency};
pub use include_loader::{IncludeLoader, MAX_INCLUDE_DEPTH};
pub use macro_events::{MacroEvent, MacroEventKind};
pub use preprocessed_source::{MacroDef, PreprocessedSource};
pub use token_provenance::TokenProvenanceEvent;
pub use tokenization::{gpu_tokenize_and_classify, ClassifiedTokens};

/// Drive the GPU preprocessor over a translation unit and recursively expand
/// active includes through `loader`.
pub fn gpu_preprocess_translation_unit(
    dispatcher: &dyn ProgramOracle,
    loader: &dyn IncludeLoader,
    tu_path: &std::path::Path,
    source: &[u8],
    cli_macros: &[MacroDef],
) -> Result<PreprocessedSource, String> {
    driver::preprocess_translation_unit(dispatcher, loader, tu_path, source, cli_macros)
}
