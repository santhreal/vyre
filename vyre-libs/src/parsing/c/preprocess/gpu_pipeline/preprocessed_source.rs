//! The preprocessor driver's output value and the macro table it carries.

use super::{
    ConditionalEvent, HeaderReuseEvent, IncludeAccelerationEvent, IncludeByteCacheStats,
    IncludeEvent, MacroEvent, MacroExpansionEvent, TokenProvenanceEvent,
};

/// Output of the driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreprocessedSource {
    /// Concatenated active bytes  -  line-spliced, comment-stripped,
    /// conditional-masked, include-expanded. Macro expansion is
    /// deliberately NOT performed here (mirrors the v0.4
    /// `prepare_resident_translation_unit_source` contract).
    pub bytes: Vec<u8>,
    /// Macros accumulated during the walk (CLI macros + every
    /// `#define` in active branches). Downstream macro-expansion
    /// kernels consume this.
    pub macros: Vec<MacroDef>,
    /// Include graph events whose requests were extracted by GPU directive
    /// payload kernels. Resolution/read remains host filesystem metadata.
    pub include_events: Vec<IncludeEvent>,
    /// Per-run include byte-cache counters for loader avoidance evidence.
    pub include_byte_cache_stats: IncludeByteCacheStats,
    /// Conditional stack events whose directive payloads are GPU-derived.
    pub conditional_events: Vec<ConditionalEvent>,
    /// Macro definition-table events whose directive payloads are GPU-derived.
    pub macro_events: Vec<MacroEvent>,
    /// Macro expansion origin events.
    pub macro_expansion_events: Vec<MacroExpansionEvent>,
    /// Token-level spelling and expansion provenance for the preprocessed output.
    pub token_provenance_events: Vec<TokenProvenanceEvent>,
    /// Include guard / pragma-once acceleration evidence.
    pub include_acceleration_events: Vec<IncludeAccelerationEvent>,
    /// Header-analysis cache reuse evidence.
    pub header_reuse_events: Vec<HeaderReuseEvent>,
}

/// A `#define`'d macro encountered during preprocessing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroDef {
    /// Macro identifier bytes.
    pub name: Vec<u8>,
    /// Comma-separated argument-list bytes for function-like macros;
    /// empty for object-like.
    pub args: Vec<u8>,
    /// Replacement body bytes.
    pub body: Vec<u8>,
    /// `true` for function-like (`#define M(a) …`).
    pub is_function_like: bool,
}
