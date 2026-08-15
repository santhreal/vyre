//! Token-level spelling/expansion provenance emitted by the GPU preprocessor.

use std::sync::{Mutex, OnceLock};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use smallvec::SmallVec;

use super::macro_events::{stable_macro_symbol_id, MacroEvent, MacroEventKind};
use super::tokenization::gpu_tokenize_without_directive_metadata;
use super::{ClassifiedTokens, MacroDef, ProgramOracle};

type MacroBucket<'a> = SmallVec<[&'a MacroDef; 2]>;

mod anchor_match;
mod checked;
mod direct;
mod invocation;
mod macro_record;
mod missing_invocation;
mod model;
mod object_backfill;
mod parameter_substitution;
mod params;
mod replacement_cache;
mod replacement_tokens;
mod span_dedupe;
mod spelling_origin;
mod token_columns;

pub(super) use direct::record_direct_token_provenance;
pub(super) use macro_record::record_macro_token_provenance;
pub use model::TokenProvenanceEvent;

use anchor_match::*;
use checked::*;
use invocation::*;
use missing_invocation::record_missing_invocation_provenance;
use model::{ReplacementTokenCacheKey, REPLACEMENT_TOKEN_CACHE_MAX_ENTRIES};
use object_backfill::record_missing_object_replacement_provenance;
use parameter_substitution::record_missing_parameter_substitution_provenance;
use params::*;
use replacement_cache::cached_replacement_tokens;
use replacement_tokens::*;
use span_dedupe::SpanDedupe;
use spelling_origin::macro_spelling_origin;
use token_columns::{token_len, token_start};

fn reserve_token_provenance_events(
    token_provenance_events: &mut Vec<TokenProvenanceEvent>,
    additional: usize,
    label: &'static str,
) -> Result<(), String> {
    token_provenance_events
        .try_reserve_exact(additional)
        .map_err(|error| {
            format!(
                "vyre-libs::gpu_pipeline: could not reserve {additional} token provenance events for {label}: {error:?}. Fix: shard preprocessing before provenance export."
            )
        })
}
