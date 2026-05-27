//! Source contracts for C GPU-preprocess macro-table staging allocation.

use std::fs;
use std::path::PathBuf;

fn src_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn read_src(relative: &str) -> String {
    fs::read_to_string(src_path(relative)).unwrap_or_else(|err| {
        panic!("failed to read {relative}: {err}");
    })
}

#[test]
fn macro_table_staging_uses_checked_fallible_allocation_paths() {
    let macro_table = read_src("src/parsing/c/preprocess/gpu_pipeline/macro_table.rs");
    assert!(
        macro_table.contains("fn reserve_macro_table_vec<T>(")
            && macro_table.contains("fn filled_macro_table_vec<T: Clone>(")
            && macro_table.contains("fn padded_macro_table_byte_len("),
        "macro table packing must centralize fallible reservation and checked padding"
    );
    assert!(
        macro_table.contains("seen_names.try_reserve(macros.len())")
            && macro_table.contains("indexes.try_reserve(params.len())"),
        "macro table dedupe and parameter indexes must reserve fallibly"
    );
    assert!(
        macro_table.contains("let total_name_bytes = macros.iter().try_fold"),
        "macro table name staging must precompute byte totals with overflow checks"
    );
    assert!(
        macro_table.contains("let name_word_slots = macros")
            && macro_table.contains("let replacement_word_slots = macros")
            && macro_table.contains("reserve_macro_table_vec(\n            &mut pending,"),
        "macro expansion table packing must precompute and reserve expansion table storage"
    );
    assert!(
        !macro_table.contains("seen_names.reserve(macros.len())")
            && !macro_table.contains("Vec::with_capacity(macros.len() + 1)")
            && !macro_table.contains("Vec::with_capacity(macros.len())")
            && !macro_table.contains("vec![EMPTY_MACRO_SLOT; slots]")
            && !macro_table.contains("vec![false; slots]")
            && !macro_table.contains("let mut names_padded = vec![0u8; names_target]")
            && !macro_table.contains("indexes.reserve(params.len())"),
        "macro table packing must not use unchecked reserve or padded allocation paths"
    );
}
