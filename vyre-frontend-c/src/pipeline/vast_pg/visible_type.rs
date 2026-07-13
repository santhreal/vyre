use std::cell::RefCell;
use std::mem;

use super::*;

#[derive(Default)]
struct VisibleTypeScratch {
    outputs: Vec<Vec<u8>>,
}

thread_local! {
    static VISIBLE_TYPE_SCRATCH: RefCell<VisibleTypeScratch> =
        RefCell::new(VisibleTypeScratch::default());
}

/// Precompute the per-node visible-typedef-name table that the precomputed-context
/// annotation variant consumes.
///
/// The precomputed-context declaration-kind annotation path is haystack-free and
/// can only match builtin type KEYWORDS, so on its own it dropped the ordinary
/// declarator flag for `T x;` where `T` is a typedef-name. This stage resolves the
/// visible-typedef-name bit ONCE per node (reading the completed `decl_contexts`
/// table plus `vast_nodes`/`haystack`) so the annotate pass just reads the bit,
/// closing that correctness divergence without re-running the O(chain) resolver
/// inside every annotate invocation.
///
/// Must run AFTER `precompute_decl_contexts` (it reads the settled `decl_contexts`
/// table) and BEFORE `classify_typedef_vast`'s annotation dispatch.
pub(super) fn precompute_visible_type(
    backend: &dyn VyreBackend,
    path: &Path,
    scoped_vast_blob: &[u8],
    decl_context_blob: &[u8],
    haystack: &[u8],
    haystack_len: u32,
    vast_count: u32,
    packed_haystack: bool,
    cfg: &mut DispatchConfig,
    log: &mut impl FnMut(&str),
) -> Result<Vec<u8>, String> {
    VISIBLE_TYPE_SCRATCH.with(|scratch| {
        let mut scratch = scratch.try_borrow_mut().map_err(|_| {
            "VAST visible-type dispatch scratch was re-entered on the same thread. Fix: call visible-type precompute from a non-nested parser context or add explicit caller-owned scratch.".to_string()
        })?;
        precompute_visible_type_with_scratch(
            backend,
            path,
            scoped_vast_blob,
            decl_context_blob,
            haystack,
            haystack_len,
            vast_count,
            packed_haystack,
            cfg,
            log,
            &mut scratch,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn precompute_visible_type_with_scratch(
    backend: &dyn VyreBackend,
    path: &Path,
    scoped_vast_blob: &[u8],
    decl_context_blob: &[u8],
    haystack: &[u8],
    haystack_len: u32,
    vast_count: u32,
    packed_haystack: bool,
    cfg: &mut DispatchConfig,
    log: &mut impl FnMut(&str),
    scratch: &mut VisibleTypeScratch,
) -> Result<Vec<u8>, String> {
    cfg.label = Some(format!(
        "vyre-frontend-c vast-visible-type {}",
        path.display()
    ));
    let visible_type_key = super::stage_pipeline_cache_key(
        "c11_precompute_vast_visible_type",
        &[
            haystack_len.max(1) as u64,
            vast_count.max(1) as u64,
            packed_haystack as u64,
        ],
    );
    // Binding order for the visible-type program: vast_nodes(0), haystack(1),
    // decl_contexts(2), visible_type(3, output). One input blob per non-output
    // buffer in that order.
    let inputs = [scoped_vast_blob, haystack, decl_context_blob];
    scratch.outputs.clear();
    super::dispatch_borrowed_stage_cached_into(
        backend,
        visible_type_key,
        || {
            let visible_type_prog = if packed_haystack {
                c11_precompute_vast_visible_type_packed_haystack(
                    "vast_nodes",
                    "haystack",
                    "decl_contexts",
                    Expr::u32(haystack_len.max(1)),
                    Expr::u32(vast_count.max(1)),
                    "visible_type",
                )
            } else {
                c11_precompute_vast_visible_type(
                    "vast_nodes",
                    "haystack",
                    "decl_contexts",
                    Expr::u32(haystack_len.max(1)),
                    Expr::u32(vast_count.max(1)),
                    "visible_type",
                )
            };
            let visible_type_prog =
                super::buffers::mark_program_outputs(visible_type_prog, &["visible_type"]);
            super::validate_internal_stage(&visible_type_prog, "c11_precompute_vast_visible_type")?;
            Ok(visible_type_prog)
        },
        &inputs,
        cfg,
        &mut scratch.outputs,
    )
    .map_err(|error| format!("c11_precompute_vast_visible_type dispatch failed: {error}"))?;
    log("dispatch c11_precompute_vast_visible_type");
    if scratch.outputs.len() != 1 {
        return Err(format!(
            "c11_precompute_vast_visible_type returned {} output buffer(s), expected exactly 1. Fix: backend must return only visible_type.",
            scratch.outputs.len()
        ));
    }
    let mut visible_type = Vec::new();
    mem::swap(&mut visible_type, &mut scratch.outputs[0]);
    Ok(visible_type)
}
