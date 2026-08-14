//! CPU oracle dispatcher for this crate's tests and explicit CPU-parity builds.
//!
//! Maps the pass engine's own analysis Programs onto their `vyre_primitives`
//! `cpu_ref` reference implementations and reproduces the dispatch byte
//! contract, so the encoder and decoder can be proven sound against the same
//! numerical contract the production GPU path must honor.

use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

type IfdsIntraRule = (u32, u32, u32);
type IfdsInterRule = (u32, u32, u32, u32);
type IfdsFactRule = (u32, u32, u32);
type ParsedIfdsRules = (
    Vec<IfdsIntraRule>,
    Vec<IfdsInterRule>,
    Vec<IfdsFactRule>,
    Vec<IfdsFactRule>,
);

/// CPU oracle dispatcher. Recognizes only the optimizer's own
/// canonical Programs by matching the wrapping Region's generator
/// op-id and the declared buffer set.
pub struct CpuOracleDispatcher {
    persistent_bfs_aliases: Vec<&'static str>,
}

impl CpuOracleDispatcher {
    /// Construct the oracle dispatcher. Cheap; does no backend
    /// probing.
    #[must_use]
    pub fn new() -> Self {
        Self {
            persistent_bfs_aliases: Vec::new(),
        }
    }

    /// Route `generator` to the persistent-BFS oracle as well. An encoded pass
    /// whose Program is a persistent BFS published under the caller's own op id
    /// registers that id here instead of the oracle naming a consumer it sits
    /// below.
    #[must_use]
    pub fn with_persistent_bfs_alias(mut self, generator: &'static str) -> Self {
        self.persistent_bfs_aliases.push(generator);
        self
    }
}

impl Default for CpuOracleDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgramDispatcher for CpuOracleDispatcher {
    fn dispatch(
        &self,
        program: &Program,
        inputs: &[Vec<u8>],
        _grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        // Identify the optimizer Program by its top-level Region
        // generator. Self-hosted Programs all wrap their bodies
        // in a Region with a known op-id.
        let generator = top_level_region_generator(program).ok_or_else(|| {
            DispatchError::Rejected(
                "Fix: oracle dispatcher only accepts canonical \
                 graph-primitive Programs whose entry is a single \
                 wrapping Region with a generator id."
                    .to_string(),
            )
        })?;

        match generator {
            vyre_primitives::graph::persistent_bfs::OP_ID => {
                persistent_bfs_oracle(program, inputs)
            }
            vyre_primitives::graph::exploded::OP_ID => {
                exploded_ifds_csr_oracle(program, inputs)
            }
            other if self.persistent_bfs_aliases.contains(&other) => {
                persistent_bfs_oracle(program, inputs)
            }
            other => Err(DispatchError::Rejected(format!(
                "Fix: oracle dispatcher does not recognize generator \
                 `{other}`. Wire the oracle for this primitive or \
                 dispatch through the production backend."
            ))),
        }
    }
}

fn top_level_region_generator(program: &Program) -> Option<&str> {
    match program.entry() {
        [vyre_foundation::ir::Node::Region { generator, .. }] => Some(generator.as_str()),
        _ => None,
    }
}

fn persistent_bfs_oracle(
    program: &Program,
    inputs: &[Vec<u8>],
) -> Result<Vec<Vec<u8>>, DispatchError> {
    // Buffer order (per `persistent_bfs.rs::persistent_bfs`):
    //   0 pg_nodes (RO)
    //   1 pg_edge_offsets (RO)
    //   2 pg_edge_targets (RO)
    //   3 pg_edge_kind_mask (RO)
    //   4 pg_node_tags (RO)
    //   5 frontier_in (RO)
    //   6 frontier_out (RW)
    //   7 changed (RW)
    //   8 converged (RW)
    //   9 wg_scratch (workgroup)   -  not an input
    if inputs.len() < 6 {
        return Err(DispatchError::BadInputs(format!(
            "Fix: persistent_bfs oracle expects ≥ 6 input buffers, got {}",
            inputs.len()
        )));
    }
    let nodes = crate::dispatch_buffers::read_u32s(&inputs[0]);
    let edge_offsets = crate::dispatch_buffers::read_u32s(&inputs[1]);
    let edge_targets_raw = crate::dispatch_buffers::read_u32s(&inputs[2]);
    let edge_kind_mask_raw = crate::dispatch_buffers::read_u32s(&inputs[3]);
    let _node_tags = crate::dispatch_buffers::read_u32s(&inputs[4]);
    let frontier_in = crate::dispatch_buffers::read_u32s(&inputs[5]);

    // The Region carries the shape and max_iters in its body
    // structure; rather than re-derive that from IR walks, the
    // oracle re-computes via cpu_ref using the buffers' lengths.
    let node_count = nodes.len() as u32;

    // Iteration cap: if the caller declared `frontier_in` of length L
    // (= bitset_words(node_count)) the oracle uses `node_count` as
    // the saturation budget  -  same default the Program builder uses
    // when callers want closure.
    let max_iters = node_count.max(1);

    let allow_mask = u32::MAX;
    let edge_count = declared_edge_count(&edge_offsets)?;
    let edge_targets = trim_padded_edge_buffer("edge_targets", &edge_targets_raw, edge_count)?;
    let edge_kind_mask =
        trim_padded_edge_buffer("edge_kind_mask", &edge_kind_mask_raw, edge_count)?;

    let (frontier_out, convergence) =
        vyre_primitives::graph::persistent_bfs::try_cpu_ref_converged(
            node_count,
            &edge_offsets,
            edge_targets,
            edge_kind_mask,
            &frontier_in,
            allow_mask,
            max_iters,
        )
        .map_err(DispatchError::BadInputs)?;

    // Emit one buffer per DECLARED writable output, in declared order, rather
    // than a hardcoded list. The oracle stands in for a device backend, so it
    // must return exactly the layout the program declares; a pasted list goes
    // stale the moment a buffer is added and then reports a shape the program
    // does not have. The converged word matches the device readback
    // bit-for-bit (proven in vyre-primitives converged_device_parity), so the
    // oracle is a faithful stand-in for the real GPU signal.
    let declared = vyre_foundation::program_dispatch::declared_dispatch_outputs(program);
    let mut outputs = Vec::with_capacity(declared.len());
    for decl in declared {
        let words = match decl.name() {
            "frontier_out" => frontier_out.clone(),
            "changed" => changed_words_for(decl.count(), &convergence)?,
            "converged" => vec![u32::from(convergence.converged)],
            other => {
                return Err(DispatchError::Rejected(format!(
                    concat!(
                        "Fix: persistent_bfs oracle has no value for declared output ",
                        "buffer `{other}`. Teach the oracle to produce it or stop ",
                        "declaring it; returning a short output list would silently ",
                        "shift every later output index."
                    ),
                    other = other
                )))
            }
        };
        outputs.push(u32_buffer_to_bytes(&words));
    }
    Ok(outputs)
}

/// Reconstruct the `changed` buffer the kernel would leave behind, sized to the
/// program's declared element count.
///
/// The two BFS variants give slot 0 different meanings, and the declared count is
/// what distinguishes them, so deriving from it keeps the oracle honest for both:
///
/// - count 1 (the DCE variant): slot 0 is the LAST iteration's progress, because
///   the kernel zeroes it every iteration to drive its early exit. On a real
///   fixpoint that leaves 0; on an exhausted budget the final iteration grew, so
///   it leaves 1.
/// - count 2 (the sticky variant): slot 0 is that same per-iteration flag and
///   slot 1 is the sticky OR across all iterations, which is what `cpu_ref`
///   reports as `changed`.
///
/// # Errors
/// Rejects any other declared count rather than guessing, since a wrong guess
/// here is a silently misread flag rather than a visible failure.
fn changed_words_for(
    count: u32,
    convergence: &vyre_primitives::graph::persistent_bfs::PersistentBfsConvergence,
) -> Result<Vec<u32>, DispatchError> {
    let per_iteration = u32::from(!convergence.converged);
    match count {
        1 => Ok(vec![per_iteration]),
        2 => Ok(vec![per_iteration, convergence.changed]),
        other => Err(DispatchError::Rejected(format!(
            concat!(
                "Fix: persistent_bfs oracle understands a `changed` buffer of 1 ",
                "element (per-iteration) or 2 (per-iteration plus sticky), not ",
                "{other}. Declare one of those or teach the oracle what the extra ",
                "slots mean."
            ),
            other = other
        ))),
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod changed_words_tests {
    use super::changed_words_for;
    use vyre_primitives::graph::persistent_bfs::PersistentBfsConvergence;

    fn convergence(changed: u32, converged: bool) -> PersistentBfsConvergence {
        PersistentBfsConvergence {
            changed,
            converged,
            stop_iter: 0,
        }
    }

    /// The DCE variant declares one `changed` word holding the LAST iteration's
    /// progress, because the kernel zeroes it every iteration to drive the early
    /// exit. Reaching a fixpoint means that final compare saw zero, even when
    /// earlier iterations did grow the frontier.
    #[test]
    fn a_single_changed_word_reports_the_last_iterations_progress() {
        assert_eq!(
            changed_words_for(1, &convergence(1, true)).expect("count 1 is supported"),
            vec![0],
            "a converged run exits on a zero compare, so the per-iteration flag is 0"
        );
        assert_eq!(
            changed_words_for(1, &convergence(1, false)).expect("count 1 is supported"),
            vec![1],
            "an exhausted budget means the final iteration still grew the frontier"
        );
    }

    /// The sticky variant declares two words: the same per-iteration flag, then the
    /// OR across every iteration. Collapsing them would make a converged run look
    /// like it never grew at all.
    #[test]
    fn two_changed_words_keep_the_per_iteration_and_sticky_flags_distinct() {
        assert_eq!(
            changed_words_for(2, &convergence(1, true)).expect("count 2 is supported"),
            vec![0, 1],
            "slot 0 is the final compare, slot 1 latched because earlier steps grew"
        );
        assert_eq!(
            changed_words_for(2, &convergence(0, true)).expect("count 2 is supported"),
            vec![0, 0],
            "a traversal that never grew converges immediately with nothing latched"
        );
    }

    /// An unrecognized count is refused rather than guessed. Guessing would hand a
    /// consumer a flag read from the wrong slot, which looks like valid data.
    #[test]
    fn an_unrecognized_changed_count_is_refused() {
        let err = changed_words_for(3, &convergence(1, true))
            .expect_err("an unknown changed layout must not be guessed");
        assert!(
            format!("{err}").contains("not 3"),
            "the refusal must name the count it saw, got: {err}"
        );
    }
}

fn exploded_ifds_csr_oracle(
    program: &Program,
    inputs: &[Vec<u8>],
) -> Result<Vec<Vec<u8>>, DispatchError> {
    if inputs.len() != 18 {
        return Err(DispatchError::BadInputs(format!(
            "Fix: exploded IFDS oracle expected 18 input buffers, got {}.",
            inputs.len()
        )));
    }

    let key = vyre_primitives::graph::exploded::ifds_program_cache_key_from_program(program)
        .map_err(DispatchError::BackendError)?;
    let (intra_edges, inter_edges, flow_gen, flow_kill) = parse_ifds_rule_inputs(&key, inputs)?;

    let (row_ptr, col_idx) = vyre_primitives::graph::exploded::build_cpu_reference(
        key.num_procs,
        key.blocks_per_proc,
        key.facts_per_proc,
        &intra_edges,
        &inter_edges,
        &flow_gen,
        &flow_kill,
    );

    let col_len = u32::try_from(col_idx.len()).map_err(|error| {
        DispatchError::BackendError(format!(
            "Fix: exploded IFDS oracle col_idx length does not fit u32: {error}."
        ))
    })?;
    let col_idx_words = program
        .buffer("col_idx")
        .map(|buffer| buffer.count() as usize)
        .unwrap_or(1);
    let mut col_idx_padded = vec![0u32; col_idx_words];
    if col_idx.len() > col_idx_words {
        return Err(DispatchError::BackendError(format!(
            "Fix: exploded IFDS oracle emitted {} columns but program allocates {col_idx_words}."
            ,
            col_idx.len()
        )));
    }
    col_idx_padded[..col_idx.len()].copy_from_slice(&col_idx);

    let row_cursor_words = program
        .buffer("row_cursor")
        .map(|buffer| buffer.count() as usize)
        .unwrap_or(1);
    let row_cursor = vec![0u32; row_cursor_words];

    Ok(vec![
        u32_buffer_to_bytes(&row_ptr),
        u32_buffer_to_bytes(&row_cursor),
        u32_buffer_to_bytes(&col_idx_padded),
        u32_buffer_to_bytes(&[col_len]),
    ])
}

fn parse_ifds_rule_inputs(
    key: &vyre_primitives::graph::exploded::IfdsCsrProgramCacheKey,
    inputs: &[Vec<u8>],
) -> Result<ParsedIfdsRules, DispatchError> {
    let intra_proc = crate::dispatch_buffers::read_u32s(&inputs[0]);
    let intra_src_block = crate::dispatch_buffers::read_u32s(&inputs[1]);
    let intra_dst_block = crate::dispatch_buffers::read_u32s(&inputs[2]);
    let inter_src_proc = crate::dispatch_buffers::read_u32s(&inputs[3]);
    let inter_src_block = crate::dispatch_buffers::read_u32s(&inputs[4]);
    let inter_dst_proc = crate::dispatch_buffers::read_u32s(&inputs[5]);
    let inter_dst_block = crate::dispatch_buffers::read_u32s(&inputs[6]);
    let gen_proc = crate::dispatch_buffers::read_u32s(&inputs[7]);
    let gen_block = crate::dispatch_buffers::read_u32s(&inputs[8]);
    let gen_fact = crate::dispatch_buffers::read_u32s(&inputs[9]);
    let kill_proc = crate::dispatch_buffers::read_u32s(&inputs[10]);
    let kill_block = crate::dispatch_buffers::read_u32s(&inputs[11]);
    let kill_fact = crate::dispatch_buffers::read_u32s(&inputs[12]);

    let intra_edges = read_ifds_triples(
        "intra",
        key.intra_count,
        &intra_proc,
        &intra_src_block,
        &intra_dst_block,
    )?;
    let inter_edges = read_ifds_quads(
        "inter",
        key.inter_count,
        &inter_src_proc,
        &inter_src_block,
        &inter_dst_proc,
        &inter_dst_block,
    )?;
    let flow_gen = read_ifds_triples("GEN", key.gen_count, &gen_proc, &gen_block, &gen_fact)?;
    let flow_kill =
        read_ifds_triples("KILL", key.kill_count, &kill_proc, &kill_block, &kill_fact)?;

    Ok((intra_edges, inter_edges, flow_gen, flow_kill))
}

fn read_ifds_triples(
    kind: &str,
    count: u32,
    proc: &[u32],
    a: &[u32],
    b: &[u32],
) -> Result<Vec<(u32, u32, u32)>, DispatchError> {
    let count = count as usize;
    for (name, column) in [("proc", proc), ("a", a), ("b", b)] {
        if column.len() < count {
            return Err(DispatchError::BadInputs(format!(
                "Fix: exploded IFDS oracle {kind} {name} column has {} word(s), expected {count}."
                ,
                column.len()
            )));
        }
    }
    Ok((0..count)
        .map(|index| (proc[index], a[index], b[index]))
        .collect())
}

fn read_ifds_quads(
    kind: &str,
    count: u32,
    a: &[u32],
    b: &[u32],
    c: &[u32],
    d: &[u32],
) -> Result<Vec<(u32, u32, u32, u32)>, DispatchError> {
    let count = count as usize;
    for (name, column) in [
        ("src_proc", a),
        ("src_block", b),
        ("dst_proc", c),
        ("dst_block", d),
    ] {
        if column.len() < count {
            return Err(DispatchError::BadInputs(format!(
                "Fix: exploded IFDS oracle {kind} {name} column has {} word(s), expected {count}."
                ,
                column.len()
            )));
        }
    }
    Ok((0..count)
        .map(|index| (a[index], b[index], c[index], d[index]))
        .collect())
}

fn declared_edge_count(edge_offsets: &[u32]) -> Result<usize, DispatchError> {
    edge_offsets
        .last()
        .copied()
        .map(|edge_count| edge_count as usize)
        .ok_or_else(|| {
            DispatchError::BadInputs(
                "Fix: persistent_bfs oracle requires a CSR offset sentinel.".to_string(),
            )
        })
}

fn trim_padded_edge_buffer<'a>(
    name: &str,
    buffer: &'a [u32],
    edge_count: usize,
) -> Result<&'a [u32], DispatchError> {
    if buffer.len() < edge_count {
        return Err(DispatchError::BadInputs(format!(
            "Fix: persistent_bfs oracle {name} has {} words but CSR declares {edge_count} edges.",
            buffer.len()
        )));
    }
    Ok(&buffer[..edge_count])
}

fn u32_buffer_to_bytes(words: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(words)
}
