//! Exploded supergraph primitive (G3).
//!
//! # What this is
//!
//! IFDS / IDE reframes interprocedural dataflow as a reachability
//! problem on the **exploded supergraph**: each `(proc, block,
//! fact)` triple is a graph vertex, and the edges are the flow
//! functions (GEN / KILL + summary + call-to-return). Once
//! expanded, the analysis collapses to a BFS over this graph  -
//! which is the exact shape
//! [`crate::graph::csr_forward_traverse`] already handles.
//!
//! This module owns the **node encoding**  -  the bit-layout that
//! packs `(proc_id, block_id, fact_id)` into a single `u32` node id
//!  -  plus a CPU reference that builds the exploded CSR so tests in
//! `vyre-libs::dataflow::ifds_gpu` can prove the GPU kernel produces
//! byte-identical CSR output.
//!
//! # Bit layout
//!
//! ```text
//!   bits 31..20   proc_id   (12 bits  -  4096 procedures per module)
//!   bits 19..10   block_id  (10 bits  -  1024 blocks per procedure)
//!   bits 9..0     fact_id   (10 bits  -  1024 facts per workgroup;
//!                            matches FACTS_PER_WORKGROUP and the
//!                            NFA subgroup sizing)
//! ```
//!
//! This deliberately leaves no room for >4096 procedures in a
//! single module. Any real codebase that exceeds that split along
//! a module boundary first  -  doing interprocedural dataflow over
//! 10 000+ procs in one pass is a different problem that we don't
//! solve here and shouldn't pretend to.
//!
//! # Status
//!
//! Node encoding, CSR builder, and tests. The GPU Program wrapper
//! (the actual kernel that walks edges in parallel) lives in
//! `vyre-libs::dataflow::ifds_gpu` and composes this encoding with
//! `csr_forward_traverse`.

use std::sync::Arc;

use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Canonical op id for the IFDS CSR construction program.
pub const OP_ID: &str = "vyre-primitives::graph::exploded_build_ifds_csr";

/// Bits reserved for each component of the packed node id.
pub const PROC_BITS: u32 = 12;
/// Bits reserved for the basic-block component of the packed node id.
pub const BLOCK_BITS: u32 = 10;
/// Bits reserved for the fact component of the packed node id.
pub const FACT_BITS: u32 = 10;
const _SANITY: () = assert!(PROC_BITS + BLOCK_BITS + FACT_BITS == 32);

/// Max values for each component  -  one less than the available
/// space because zero is a valid id.
pub const MAX_PROC_ID: u32 = (1 << PROC_BITS) - 1;
/// Maximum encodable basic-block id.
pub const MAX_BLOCK_ID: u32 = (1 << BLOCK_BITS) - 1;
/// Maximum encodable fact id.
pub const MAX_FACT_ID: u32 = (1 << FACT_BITS) - 1;

/// Number of facts per workgroup lane. A 32-lane subgroup x
/// 32 bits = 1024 facts; wider subgroup layouts preserve the same budget.
/// Matches the
/// NFA window sizing in `nfa::subgroup_nfa` so both subsystems
/// share occupancy budget.
pub const FACTS_PER_WORKGROUP: usize = 1024;

/// Canonical dispatch input label for intra-procedural procedure ids.
pub const IFDS_CSR_INTRA_PROC_BUFFER: &str = "exploded_ifds_csr intra_proc";
/// Canonical dispatch input label for intra-procedural source blocks.
pub const IFDS_CSR_INTRA_SRC_BLOCK_BUFFER: &str = "exploded_ifds_csr intra_src_block";
/// Canonical dispatch input label for intra-procedural destination blocks.
pub const IFDS_CSR_INTRA_DST_BLOCK_BUFFER: &str = "exploded_ifds_csr intra_dst_block";
/// Canonical dispatch input label for inter-procedural source procedures.
pub const IFDS_CSR_INTER_SRC_PROC_BUFFER: &str = "exploded_ifds_csr inter_src_proc";
/// Canonical dispatch input label for inter-procedural source blocks.
pub const IFDS_CSR_INTER_SRC_BLOCK_BUFFER: &str = "exploded_ifds_csr inter_src_block";
/// Canonical dispatch input label for inter-procedural destination procedures.
pub const IFDS_CSR_INTER_DST_PROC_BUFFER: &str = "exploded_ifds_csr inter_dst_proc";
/// Canonical dispatch input label for inter-procedural destination blocks.
pub const IFDS_CSR_INTER_DST_BLOCK_BUFFER: &str = "exploded_ifds_csr inter_dst_block";
/// Canonical dispatch input label for GEN rule procedures.
pub const IFDS_CSR_GEN_PROC_BUFFER: &str = "exploded_ifds_csr gen_proc";
/// Canonical dispatch input label for GEN rule blocks.
pub const IFDS_CSR_GEN_BLOCK_BUFFER: &str = "exploded_ifds_csr gen_block";
/// Canonical dispatch input label for GEN rule facts.
pub const IFDS_CSR_GEN_FACT_BUFFER: &str = "exploded_ifds_csr gen_fact";
/// Canonical dispatch input label for KILL rule procedures.
pub const IFDS_CSR_KILL_PROC_BUFFER: &str = "exploded_ifds_csr kill_proc";
/// Canonical dispatch input label for KILL rule blocks.
pub const IFDS_CSR_KILL_BLOCK_BUFFER: &str = "exploded_ifds_csr kill_block";
/// Canonical dispatch input label for KILL rule facts.
pub const IFDS_CSR_KILL_FACT_BUFFER: &str = "exploded_ifds_csr kill_fact";
/// Canonical dispatch output label for CSR row pointers.
pub const IFDS_CSR_ROW_PTR_BUFFER: &str = "exploded_ifds_csr row_ptr";
/// Canonical dispatch scratch label for row cursors.
pub const IFDS_CSR_ROW_CURSOR_BUFFER: &str = "exploded_ifds_csr row_cursor";
/// Canonical dispatch output label for CSR column indices.
pub const IFDS_CSR_COL_IDX_BUFFER: &str = "exploded_ifds_csr col_idx";
/// Canonical dispatch output label for emitted column length.
pub const IFDS_CSR_COL_LEN_BUFFER: &str = "exploded_ifds_csr col_len";
/// Single-lane deterministic CSR construction grid.
pub const IFDS_CSR_DISPATCH_GRID: [u32; 3] = [1, 1, 1];

const BLOCK_SHIFT: u32 = FACT_BITS;
const PROC_SHIFT: u32 = FACT_BITS + BLOCK_BITS;
const FACT_MASK: u32 = MAX_FACT_ID;
const BLOCK_MASK: u32 = MAX_BLOCK_ID;
const PROC_MASK: u32 = MAX_PROC_ID;

/// Checked dispatch layout for an exploded IFDS CSR build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IfdsCsrLayout {
    /// Whether the declared IFDS domain is empty and should not dispatch.
    pub empty: bool,
    /// Number of procedures in the exploded domain.
    pub num_procs: u32,
    /// Number of blocks per procedure.
    pub blocks_per_proc: u32,
    /// Number of facts per procedure.
    pub facts_per_proc: u32,
    /// Number of intra-procedural control-flow edges.
    pub intra_count: u32,
    /// Number of inter-procedural call/return edges.
    pub inter_count: u32,
    /// Number of GEN rules.
    pub gen_count: u32,
    /// Number of KILL rules.
    pub kill_count: u32,
    /// Number of u32 words required by each intra edge field buffer.
    pub intra_storage_words: usize,
    /// Number of u32 words required by each inter edge field buffer.
    pub inter_storage_words: usize,
    /// Number of u32 words required by each GEN rule field buffer.
    pub gen_storage_words: usize,
    /// Number of u32 words required by each KILL rule field buffer.
    pub kill_storage_words: usize,
    /// Dense nodes per procedure.
    pub slots_per_proc: u32,
    /// Total dense node count.
    pub total_nodes: u32,
    /// Number of `u32` words in `row_ptr`.
    pub row_words: usize,
    /// Number of `u32` words in the dispatch row cursor scratch buffer.
    pub row_cursor_words: usize,
    /// Maximum emitted column count for the declared edge/rule counts.
    pub max_col_count: u32,
    /// Number of `u32` words allocated for `col_idx`.
    pub col_buffer_words: usize,
}

/// Primitive-owned cache identity for exploded IFDS CSR construction Programs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IfdsCsrProgramCacheKey {
    /// Number of procedures in the exploded domain.
    pub num_procs: u32,
    /// Number of blocks per procedure.
    pub blocks_per_proc: u32,
    /// Number of facts per procedure.
    pub facts_per_proc: u32,
    /// Number of intra-procedural control-flow edges.
    pub intra_count: u32,
    /// Number of inter-procedural call/return edges.
    pub inter_count: u32,
    /// Number of GEN rules.
    pub gen_count: u32,
    /// Number of KILL rules.
    pub kill_count: u32,
    /// Maximum emitted column count baked into the generated Program.
    pub max_col_count: u32,
}

/// Stable identity for IFDS rule tuples supplied to the CSR builder.
///
/// This is intentionally distinct from [`IfdsCsrProgramCacheKey`]: the generated
/// program depends on dimensions and rule counts, while staged dispatch input
/// reuse also depends on the actual tuple contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IfdsCsrRuleInputFingerprint {
    /// Fingerprint of `(proc, src_block, dst_block)` intra edges.
    pub intra: u128,
    /// Fingerprint of `(src_proc, src_block, dst_proc, dst_block)` inter edges.
    pub inter: u128,
    /// Fingerprint of `(proc, block, fact)` GEN rules.
    pub gen: u128,
    /// Fingerprint of `(proc, block, fact)` KILL rules.
    pub kill: u128,
}

/// Primitive-owned identity for reusable exploded IFDS static inputs.
///
/// The generated Program depends on [`IfdsCsrProgramCacheKey`]. Staged rule
/// inputs also depend on tuple contents, so dispatch wrappers use this key to
/// refresh uploads without owning IFDS fingerprint composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IfdsCsrStaticInputKey {
    /// Program-shape key selected by the primitive dispatch plan.
    pub program_key: IfdsCsrProgramCacheKey,
    /// Stable content fingerprint of all staged IFDS rule tuples.
    pub rule_fingerprint: IfdsCsrRuleInputFingerprint,
}

impl IfdsCsrRuleInputFingerprint {
    /// Build a stable rule-content fingerprint without allocating columns.
    #[must_use]
    pub fn from_rules(
        intra_edges: &[(u32, u32, u32)],
        inter_edges: &[(u32, u32, u32, u32)],
        flow_gen: &[(u32, u32, u32)],
        flow_kill: &[(u32, u32, u32)],
    ) -> Self {
        Self {
            intra: fingerprint_rule_triples(intra_edges),
            inter: fingerprint_rule_quads(inter_edges),
            gen: fingerprint_rule_triples(flow_gen),
            kill: fingerprint_rule_triples(flow_kill),
        }
    }
}

fn mix_rule_word(hash: &mut u128, value: u32) {
    *hash ^= u128::from(value)
        .wrapping_add(0x9E37_79B9_7F4A_7C15_6A09_E667_F3BC_C909)
        .wrapping_add(*hash << 7)
        .wrapping_add(*hash >> 3);
    *hash = hash
        .rotate_left(31)
        .wrapping_mul(0xD6E8_FD9D_DA37_3C91_BB67_AE85_84CA_A73B);
}

fn fingerprint_rule_triples(rules: &[(u32, u32, u32)]) -> u128 {
    let mut hash = 0x243F_6A88_85A3_08D3_1319_8A2E_0370_7344_u128 ^ rules.len() as u128;
    for &(a, b, c) in rules {
        mix_rule_word(&mut hash, a);
        mix_rule_word(&mut hash, b);
        mix_rule_word(&mut hash, c);
    }
    hash
}

fn fingerprint_rule_quads(rules: &[(u32, u32, u32, u32)]) -> u128 {
    let mut hash = 0xA409_3822_299F_31D0_082E_FA98_EC4E_6C89_u128 ^ rules.len() as u128;
    for &(a, b, c, d) in rules {
        mix_rule_word(&mut hash, a);
        mix_rule_word(&mut hash, b);
        mix_rule_word(&mut hash, c);
        mix_rule_word(&mut hash, d);
    }
    hash
}

impl IfdsCsrProgramCacheKey {
    /// Build a Program cache key from a validated IFDS layout.
    #[must_use]
    pub const fn from_layout(layout: &IfdsCsrLayout) -> Self {
        Self {
            num_procs: layout.num_procs,
            blocks_per_proc: layout.blocks_per_proc,
            facts_per_proc: layout.facts_per_proc,
            intra_count: layout.intra_count,
            inter_count: layout.inter_count,
            gen_count: layout.gen_count,
            kill_count: layout.kill_count,
            max_col_count: layout.max_col_count,
        }
    }
}

/// Primitive-owned dispatch plan for exploded IFDS CSR construction.
///
/// Consumers own only rule marshalling and backend invocation. Buffer labels,
/// padded storage widths, readback widths, and grid shape live here so
/// self-substrate and future consumers cannot fork the dispatch contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IfdsCsrDispatchPlan {
    /// Validated CSR layout and rule counts.
    pub layout: IfdsCsrLayout,
    /// Primitive-owned generated-Program cache identity.
    pub program_key: IfdsCsrProgramCacheKey,
    /// Dispatch grid override.
    pub grid: [u32; 3],
    /// Padded words for each intra edge field.
    pub intra_field_words: usize,
    /// Padded words for each inter edge field.
    pub inter_field_words: usize,
    /// Padded words for each GEN field.
    pub gen_field_words: usize,
    /// Padded words for each KILL field.
    pub kill_field_words: usize,
    /// Words in the CSR row-pointer output.
    pub row_ptr_words: usize,
    /// Words in the row-cursor scratch output.
    pub row_cursor_words: usize,
    /// Words in the CSR column-index output.
    pub col_idx_words: usize,
    /// Words in the emitted-column-length output.
    pub col_len_words: usize,
    /// Maximum legal emitted column count.
    pub max_col_count: u32,
}

impl IfdsCsrDispatchPlan {
    /// Build the GPU program for this validated dispatch plan.
    #[must_use]
    pub fn program(&self) -> Program {
        build_ifds_csr_program(
            self.layout.num_procs,
            self.layout.blocks_per_proc,
            self.layout.facts_per_proc,
            self.layout.intra_count,
            self.layout.inter_count,
            self.layout.gen_count,
            self.layout.kill_count,
            self.layout.max_col_count,
        )
    }

    /// Stable generated-Program cache identity for this dispatch shape.
    #[must_use]
    pub const fn program_cache_key(&self) -> IfdsCsrProgramCacheKey {
        self.program_key
    }

    /// Stable identity for static rule uploads under this dispatch plan.
    #[must_use]
    pub const fn static_input_key(
        &self,
        rule_fingerprint: IfdsCsrRuleInputFingerprint,
    ) -> IfdsCsrStaticInputKey {
        IfdsCsrStaticInputKey {
            program_key: self.program_key,
            rule_fingerprint,
        }
    }
}

/// Validate CSR data returned by an exploded IFDS backend.
///
/// # Errors
///
/// Returns an actionable diagnostic when row pointers, live column length, or
/// live column indices do not satisfy the primitive CSR contract.
pub fn validate_ifds_csr_readback(
    layout: &IfdsCsrLayout,
    row_ptr: &[u32],
    col_idx: &[u32],
    col_len: u32,
) -> Result<usize, String> {
    if row_ptr.len() != layout.row_words {
        return Err(format!(
            "Fix: exploded IFDS row_ptr readback expected {} word(s), got {}.",
            layout.row_words,
            row_ptr.len()
        ));
    }
    if row_ptr.first().copied() != Some(0) {
        return Err("Fix: exploded IFDS CSR row_ptr[0] must be 0.".to_string());
    }
    if col_len > layout.max_col_count {
        return Err(format!(
            "Fix: exploded IFDS GPU reported col_len {col_len} above allocated maximum {}.",
            layout.max_col_count
        ));
    }
    let live_cols = usize::try_from(col_len).map_err(|_| {
        format!("Fix: exploded IFDS col_len {col_len} cannot be represented as usize.")
    })?;
    if live_cols > col_idx.len() {
        return Err(format!(
            "Fix: exploded IFDS col_len {col_len} exceeds col_idx readback words {}.",
            col_idx.len()
        ));
    }
    for (row, window) in row_ptr.windows(2).enumerate() {
        let start = window[0];
        let end = window[1];
        if start > end {
            return Err(format!(
                "Fix: exploded IFDS row_ptr is not monotonic at row {row}: {start} > {end}."
            ));
        }
        if end > col_len {
            return Err(format!(
                "Fix: exploded IFDS row {row} ends at {end}, beyond live col_len {col_len}."
            ));
        }
    }
    let final_row = row_ptr.last().copied().unwrap_or(0);
    if final_row != col_len {
        return Err(format!(
            "Fix: exploded IFDS final row_ptr value {final_row} must equal col_len {col_len}."
        ));
    }
    for (index, &column) in col_idx.iter().take(live_cols).enumerate() {
        if column >= layout.total_nodes {
            return Err(format!(
                "Fix: exploded IFDS col_idx[{index}]={column} is outside total_nodes {}.",
                layout.total_nodes
            ));
        }
    }
    Ok(live_cols)
}

/// Checked exploded-supergraph node count.
#[must_use]
pub fn ifds_node_count_checked(
    num_procs: u32,
    blocks_per_proc: u32,
    facts_per_proc: u32,
) -> Option<u32> {
    num_procs
        .checked_mul(blocks_per_proc)?
        .checked_mul(facts_per_proc)
}

/// Saturating exploded-supergraph node count for capacity planning UIs.
#[must_use]
pub fn ifds_node_count_saturating(
    num_procs: u32,
    blocks_per_proc: u32,
    facts_per_proc: u32,
) -> u32 {
    num_procs
        .saturating_mul(blocks_per_proc)
        .saturating_mul(facts_per_proc)
}

/// Maximum column count needed by the deterministic IFDS CSR builder.
#[must_use]
pub fn max_ifds_col_count(
    intra_count: u32,
    inter_count: u32,
    gen_count: u32,
    facts_per_proc: u32,
) -> Option<u32> {
    intra_count
        .checked_mul(facts_per_proc)
        .and_then(|v| v.checked_add(intra_count.checked_mul(gen_count)?))
        .and_then(|v| v.checked_add(inter_count.checked_mul(facts_per_proc)?))
}

/// Validate dimensions/counts and return the exact dispatch buffer layout.
pub fn validate_ifds_csr_layout(
    num_procs: u32,
    blocks_per_proc: u32,
    facts_per_proc: u32,
    intra_count: u32,
    inter_count: u32,
    gen_count: u32,
) -> Result<IfdsCsrLayout, String> {
    if num_procs == 0 || blocks_per_proc == 0 || facts_per_proc == 0 {
        return Err(format!(
            "Fix: exploded IFDS dimensions must be nonzero, got procs={num_procs}, blocks={blocks_per_proc}, facts={facts_per_proc}."
        ));
    }
    if !fits(
        num_procs.saturating_sub(1),
        blocks_per_proc.saturating_sub(1),
        facts_per_proc.saturating_sub(1),
    ) {
        return Err(format!(
            "Fix: exploded IFDS dimensions exceed packed IFDS limits: procs={num_procs}, blocks={blocks_per_proc}, facts={facts_per_proc}."
        ));
    }
    let slots_per_proc = blocks_per_proc.checked_mul(facts_per_proc).ok_or_else(|| {
        format!(
            "Fix: exploded IFDS blocks*facts overflows u32: {blocks_per_proc}*{facts_per_proc}."
        )
    })?;
    let total_nodes = num_procs.checked_mul(slots_per_proc).ok_or_else(|| {
        format!(
            "Fix: exploded IFDS procs*blocks*facts overflows u32: {num_procs}*{blocks_per_proc}*{facts_per_proc}."
        )
    })?;
    let row_ptr_count = total_nodes.checked_add(1).ok_or_else(|| {
        format!(
            "Fix: exploded IFDS total_nodes={total_nodes} overflows row_ptr count. Shard the IFDS graph before GPU dispatch."
        )
    })?;
    let max_col_count = max_ifds_col_count(intra_count, inter_count, gen_count, facts_per_proc)
        .ok_or_else(|| "Fix: exploded IFDS maximum column count overflows u32.".to_string())?;
    Ok(IfdsCsrLayout {
        empty: false,
        num_procs,
        blocks_per_proc,
        facts_per_proc,
        intra_count,
        inter_count,
        gen_count,
        kill_count: 0,
        intra_storage_words: (intra_count as usize).max(1),
        inter_storage_words: (inter_count as usize).max(1),
        gen_storage_words: (gen_count as usize).max(1),
        kill_storage_words: 1,
        slots_per_proc,
        total_nodes,
        row_words: row_ptr_count as usize,
        row_cursor_words: (total_nodes as usize).max(1),
        max_col_count,
        col_buffer_words: (max_col_count as usize).max(1),
    })
}

/// Validate the full IFDS CSR dispatch contract from caller-owned rule slices.
///
/// Returns the exact primitive dispatch layout so consumers do not narrow rule
/// counts or decide padded input-buffer widths locally.
pub fn validate_ifds_csr_inputs(
    num_procs: u32,
    blocks_per_proc: u32,
    facts_per_proc: u32,
    intra_edges: &[(u32, u32, u32)],
    inter_edges: &[(u32, u32, u32, u32)],
    flow_gen: &[(u32, u32, u32)],
    flow_kill: &[(u32, u32, u32)],
) -> Result<IfdsCsrLayout, String> {
    let intra_count = checked_rule_count("intra edge", intra_edges.len())?;
    let inter_count = checked_rule_count("inter edge", inter_edges.len())?;
    let gen_count = checked_rule_count("GEN", flow_gen.len())?;
    let kill_count = checked_rule_count("KILL", flow_kill.len())?;

    if num_procs == 0 || blocks_per_proc == 0 || facts_per_proc == 0 {
        if intra_count == 0 && inter_count == 0 && gen_count == 0 && kill_count == 0 {
            return Ok(IfdsCsrLayout {
                empty: true,
                num_procs,
                blocks_per_proc,
                facts_per_proc,
                intra_count,
                inter_count,
                gen_count,
                kill_count,
                intra_storage_words: 1,
                inter_storage_words: 1,
                gen_storage_words: 1,
                kill_storage_words: 1,
                slots_per_proc: 0,
                total_nodes: 0,
                row_words: 1,
                row_cursor_words: 1,
                max_col_count: 0,
                col_buffer_words: 1,
            });
        }
        return Err(format!(
            "Fix: exploded IFDS empty dimensions cannot carry rules, got intra={intra_count}, inter={inter_count}, gen={gen_count}, kill={kill_count}."
        ));
    }

    let mut layout = validate_ifds_csr_layout(
        num_procs,
        blocks_per_proc,
        facts_per_proc,
        intra_count,
        inter_count,
        gen_count,
    )?;
    layout.kill_count = kill_count;
    layout.kill_storage_words = flow_kill.len().max(1);
    Ok(layout)
}

/// Validate caller-owned IFDS rules and return the complete primitive dispatch plan.
pub fn plan_ifds_csr_dispatch(
    num_procs: u32,
    blocks_per_proc: u32,
    facts_per_proc: u32,
    intra_edges: &[(u32, u32, u32)],
    inter_edges: &[(u32, u32, u32, u32)],
    flow_gen: &[(u32, u32, u32)],
    flow_kill: &[(u32, u32, u32)],
) -> Result<IfdsCsrDispatchPlan, String> {
    let layout = validate_ifds_csr_inputs(
        num_procs,
        blocks_per_proc,
        facts_per_proc,
        intra_edges,
        inter_edges,
        flow_gen,
        flow_kill,
    )?;
    Ok(IfdsCsrDispatchPlan {
        intra_field_words: layout.intra_storage_words,
        inter_field_words: layout.inter_storage_words,
        gen_field_words: layout.gen_storage_words,
        kill_field_words: layout.kill_storage_words,
        row_ptr_words: layout.row_words,
        row_cursor_words: layout.row_cursor_words,
        col_idx_words: layout.col_buffer_words,
        col_len_words: 1,
        max_col_count: layout.max_col_count,
        program_key: IfdsCsrProgramCacheKey::from_layout(&layout),
        layout,
        grid: IFDS_CSR_DISPATCH_GRID,
    })
}

/// Caller-owned structure-of-arrays rule columns for IFDS CSR dispatch.
///
/// This lives with the primitive dispatch plan because field order, padding,
/// and rule-domain grouping are part of the primitive ABI. Dispatch consumers
/// reuse one value across calls and upload these columns without re-forking
/// IFDS tuple marshalling.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct IfdsCsrRuleColumns {
    /// Intra-procedural procedure ids.
    pub intra_proc: Vec<u32>,
    /// Intra-procedural source blocks.
    pub intra_src_block: Vec<u32>,
    /// Intra-procedural destination blocks.
    pub intra_dst_block: Vec<u32>,
    /// Inter-procedural source procedures.
    pub inter_src_proc: Vec<u32>,
    /// Inter-procedural source blocks.
    pub inter_src_block: Vec<u32>,
    /// Inter-procedural destination procedures.
    pub inter_dst_proc: Vec<u32>,
    /// Inter-procedural destination blocks.
    pub inter_dst_block: Vec<u32>,
    /// GEN rule procedures.
    pub gen_proc: Vec<u32>,
    /// GEN rule blocks.
    pub gen_block: Vec<u32>,
    /// GEN rule facts.
    pub gen_fact: Vec<u32>,
    /// KILL rule procedures.
    pub kill_proc: Vec<u32>,
    /// KILL rule blocks.
    pub kill_block: Vec<u32>,
    /// KILL rule facts.
    pub kill_fact: Vec<u32>,
}

impl IfdsCsrRuleColumns {
    /// Split all IFDS tuple rules into primitive-owned structure-of-arrays
    /// columns, reusing existing allocations.
    ///
    /// # Errors
    ///
    /// Returns an allocation diagnostic if any output column cannot reserve
    /// enough space for its incoming rule slice.
    pub fn prepare(
        &mut self,
        intra_edges: &[(u32, u32, u32)],
        inter_edges: &[(u32, u32, u32, u32)],
        flow_gen: &[(u32, u32, u32)],
        flow_kill: &[(u32, u32, u32)],
    ) -> Result<(), String> {
        split_ifds_rule_triples_into(
            intra_edges,
            &mut self.intra_proc,
            &mut self.intra_src_block,
            &mut self.intra_dst_block,
            "IFDS intra edge columns",
        )?;
        split_ifds_rule_quads_into(
            inter_edges,
            &mut self.inter_src_proc,
            &mut self.inter_src_block,
            &mut self.inter_dst_proc,
            &mut self.inter_dst_block,
            "IFDS inter edge columns",
        )?;
        split_ifds_rule_triples_into(
            flow_gen,
            &mut self.gen_proc,
            &mut self.gen_block,
            &mut self.gen_fact,
            "IFDS GEN columns",
        )?;
        split_ifds_rule_triples_into(
            flow_kill,
            &mut self.kill_proc,
            &mut self.kill_block,
            &mut self.kill_fact,
            "IFDS KILL columns",
        )
    }
}

#[cfg(test)]
mod dispatch_plan_tests {
    use super::*;

    #[test]
    fn plan_owns_padding_outputs_and_grid() {
        let plan = plan_ifds_csr_dispatch(
            2,
            2,
            2,
            &[(0, 0, 1)],
            &[(0, 1, 1, 0)],
            &[(0, 0, 1)],
            &[(1, 0, 0)],
        )
        .expect("Fix: valid IFDS CSR dispatch plan should build");

        assert_eq!(plan.grid, IFDS_CSR_DISPATCH_GRID);
        assert_eq!(plan.intra_field_words, 1);
        assert_eq!(plan.inter_field_words, 1);
        assert_eq!(plan.gen_field_words, 1);
        assert_eq!(plan.kill_field_words, 1);
        assert_eq!(plan.row_ptr_words, 9);
        assert_eq!(plan.row_cursor_words, 8);
        assert_eq!(plan.col_idx_words, 5);
        assert_eq!(plan.col_len_words, 1);
        assert_eq!(plan.max_col_count, 5);
        assert_eq!(
            plan.program_cache_key(),
            IfdsCsrProgramCacheKey {
                num_procs: 2,
                blocks_per_proc: 2,
                facts_per_proc: 2,
                intra_count: 1,
                inter_count: 1,
                gen_count: 1,
                kill_count: 1,
                max_col_count: 5,
            }
        );
        assert!(!plan.layout.empty);
    }

    #[test]
    fn empty_plan_keeps_dispatch_buffers_nonempty_without_fake_rules() {
        let plan = plan_ifds_csr_dispatch(0, 0, 0, &[], &[], &[], &[])
            .expect("Fix: empty no-rule IFDS dispatch plan should be representable");

        assert!(plan.layout.empty);
        assert_eq!(plan.intra_field_words, 1);
        assert_eq!(plan.inter_field_words, 1);
        assert_eq!(plan.gen_field_words, 1);
        assert_eq!(plan.kill_field_words, 1);
        assert_eq!(plan.row_ptr_words, 1);
        assert_eq!(plan.row_cursor_words, 1);
        assert_eq!(plan.col_idx_words, 1);
        assert_eq!(plan.col_len_words, 1);
        assert_eq!(plan.grid, IFDS_CSR_DISPATCH_GRID);
    }

    #[test]
    fn rule_input_fingerprint_distinguishes_same_count_rule_content() {
        let base = IfdsCsrRuleInputFingerprint::from_rules(
            &[(0, 0, 1)],
            &[(0, 1, 1, 0)],
            &[(0, 0, 1)],
            &[(1, 0, 0)],
        );

        assert_eq!(
            base,
            IfdsCsrRuleInputFingerprint::from_rules(
                &[(0, 0, 1)],
                &[(0, 1, 1, 0)],
                &[(0, 0, 1)],
                &[(1, 0, 0)],
            )
        );
        assert_ne!(
            base,
            IfdsCsrRuleInputFingerprint::from_rules(
                &[(0, 1, 0)],
                &[(0, 1, 1, 0)],
                &[(0, 0, 1)],
                &[(1, 0, 0)],
            )
        );
        assert_ne!(
            base,
            IfdsCsrRuleInputFingerprint::from_rules(
                &[(0, 0, 1)],
                &[(0, 1, 1, 1)],
                &[(0, 0, 1)],
                &[(1, 0, 0)],
            )
        );
    }

    #[test]
    fn static_input_key_combines_program_shape_and_rule_content() {
        let plan = plan_ifds_csr_dispatch(1, 2, 1, &[(0, 0, 1)], &[], &[], &[])
            .expect("Fix: valid IFDS dispatch plan should build");
        let first = IfdsCsrRuleInputFingerprint::from_rules(&[(0, 0, 1)], &[], &[], &[]);
        let changed = IfdsCsrRuleInputFingerprint::from_rules(&[(0, 1, 0)], &[], &[], &[]);

        assert_eq!(plan.static_input_key(first), plan.static_input_key(first));
        assert_ne!(plan.static_input_key(first), plan.static_input_key(changed));
        assert_eq!(
            plan.static_input_key(first).program_key,
            plan.program_cache_key()
        );
    }

    #[test]
    fn readback_validator_rejects_malformed_csr_outputs() {
        let plan = plan_ifds_csr_dispatch(1, 2, 1, &[(0, 0, 1)], &[], &[], &[])
            .expect("Fix: valid IFDS dispatch plan should build");
        let layout = &plan.layout;

        assert_eq!(
            validate_ifds_csr_readback(layout, &[0, 1, 1], &[1], 1)
                .expect("Fix: canonical readback should validate"),
            1
        );
        assert!(validate_ifds_csr_readback(layout, &[1, 1, 1], &[1], 1)
            .expect_err("Fix: row_ptr[0] drift must be rejected")
            .contains("row_ptr[0]"));
        assert!(validate_ifds_csr_readback(layout, &[0, 1, 0], &[1], 1)
            .expect_err("Fix: nonmonotonic row_ptr must be rejected")
            .contains("not monotonic"));
        assert!(validate_ifds_csr_readback(layout, &[0, 1, 1], &[2], 1)
            .expect_err("Fix: out-of-domain column must be rejected")
            .contains("outside total_nodes"));
    }
}

fn checked_rule_count(kind: &str, len: usize) -> Result<u32, String> {
    u32::try_from(len)
        .map_err(|_| format!("Fix: exploded IFDS {kind} count {len} exceeds u32 index space."))
}

/// Split IFDS triple rules into primitive-owned structure-of-arrays columns.
///
/// This keeps the rule-column layout beside the dispatch plan so wrappers do
/// not reimplement tuple marshalling before uploading GPU buffers.
///
/// # Errors
///
/// Returns an allocation diagnostic if any output column cannot reserve enough
/// space for the incoming rules.
pub fn split_ifds_rule_triples_into(
    triples: &[(u32, u32, u32)],
    first: &mut Vec<u32>,
    second: &mut Vec<u32>,
    third: &mut Vec<u32>,
    context: &str,
) -> Result<(), String> {
    first.clear();
    second.clear();
    third.clear();
    crate::graph::scratch::reserve_graph_items(
        first,
        triples.len(),
        "exploded IFDS primitive",
        context,
    )?;
    crate::graph::scratch::reserve_graph_items(
        second,
        triples.len(),
        "exploded IFDS primitive",
        context,
    )?;
    crate::graph::scratch::reserve_graph_items(
        third,
        triples.len(),
        "exploded IFDS primitive",
        context,
    )?;
    for &(a, b, c) in triples {
        first.push(a);
        second.push(b);
        third.push(c);
    }
    Ok(())
}

/// Split IFDS quadruple rules into primitive-owned structure-of-arrays columns.
///
/// # Errors
///
/// Returns an allocation diagnostic if any output column cannot reserve enough
/// space for the incoming rules.
pub fn split_ifds_rule_quads_into(
    quads: &[(u32, u32, u32, u32)],
    first: &mut Vec<u32>,
    second: &mut Vec<u32>,
    third: &mut Vec<u32>,
    fourth: &mut Vec<u32>,
    context: &str,
) -> Result<(), String> {
    first.clear();
    second.clear();
    third.clear();
    fourth.clear();
    crate::graph::scratch::reserve_graph_items(
        first,
        quads.len(),
        "exploded IFDS primitive",
        context,
    )?;
    crate::graph::scratch::reserve_graph_items(
        second,
        quads.len(),
        "exploded IFDS primitive",
        context,
    )?;
    crate::graph::scratch::reserve_graph_items(
        third,
        quads.len(),
        "exploded IFDS primitive",
        context,
    )?;
    crate::graph::scratch::reserve_graph_items(
        fourth,
        quads.len(),
        "exploded IFDS primitive",
        context,
    )?;
    for &(a, b, c, d) in quads {
        first.push(a);
        second.push(b);
        third.push(c);
        fourth.push(d);
    }
    Ok(())
}

#[cfg(test)]
mod ifds_rule_column_tests {
    use super::*;

    #[test]
    fn generated_split3_preserves_tuple_columns_after_reuse() {
        for len in 0usize..4096 {
            let triples: Vec<(u32, u32, u32)> = (0..len)
                .map(|index| {
                    let value = index as u32;
                    (value, value.wrapping_mul(3), value.wrapping_mul(7))
                })
                .collect();
            let mut first = vec![0xAAAA_AAAAu32; 5];
            let mut second = vec![0xBBBB_BBBBu32; 7];
            let mut third = vec![0xCCCC_CCCCu32; 11];
            split_ifds_rule_triples_into(
                &triples,
                &mut first,
                &mut second,
                &mut third,
                "generated split3",
            )
            .unwrap();
            assert_eq!(first.len(), len);
            assert_eq!(second.len(), len);
            assert_eq!(third.len(), len);
            for (index, &(a, b, c)) in triples.iter().enumerate() {
                assert_eq!(first[index], a);
                assert_eq!(second[index], b);
                assert_eq!(third[index], c);
            }
        }
    }

    #[test]
    fn generated_split4_preserves_tuple_columns_after_reuse() {
        for len in 0usize..4096 {
            let quads: Vec<(u32, u32, u32, u32)> = (0..len)
                .map(|index| {
                    let value = index as u32;
                    (
                        value,
                        value.wrapping_mul(5),
                        value.wrapping_mul(11),
                        value.wrapping_mul(13),
                    )
                })
                .collect();
            let mut first = vec![0xAAAA_AAAAu32; 5];
            let mut second = vec![0xBBBB_BBBBu32; 7];
            let mut third = vec![0xCCCC_CCCCu32; 11];
            let mut fourth = vec![0xDDDD_DDDDu32; 13];
            split_ifds_rule_quads_into(
                &quads,
                &mut first,
                &mut second,
                &mut third,
                &mut fourth,
                "generated split4",
            )
            .unwrap();
            assert_eq!(first.len(), len);
            assert_eq!(second.len(), len);
            assert_eq!(third.len(), len);
            assert_eq!(fourth.len(), len);
            for (index, &(a, b, c, d)) in quads.iter().enumerate() {
                assert_eq!(first[index], a);
                assert_eq!(second[index], b);
                assert_eq!(third[index], c);
                assert_eq!(fourth[index], d);
            }
        }
    }

    #[test]
    fn rule_columns_prepare_splits_all_domains_and_reuses_storage() {
        let mut columns = IfdsCsrRuleColumns::default();
        columns
            .prepare(
                &[(1, 2, 3), (4, 5, 6)],
                &[(7, 8, 9, 10)],
                &[(11, 12, 13)],
                &[(14, 15, 16), (17, 18, 19)],
            )
            .expect("Fix: IFDS rule columns should prepare");
        let capacities = [
            columns.intra_proc.capacity(),
            columns.intra_src_block.capacity(),
            columns.intra_dst_block.capacity(),
            columns.inter_src_proc.capacity(),
            columns.inter_src_block.capacity(),
            columns.inter_dst_proc.capacity(),
            columns.inter_dst_block.capacity(),
            columns.gen_proc.capacity(),
            columns.gen_block.capacity(),
            columns.gen_fact.capacity(),
            columns.kill_proc.capacity(),
            columns.kill_block.capacity(),
            columns.kill_fact.capacity(),
        ];

        assert_eq!(columns.intra_proc, [1, 4]);
        assert_eq!(columns.intra_src_block, [2, 5]);
        assert_eq!(columns.intra_dst_block, [3, 6]);
        assert_eq!(columns.inter_src_proc, [7]);
        assert_eq!(columns.inter_src_block, [8]);
        assert_eq!(columns.inter_dst_proc, [9]);
        assert_eq!(columns.inter_dst_block, [10]);
        assert_eq!(columns.gen_proc, [11]);
        assert_eq!(columns.gen_block, [12]);
        assert_eq!(columns.gen_fact, [13]);
        assert_eq!(columns.kill_proc, [14, 17]);
        assert_eq!(columns.kill_block, [15, 18]);
        assert_eq!(columns.kill_fact, [16, 19]);

        columns
            .prepare(&[(20, 21, 22)], &[], &[], &[])
            .expect("Fix: IFDS rule columns should reuse storage for smaller batches");
        assert_eq!(columns.intra_proc, [20]);
        assert_eq!(columns.intra_src_block, [21]);
        assert_eq!(columns.intra_dst_block, [22]);
        assert!(columns.inter_src_proc.is_empty());
        assert!(columns.gen_proc.is_empty());
        assert!(columns.kill_proc.is_empty());
        assert_eq!(columns.intra_proc.capacity(), capacities[0]);
        assert_eq!(columns.intra_src_block.capacity(), capacities[1]);
        assert_eq!(columns.intra_dst_block.capacity(), capacities[2]);
        assert_eq!(columns.inter_src_proc.capacity(), capacities[3]);
        assert_eq!(columns.inter_src_block.capacity(), capacities[4]);
        assert_eq!(columns.inter_dst_proc.capacity(), capacities[5]);
        assert_eq!(columns.inter_dst_block.capacity(), capacities[6]);
        assert_eq!(columns.gen_proc.capacity(), capacities[7]);
        assert_eq!(columns.gen_block.capacity(), capacities[8]);
        assert_eq!(columns.gen_fact.capacity(), capacities[9]);
        assert_eq!(columns.kill_proc.capacity(), capacities[10]);
        assert_eq!(columns.kill_block.capacity(), capacities[11]);
        assert_eq!(columns.kill_fact.capacity(), capacities[12]);
    }
}

/// Sort each CSR row in place after validating row ranges.
pub fn canonicalize_csr_within_rows_in_place(
    row_ptr: &[u32],
    col_idx: &mut [u32],
) -> Result<(), String> {
    for window in row_ptr.windows(2) {
        let start = window[0] as usize;
        let end = window[1] as usize;
        if start > end || end > col_idx.len() {
            return Err(format!(
                "Fix: exploded IFDS CSR row range {start}..{end} exceeds col_idx.len()={}.",
                col_idx.len()
            ));
        }
        col_idx[start..end].sort_unstable();
    }
    Ok(())
}

/// Return a row-canonical CSR copy.
#[must_use]
pub fn canonicalize_csr_within_rows(row_ptr: &[u32], col_idx: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut canonical_col = col_idx.to_vec();
    if canonicalize_csr_within_rows_in_place(row_ptr, &mut canonical_col).is_err() {
        canonical_col.copy_from_slice(col_idx);
    }
    (row_ptr.to_vec(), canonical_col)
}

/// Build a GPU Program that emits the exploded-supergraph CSR.
///
/// This is a deterministic single-lane construction pass: count each
/// source row, prefix row counts, then fill `col_idx`. It removes the
/// production CPU-reference path while preserving a stable API for a
/// later parallel count/scan/fill implementation.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn build_ifds_csr_program(
    num_procs: u32,
    blocks_per_proc: u32,
    facts_per_proc: u32,
    intra_count: u32,
    inter_count: u32,
    gen_count: u32,
    kill_count: u32,
    max_col_count: u32,
) -> Program {
    if num_procs == 0 || blocks_per_proc == 0 || facts_per_proc == 0 {
        return crate::invalid_output_program(
            OP_ID,
            "row_ptr",
            DataType::U32,
            format!(
                "Fix: exploded IFDS dimensions must be nonzero, got procs={num_procs}, blocks={blocks_per_proc}, facts={facts_per_proc}."
            ),
        );
    }
    let Some(slots_per_proc) = blocks_per_proc.checked_mul(facts_per_proc) else {
        return crate::invalid_output_program(
            OP_ID,
            "row_ptr",
            DataType::U32,
            "Fix: exploded IFDS slots_per_proc overflowed u32.".to_string(),
        );
    };
    let Some(total_nodes) = num_procs.checked_mul(slots_per_proc) else {
        return crate::invalid_output_program(
            OP_ID,
            "row_ptr",
            DataType::U32,
            "Fix: exploded IFDS total node count overflowed u32.".to_string(),
        );
    };
    let Some(row_ptr_count) = total_nodes.checked_add(1) else {
        return crate::invalid_output_program(
            OP_ID,
            "row_ptr",
            DataType::U32,
            format!(
                "Fix: exploded IFDS total_nodes={total_nodes} overflows row_ptr count. Shard the IFDS graph before GPU dispatch."
            ),
        );
    };

    let idx_expr = |p: Expr, b: Expr, f: Expr| {
        Expr::add(
            Expr::add(
                Expr::mul(p, Expr::u32(slots_per_proc)),
                Expr::mul(b, Expr::u32(facts_per_proc)),
            ),
            f,
        )
    };
    let in_proc_block = |p: Expr, b: Expr| {
        Expr::and(
            Expr::lt(p, Expr::u32(num_procs)),
            Expr::lt(b, Expr::u32(blocks_per_proc)),
        )
    };
    let valid_intra = Expr::and(
        in_proc_block(Expr::var("intra_p"), Expr::var("intra_src_b")),
        Expr::lt(Expr::var("intra_dst_b"), Expr::u32(blocks_per_proc)),
    );
    let valid_inter = Expr::and(
        in_proc_block(Expr::var("inter_sp"), Expr::var("inter_sb")),
        in_proc_block(Expr::var("inter_dp"), Expr::var("inter_db")),
    );

    let count_row = |src: Expr| {
        Node::store(
            "row_ptr",
            Expr::add(src.clone(), Expr::u32(1)),
            Expr::add(
                Expr::load("row_ptr", Expr::add(src, Expr::u32(1))),
                Expr::u32(1),
            ),
        )
    };
    let fill_col = |src: Expr, dst: Expr| {
        vec![
            Node::let_bind("emit_slot", Expr::load("row_cursor", src.clone())),
            Node::store("col_idx", Expr::var("emit_slot"), dst),
            Node::store(
                "row_cursor",
                src,
                Expr::add(Expr::var("emit_slot"), Expr::u32(1)),
            ),
        ]
    };

    let kill_scan = vec![
        Node::let_bind("is_killed", Expr::u32(0)),
        Node::loop_for(
            "kill_i",
            Expr::u32(0),
            Expr::u32(kill_count),
            vec![
                Node::let_bind("kill_p", Expr::load("kill_proc", Expr::var("kill_i"))),
                Node::let_bind("kill_b", Expr::load("kill_block", Expr::var("kill_i"))),
                Node::let_bind("kill_f", Expr::load("kill_fact", Expr::var("kill_i"))),
                Node::if_then(
                    Expr::and(
                        Expr::and(
                            Expr::eq(Expr::var("kill_p"), Expr::var("intra_p")),
                            Expr::eq(Expr::var("kill_b"), Expr::var("intra_src_b")),
                        ),
                        Expr::eq(Expr::var("kill_f"), Expr::var("fact")),
                    ),
                    vec![Node::assign("is_killed", Expr::u32(1))],
                ),
            ],
        ),
    ];

    let mut count_intra_fact = kill_scan.clone();
    count_intra_fact.push(Node::if_then(
        Expr::eq(Expr::var("is_killed"), Expr::u32(0)),
        vec![
            Node::let_bind(
                "src_dense",
                idx_expr(
                    Expr::var("intra_p"),
                    Expr::var("intra_src_b"),
                    Expr::var("fact"),
                ),
            ),
            count_row(Expr::var("src_dense")),
        ],
    ));

    let count_gen = vec![
        Node::let_bind("gen_p", Expr::load("gen_proc", Expr::var("gen_i"))),
        Node::let_bind("gen_b", Expr::load("gen_block", Expr::var("gen_i"))),
        Node::let_bind("gen_f", Expr::load("gen_fact", Expr::var("gen_i"))),
        Node::if_then(
            Expr::and(
                Expr::and(
                    Expr::eq(Expr::var("gen_p"), Expr::var("intra_p")),
                    Expr::eq(Expr::var("gen_b"), Expr::var("intra_src_b")),
                ),
                Expr::lt(Expr::var("gen_f"), Expr::u32(facts_per_proc)),
            ),
            vec![
                Node::let_bind(
                    "src_dense",
                    idx_expr(Expr::var("intra_p"), Expr::var("intra_src_b"), Expr::u32(0)),
                ),
                count_row(Expr::var("src_dense")),
            ],
        ),
    ];

    let mut fill_intra_fact = kill_scan;
    fill_intra_fact.push(Node::if_then(
        Expr::eq(Expr::var("is_killed"), Expr::u32(0)),
        {
            let mut nodes = vec![
                Node::let_bind(
                    "src_dense",
                    idx_expr(
                        Expr::var("intra_p"),
                        Expr::var("intra_src_b"),
                        Expr::var("fact"),
                    ),
                ),
                Node::let_bind(
                    "dst_dense",
                    idx_expr(
                        Expr::var("intra_p"),
                        Expr::var("intra_dst_b"),
                        Expr::var("fact"),
                    ),
                ),
            ];
            nodes.extend(fill_col(Expr::var("src_dense"), Expr::var("dst_dense")));
            nodes
        },
    ));

    let fill_gen = vec![
        Node::let_bind("gen_p", Expr::load("gen_proc", Expr::var("gen_i"))),
        Node::let_bind("gen_b", Expr::load("gen_block", Expr::var("gen_i"))),
        Node::let_bind("gen_f", Expr::load("gen_fact", Expr::var("gen_i"))),
        Node::if_then(
            Expr::and(
                Expr::and(
                    Expr::eq(Expr::var("gen_p"), Expr::var("intra_p")),
                    Expr::eq(Expr::var("gen_b"), Expr::var("intra_src_b")),
                ),
                Expr::lt(Expr::var("gen_f"), Expr::u32(facts_per_proc)),
            ),
            {
                let mut nodes = vec![
                    Node::let_bind(
                        "src_dense",
                        idx_expr(Expr::var("intra_p"), Expr::var("intra_src_b"), Expr::u32(0)),
                    ),
                    Node::let_bind(
                        "dst_dense",
                        idx_expr(
                            Expr::var("intra_p"),
                            Expr::var("intra_dst_b"),
                            Expr::var("gen_f"),
                        ),
                    ),
                ];
                nodes.extend(fill_col(Expr::var("src_dense"), Expr::var("dst_dense")));
                nodes
            },
        ),
    ];

    let mut entry = vec![
        Node::loop_for(
            "row_i",
            Expr::u32(0),
            Expr::u32(row_ptr_count),
            vec![Node::store("row_ptr", Expr::var("row_i"), Expr::u32(0))],
        ),
        Node::store("col_len", Expr::u32(0), Expr::u32(0)),
    ];

    entry.push(Node::loop_for(
        "intra_i",
        Expr::u32(0),
        Expr::u32(intra_count),
        vec![
            Node::let_bind("intra_p", Expr::load("intra_proc", Expr::var("intra_i"))),
            Node::let_bind(
                "intra_src_b",
                Expr::load("intra_src_block", Expr::var("intra_i")),
            ),
            Node::let_bind(
                "intra_dst_b",
                Expr::load("intra_dst_block", Expr::var("intra_i")),
            ),
            Node::if_then(
                valid_intra.clone(),
                vec![
                    Node::loop_for(
                        "fact",
                        Expr::u32(0),
                        Expr::u32(facts_per_proc),
                        count_intra_fact,
                    ),
                    Node::loop_for("gen_i", Expr::u32(0), Expr::u32(gen_count), count_gen),
                ],
            ),
        ],
    ));
    entry.push(Node::loop_for(
        "inter_i",
        Expr::u32(0),
        Expr::u32(inter_count),
        vec![
            Node::let_bind(
                "inter_sp",
                Expr::load("inter_src_proc", Expr::var("inter_i")),
            ),
            Node::let_bind(
                "inter_sb",
                Expr::load("inter_src_block", Expr::var("inter_i")),
            ),
            Node::let_bind(
                "inter_dp",
                Expr::load("inter_dst_proc", Expr::var("inter_i")),
            ),
            Node::let_bind(
                "inter_db",
                Expr::load("inter_dst_block", Expr::var("inter_i")),
            ),
            Node::if_then(
                valid_inter.clone(),
                vec![Node::loop_for(
                    "fact",
                    Expr::u32(0),
                    Expr::u32(facts_per_proc),
                    vec![
                        Node::let_bind(
                            "src_dense",
                            idx_expr(
                                Expr::var("inter_sp"),
                                Expr::var("inter_sb"),
                                Expr::var("fact"),
                            ),
                        ),
                        count_row(Expr::var("src_dense")),
                    ],
                )],
            ),
        ],
    ));
    entry.extend([
        Node::let_bind("prefix_sum", Expr::u32(0)),
        Node::loop_for(
            "prefix_row",
            Expr::u32(0),
            Expr::u32(total_nodes),
            vec![
                Node::let_bind(
                    "row_count",
                    Expr::load("row_ptr", Expr::add(Expr::var("prefix_row"), Expr::u32(1))),
                ),
                Node::assign(
                    "prefix_sum",
                    Expr::add(Expr::var("prefix_sum"), Expr::var("row_count")),
                ),
                Node::store(
                    "row_ptr",
                    Expr::add(Expr::var("prefix_row"), Expr::u32(1)),
                    Expr::var("prefix_sum"),
                ),
            ],
        ),
        Node::store("col_len", Expr::u32(0), Expr::var("prefix_sum")),
        Node::loop_for(
            "cursor_row",
            Expr::u32(0),
            Expr::u32(total_nodes),
            vec![Node::store(
                "row_cursor",
                Expr::var("cursor_row"),
                Expr::load("row_ptr", Expr::var("cursor_row")),
            )],
        ),
    ]);
    entry.push(Node::loop_for(
        "intra_i",
        Expr::u32(0),
        Expr::u32(intra_count),
        vec![
            Node::let_bind("intra_p", Expr::load("intra_proc", Expr::var("intra_i"))),
            Node::let_bind(
                "intra_src_b",
                Expr::load("intra_src_block", Expr::var("intra_i")),
            ),
            Node::let_bind(
                "intra_dst_b",
                Expr::load("intra_dst_block", Expr::var("intra_i")),
            ),
            Node::if_then(
                valid_intra,
                vec![
                    Node::loop_for(
                        "fact",
                        Expr::u32(0),
                        Expr::u32(facts_per_proc),
                        fill_intra_fact,
                    ),
                    Node::loop_for("gen_i", Expr::u32(0), Expr::u32(gen_count), fill_gen),
                ],
            ),
        ],
    ));
    entry.push(Node::loop_for(
        "inter_i",
        Expr::u32(0),
        Expr::u32(inter_count),
        vec![
            Node::let_bind(
                "inter_sp",
                Expr::load("inter_src_proc", Expr::var("inter_i")),
            ),
            Node::let_bind(
                "inter_sb",
                Expr::load("inter_src_block", Expr::var("inter_i")),
            ),
            Node::let_bind(
                "inter_dp",
                Expr::load("inter_dst_proc", Expr::var("inter_i")),
            ),
            Node::let_bind(
                "inter_db",
                Expr::load("inter_dst_block", Expr::var("inter_i")),
            ),
            Node::if_then(
                valid_inter,
                vec![Node::loop_for(
                    "fact",
                    Expr::u32(0),
                    Expr::u32(facts_per_proc),
                    {
                        let mut nodes = vec![
                            Node::let_bind(
                                "src_dense",
                                idx_expr(
                                    Expr::var("inter_sp"),
                                    Expr::var("inter_sb"),
                                    Expr::var("fact"),
                                ),
                            ),
                            Node::let_bind(
                                "dst_dense",
                                idx_expr(
                                    Expr::var("inter_dp"),
                                    Expr::var("inter_db"),
                                    Expr::var("fact"),
                                ),
                            ),
                        ];
                        nodes.extend(fill_col(Expr::var("src_dense"), Expr::var("dst_dense")));
                        nodes
                    },
                )],
            ),
        ],
    ));

    Program::wrapped(
        vec![
            BufferDecl::storage("intra_proc", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(intra_count.max(1)),
            BufferDecl::storage("intra_src_block", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(intra_count.max(1)),
            BufferDecl::storage("intra_dst_block", 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(intra_count.max(1)),
            BufferDecl::storage("inter_src_proc", 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(inter_count.max(1)),
            BufferDecl::storage("inter_src_block", 4, BufferAccess::ReadOnly, DataType::U32)
                .with_count(inter_count.max(1)),
            BufferDecl::storage("inter_dst_proc", 5, BufferAccess::ReadOnly, DataType::U32)
                .with_count(inter_count.max(1)),
            BufferDecl::storage("inter_dst_block", 6, BufferAccess::ReadOnly, DataType::U32)
                .with_count(inter_count.max(1)),
            BufferDecl::storage("gen_proc", 7, BufferAccess::ReadOnly, DataType::U32)
                .with_count(gen_count.max(1)),
            BufferDecl::storage("gen_block", 8, BufferAccess::ReadOnly, DataType::U32)
                .with_count(gen_count.max(1)),
            BufferDecl::storage("gen_fact", 9, BufferAccess::ReadOnly, DataType::U32)
                .with_count(gen_count.max(1)),
            BufferDecl::storage("kill_proc", 10, BufferAccess::ReadOnly, DataType::U32)
                .with_count(kill_count.max(1)),
            BufferDecl::storage("kill_block", 11, BufferAccess::ReadOnly, DataType::U32)
                .with_count(kill_count.max(1)),
            BufferDecl::storage("kill_fact", 12, BufferAccess::ReadOnly, DataType::U32)
                .with_count(kill_count.max(1)),
            BufferDecl::storage("row_ptr", 13, BufferAccess::ReadWrite, DataType::U32)
                .with_count(row_ptr_count),
            BufferDecl::storage("row_cursor", 14, BufferAccess::ReadWrite, DataType::U32)
                .with_count(total_nodes.max(1)),
            BufferDecl::storage("col_idx", 15, BufferAccess::ReadWrite, DataType::U32)
                .with_count(max_col_count.max(1)),
            BufferDecl::storage("col_len", 16, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [1, 1, 1],
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new(vec![Node::if_then(
                Expr::eq(Expr::gid_x(), Expr::u32(0)),
                entry,
            )]),
        }],
    )
}

/// Pack a `(proc_id, block_id, fact_id)` triple into a 32-bit
/// node id.
///
/// Invalid triples have no non-aliasing `u32` representation, so the
/// failure is explicit instead of silently clamping or masking.
#[must_use]
pub fn encode_node(proc_id: u32, block_id: u32, fact_id: u32) -> Option<u32> {
    fits(proc_id, block_id, fact_id)
        .then_some((proc_id << PROC_SHIFT) | (block_id << BLOCK_SHIFT) | fact_id)
}

/// Unpack a node id back into `(proc_id, block_id, fact_id)`.
#[must_use]
pub fn decode_node(node_id: u32) -> (u32, u32, u32) {
    let proc_id = (node_id >> PROC_SHIFT) & PROC_MASK;
    let block_id = (node_id >> BLOCK_SHIFT) & BLOCK_MASK;
    let fact_id = node_id & FACT_MASK;
    (proc_id, block_id, fact_id)
}

/// Whether a `(proc, block, fact)` triple fits in the packed
/// 32-bit representation. Callers on the production path should
/// verify this before calling [`encode_node`].
#[must_use]
pub fn fits(proc_id: u32, block_id: u32, fact_id: u32) -> bool {
    proc_id <= MAX_PROC_ID && block_id <= MAX_BLOCK_ID && fact_id <= MAX_FACT_ID
}

/// Recover the exploded IFDS program cache key baked into a generated CSR
/// builder [`Program`].
///
/// Test and parity dispatchers use this to route GPU-shaped byte inputs through
/// the CPU reference without re-deriving dimensions from padded buffers alone.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn ifds_program_cache_key_from_program(
    program: &Program,
) -> Result<IfdsCsrProgramCacheKey, String> {
    let intra_count = loop_upper_bound(program, "intra_i")
        .ok_or_else(|| "Fix: exploded IFDS program missing intra_i loop bound.".to_string())?;
    let inter_count = loop_upper_bound(program, "inter_i")
        .ok_or_else(|| "Fix: exploded IFDS program missing inter_i loop bound.".to_string())?;
    let gen_count = loop_upper_bound(program, "gen_i")
        .ok_or_else(|| "Fix: exploded IFDS program missing gen_i loop bound.".to_string())?;
    let kill_count = loop_upper_bound(program, "kill_i")
        .ok_or_else(|| "Fix: exploded IFDS program missing kill_i loop bound.".to_string())?;
    let facts_per_proc = loop_upper_bound(program, "fact")
        .ok_or_else(|| "Fix: exploded IFDS program missing fact loop bound.".to_string())?;
    let total_nodes = loop_upper_bound(program, "prefix_row")
        .or_else(|| loop_upper_bound(program, "cursor_row"))
        .ok_or_else(|| "Fix: exploded IFDS program missing total_nodes loop bound.".to_string())?;

    let num_procs = upper_limit_for_var(program, "intra_p")
        .or_else(|| upper_limit_for_var(program, "inter_sp"))
        .ok_or_else(|| "Fix: exploded IFDS program missing num_procs bound.".to_string())?;
    let blocks_per_proc = upper_limit_for_var(program, "intra_dst_b")
        .or_else(|| upper_limit_for_var(program, "intra_src_b"))
        .or_else(|| upper_limit_for_var(program, "inter_sb"))
        .ok_or_else(|| "Fix: exploded IFDS program missing blocks_per_proc bound.".to_string())?;

    let slots_per_proc = blocks_per_proc
        .checked_mul(facts_per_proc)
        .ok_or_else(|| "Fix: exploded IFDS blocks*facts overflowed u32.".to_string())?;
    let expected_total = num_procs
        .checked_mul(slots_per_proc)
        .ok_or_else(|| "Fix: exploded IFDS procs*blocks*facts overflowed u32.".to_string())?;
    if expected_total != total_nodes {
        return Err(format!(
            "Fix: exploded IFDS program shape mismatch: procs={num_procs} blocks={blocks_per_proc} facts={facts_per_proc} implies total_nodes={expected_total}, program loop bound={total_nodes}."
        ));
    }

    let max_col_count = max_ifds_col_count(intra_count, inter_count, gen_count, facts_per_proc)
        .ok_or_else(|| "Fix: exploded IFDS maximum column count overflowed u32.".to_string())?;

    Ok(IfdsCsrProgramCacheKey {
        num_procs,
        blocks_per_proc,
        facts_per_proc,
        intra_count,
        inter_count,
        gen_count,
        kill_count,
        max_col_count,
    })
}

#[cfg(any(test, feature = "cpu-parity"))]
fn loop_upper_bound(program: &Program, var: &str) -> Option<u32> {
    use vyre_foundation::transform::visit::walk_nodes;

    let mut found: Option<u32> = None;
    walk_nodes(program, |node| {
        if let Node::Loop {
            var: loop_var,
            to,
            ..
        } = node
        {
            if loop_var.as_str() == var {
                if let Expr::LitU32(limit) = to {
                    found = Some(*limit);
                }
            }
        }
    });
    found
}

#[cfg(any(test, feature = "cpu-parity"))]
fn upper_limit_for_var(program: &Program, var: &str) -> Option<u32> {
    use vyre_foundation::ir::BinOp;
    use vyre_foundation::transform::visit::walk_exprs;

    let mut found: Option<u32> = None;
    walk_exprs(program, |expr| {
        if let Expr::BinOp {
            op: BinOp::Lt,
            left,
            right,
        } = expr
        {
            if let (Expr::Var(name), Expr::LitU32(limit)) = (left.as_ref(), right.as_ref()) {
                if name.as_str() == var {
                    found = Some(*limit);
                }
            }
        }
    });
    found
}

/// CPU-reference CSR builder for the exploded supergraph.
///
/// `intra_edges` are `(src_block, dst_block)` pairs **within** a
/// procedure  -  the standard CFG. `inter_edges` are `(src_proc,
/// src_block, dst_proc, dst_block)` call / return edges. Flow
/// functions are encoded as per-block GEN / KILL bitsets over the
/// fact domain.
///
/// Caller-owned workspace for exploded IFDS CPU-reference CSR construction.
#[cfg(any(test, feature = "cpu-parity"))]
#[derive(Debug, Default, Clone)]
pub struct ExplodedIfdsCpuScratch {
    /// Flat `(src, dst)` edge list before CSR compaction.
    pub edges_flat: Vec<(u32, u32)>,
    /// Per-dense-node KILL bitmap.
    pub killed: Vec<bool>,
    /// Per-block GEN prefix offsets.
    pub gen_offsets: Vec<usize>,
    /// Per-block GEN fill cursor.
    pub gen_cursor: Vec<usize>,
    /// Flat GEN fact table keyed by `gen_offsets`.
    pub gen_facts: Vec<u32>,
    /// Per-row CSR fill cursor.
    pub cursor: Vec<usize>,
}

#[cfg(any(test, feature = "cpu-parity"))]
impl ExplodedIfdsCpuScratch {
    /// Create an empty reusable exploded-IFDS CPU workspace.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Returns `(row_ptr, col_idx)` in the **dense** index space
/// `idx(p, b, f) = p * blocks * facts + b * facts + f`. This is
/// the space every traversal kernel operates in  -  packing via
/// [`encode_node`] is only used at the I/O boundary when the
/// caller needs to report results as `(proc, block, fact)`
/// triples. The two spaces coincide only in the degenerate case
/// `blocks_per_proc == 1 << BLOCK_BITS` and `facts_per_proc == 1 << FACT_BITS`;
/// the dense layout works for any dimensions that fit in
/// 32-bit encoding.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn build_cpu_reference(
    num_procs: u32,
    blocks_per_proc: u32,
    facts_per_proc: u32,
    intra_edges: &[(u32, u32, u32)], // (proc, src_block, dst_block)
    inter_edges: &[(u32, u32, u32, u32)], // (src_proc, src_block, dst_proc, dst_block)
    flow_gen: &[(u32, u32, u32)],    // (proc, block, fact)  -  GEN bits
    flow_kill: &[(u32, u32, u32)],   // (proc, block, fact)  -  KILL bits
) -> (Vec<u32>, Vec<u32>) {
    try_build_cpu_reference(
        num_procs,
        blocks_per_proc,
        facts_per_proc,
        intra_edges,
        inter_edges,
        flow_gen,
        flow_kill,
    )
    .unwrap_or_else(|err| panic!("exploded IFDS CPU reference received malformed input. {err}"))
}

/// Fallible CPU-reference CSR builder for the exploded supergraph.
///
/// This is the allocation-safe oracle entry point for fuzz, hostile-dimension,
/// and parity harnesses. [`build_cpu_reference`] preserves the legacy panicking
/// API by delegating here.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_build_cpu_reference(
    num_procs: u32,
    blocks_per_proc: u32,
    facts_per_proc: u32,
    intra_edges: &[(u32, u32, u32)],
    inter_edges: &[(u32, u32, u32, u32)],
    flow_gen: &[(u32, u32, u32)],
    flow_kill: &[(u32, u32, u32)],
) -> Result<(Vec<u32>, Vec<u32>), String> {
    let mut row_ptr = Vec::new();
    let mut col_idx = Vec::new();
    let mut scratch = ExplodedIfdsCpuScratch::default();
    try_build_cpu_reference_into(
        num_procs,
        blocks_per_proc,
        facts_per_proc,
        intra_edges,
        inter_edges,
        flow_gen,
        flow_kill,
        &mut row_ptr,
        &mut col_idx,
        &mut scratch,
    )?;
    Ok((row_ptr, col_idx))
}

/// Fallible CPU-reference CSR builder into caller-owned output and scratch.
///
/// Validation happens before output and scratch storage are cleared. This keeps
/// fuzz/parity diagnostics intact when a malformed IFDS domain or rule set is
/// rejected before CSR construction begins.
#[cfg(any(test, feature = "cpu-parity"))]
#[allow(clippy::too_many_arguments)]
pub fn try_build_cpu_reference_into(
    num_procs: u32,
    blocks_per_proc: u32,
    facts_per_proc: u32,
    intra_edges: &[(u32, u32, u32)],
    inter_edges: &[(u32, u32, u32, u32)],
    flow_gen: &[(u32, u32, u32)],
    flow_kill: &[(u32, u32, u32)],
    row_ptr: &mut Vec<u32>,
    col_idx: &mut Vec<u32>,
    scratch: &mut ExplodedIfdsCpuScratch,
) -> Result<(), String> {
    let layout = validate_ifds_csr_inputs(
        num_procs,
        blocks_per_proc,
        facts_per_proc,
        intra_edges,
        inter_edges,
        flow_gen,
        flow_kill,
    )?;
    if layout.empty {
        return Err(format!(
            "exploded IFDS CPU reference dimensions must be nonzero, got procs={num_procs}, blocks={blocks_per_proc}, facts={facts_per_proc}. Fix: pass a real exploded-supergraph domain before parity comparison."
        ));
    }

    // PHASE7_GRAPH C4: every multiply checked. The previous unchecked
    // chain (`blocks * facts`, then `procs * slots`) wraps silently
    // when the caller passes the maximum dimensions for each field
    // (4096 × 1024 × 1024 = 2^32 = wraps to 0 on 32-bit usize and
    // sits exactly at the overflow boundary on 64-bit). Either case
    // produced a tiny `Vec<Vec<u32>>` and catastrophic OOB writes in
    // the edge-emit loops below.
    let slots_per_proc = layout.slots_per_proc as usize;
    let total_nodes = layout.total_nodes as usize;
    crate::graph::scratch::reserve_graph_items(
        &mut scratch.edges_flat,
        layout.max_col_count as usize,
        "exploded IFDS CPU reference",
        "flat exploded edge list",
    )?;
    scratch.edges_flat.clear();
    let block_count = (num_procs as usize) * (blocks_per_proc as usize);

    let idx = |p: u32, b: u32, f: u32| -> u32 {
        ((p as usize) * slots_per_proc + (b as usize) * facts_per_proc as usize + f as usize) as u32
    };
    let block_idx =
        |p: u32, b: u32| -> usize { (p as usize) * blocks_per_proc as usize + b as usize };
    let in_space =
        |p: u32, b: u32, f: u32| p < num_procs && b < blocks_per_proc && f < facts_per_proc;

    crate::graph::scratch::reserve_graph_items(
        &mut scratch.killed,
        total_nodes,
        "exploded IFDS CPU reference",
        "kill bitmap",
    )?;
    scratch.killed.clear();
    scratch.killed.resize(total_nodes, false);
    for &(p, b, f) in flow_kill {
        if in_space(p, b, f) {
            scratch.killed[idx(p, b, f) as usize] = true;
        }
    }

    let gen_offset_count = block_count
        .checked_add(1)
        .ok_or_else(|| "Fix: exploded IFDS block_count+1 overflows usize.".to_string())?;
    crate::graph::scratch::reserve_graph_items(
        &mut scratch.gen_offsets,
        gen_offset_count,
        "exploded IFDS CPU reference",
        "GEN offsets",
    )?;
    scratch.gen_offsets.clear();
    scratch.gen_offsets.resize(gen_offset_count, 0);
    for &(p, b, f) in flow_gen {
        if in_space(p, b, f) {
            scratch.gen_offsets[block_idx(p, b) + 1] += 1;
        }
    }
    for i in 1..scratch.gen_offsets.len() {
        scratch.gen_offsets[i] += scratch.gen_offsets[i - 1];
    }
    crate::graph::scratch::reserve_graph_items(
        &mut scratch.gen_cursor,
        block_count,
        "exploded IFDS CPU reference",
        "GEN cursor",
    )?;
    scratch.gen_cursor.clear();
    scratch
        .gen_cursor
        .extend_from_slice(&scratch.gen_offsets[..block_count]);
    let gen_fact_count = scratch.gen_offsets[block_count];
    crate::graph::scratch::reserve_graph_items(
        &mut scratch.gen_facts,
        gen_fact_count,
        "exploded IFDS CPU reference",
        "GEN fact table",
    )?;
    scratch.gen_facts.clear();
    scratch.gen_facts.resize(gen_fact_count, 0);
    for &(p, b, f) in flow_gen {
        if in_space(p, b, f) {
            let key = block_idx(p, b);
            let slot = scratch.gen_cursor[key];
            scratch.gen_facts[slot] = f;
            scratch.gen_cursor[key] += 1;
        }
    }

    // Intra-procedural CFG edges, cross-producted with fact-propagation:
    // an edge (B_src -> B_dst) gives rise to an edge in the exploded
    // supergraph between every pair (f, f) that survives the flow
    // function at B_src (fact f propagates iff f is not killed).
    for &(p, src_b, dst_b) in intra_edges {
        if p >= num_procs || src_b >= blocks_per_proc || dst_b >= blocks_per_proc {
            continue;
        }
        for f in 0..facts_per_proc {
            if scratch.killed[idx(p, src_b, f) as usize] {
                continue;
            }
            scratch
                .edges_flat
                .push((idx(p, src_b, f), idx(p, dst_b, f)));
        }
        // GEN edges: standard IFDS 0-fact encoding  -  fact 0 is the
        // tautological "always present" fact. `GEN(src_b, gf)` emits
        // edge `(src_b, 0) → (dst_b, gf)`, so seeding `(entry, 0)`
        // triggers every GEN along the reachable CFG. Callers that
        // don't use the 0-fact convention see GEN as a no-op.
        let gen_key = block_idx(p, src_b);
        for &gf in
            &scratch.gen_facts[scratch.gen_offsets[gen_key]..scratch.gen_offsets[gen_key + 1]]
        {
            scratch
                .edges_flat
                .push((idx(p, src_b, 0), idx(p, dst_b, gf)));
        }
    }

    // Inter-procedural call / return edges propagate every fact
    // (IFDS handles parameter mapping via summary edges in the full
    // algorithm; this CPU reference is the unfiltered
    // "every-fact-flows" upper bound used for correctness tests).
    for &(sp, sb, dp, db) in inter_edges {
        if sp >= num_procs || dp >= num_procs || sb >= blocks_per_proc || db >= blocks_per_proc {
            continue;
        }
        for f in 0..facts_per_proc {
            scratch.edges_flat.push((idx(sp, sb, f), idx(dp, db, f)));
        }
    }

    // Flatten into CSR  -  row_ptr has total_nodes+1 entries.
    if scratch.edges_flat.len() > u32::MAX as usize {
        return Err(format!(
            "exploded IFDS CPU reference edge_count={} exceeds u32 CSR encoding. Fix: shard the IFDS graph before parity comparison.",
            scratch.edges_flat.len()
        ));
    }
    let row_ptr_len = layout.row_words;
    crate::graph::scratch::reserve_graph_items(
        row_ptr,
        row_ptr_len,
        "exploded IFDS CPU reference",
        "CSR row_ptr",
    )?;
    row_ptr.clear();
    row_ptr.resize(row_ptr_len, 0);
    for &(src, _) in &scratch.edges_flat {
        let row = src as usize;
        row_ptr[row + 1] = row_ptr[row + 1].checked_add(1).ok_or_else(|| {
            format!(
                "exploded IFDS CPU reference row {row} edge count overflowed u32. Fix: shard the IFDS graph before parity comparison."
            )
        })?;
    }
    for row in 1..row_ptr.len() {
        row_ptr[row] = row_ptr[row].checked_add(row_ptr[row - 1]).ok_or_else(|| {
            format!(
                "exploded IFDS CPU reference CSR prefix overflowed at row {row}. Fix: shard the IFDS graph before parity comparison."
            )
        })?;
    }
    crate::graph::scratch::reserve_graph_items(
        &mut scratch.cursor,
        total_nodes,
        "exploded IFDS CPU reference",
        "CSR cursor",
    )?;
    scratch.cursor.clear();
    for &offset in &row_ptr[..total_nodes] {
        scratch.cursor.push(offset as usize);
    }
    crate::graph::scratch::reserve_graph_items(
        col_idx,
        scratch.edges_flat.len(),
        "exploded IFDS CPU reference",
        "CSR col_idx",
    )?;
    col_idx.clear();
    col_idx.resize(scratch.edges_flat.len(), 0);
    for &(src, dst) in &scratch.edges_flat {
        let row = src as usize;
        let slot = scratch.cursor[row];
        col_idx[slot] = dst;
        scratch.cursor[row] += 1;
    }
    Ok(())
}

#[cfg(test)]
mod generated_exploded_cpu_reference_tests {
    use super::*;

    #[test]
    fn generated_try_build_cpu_reference_emits_valid_csr_shapes() {
        for procs in 1u32..=4 {
            for blocks in 1u32..=16 {
                for facts in 1u32..=16 {
                    let intra: Vec<(u32, u32, u32)> = (0..procs)
                        .flat_map(|proc_id| {
                            (0..blocks.saturating_sub(1))
                                .map(move |block| (proc_id, block, block + 1))
                        })
                        .collect();
                    let inter: Vec<(u32, u32, u32, u32)> = if procs > 1 {
                        (0..procs - 1)
                            .map(|proc_id| (proc_id, blocks - 1, proc_id + 1, 0))
                            .collect()
                    } else {
                        Vec::new()
                    };
                    let gen_rules: Vec<(u32, u32, u32)> = (0..procs)
                        .flat_map(|proc_id| {
                            (0..blocks).filter_map(move |block| {
                                if facts > 1 {
                                    Some((proc_id, block, (block % (facts - 1)) + 1))
                                } else {
                                    None
                                }
                            })
                        })
                        .collect();
                    let kill_rules: Vec<(u32, u32, u32)> = (0..procs)
                        .flat_map(|proc_id| {
                            (0..blocks).filter_map(move |block| {
                                (facts > 2 && block % 3 == 0).then_some((proc_id, block, 1))
                            })
                        })
                        .collect();
                    let (row_ptr, col_idx) = try_build_cpu_reference(
                        procs,
                        blocks,
                        facts,
                        &intra,
                        &inter,
                        &gen_rules,
                        &kill_rules,
                    )
                    .unwrap();
                    let total_nodes = procs as usize * blocks as usize * facts as usize;
                    assert_eq!(row_ptr.len(), total_nodes + 1);
                    assert_eq!(row_ptr[total_nodes] as usize, col_idx.len());
                    for window in row_ptr.windows(2) {
                        assert!(window[0] <= window[1]);
                    }
                    for &dst in &col_idx {
                        assert!((dst as usize) < total_nodes);
                    }
                }
            }
        }
    }

    #[test]
    fn try_build_cpu_reference_rejects_empty_domain_without_panicking() {
        let err = try_build_cpu_reference(0, 0, 0, &[], &[], &[], &[]).unwrap_err();
        assert!(err.contains("nonzero"));
    }

    #[test]
    fn try_build_cpu_reference_into_reuses_output_and_workspace() {
        let mut row_ptr = Vec::with_capacity(32);
        row_ptr.extend_from_slice(&[9, 8, 7]);
        let mut col_idx = Vec::with_capacity(32);
        col_idx.extend_from_slice(&[6, 5, 4]);
        let mut scratch = ExplodedIfdsCpuScratch {
            edges_flat: Vec::with_capacity(32),
            killed: Vec::with_capacity(32),
            gen_offsets: Vec::with_capacity(16),
            gen_cursor: Vec::with_capacity(16),
            gen_facts: Vec::with_capacity(16),
            cursor: Vec::with_capacity(32),
        };
        scratch.edges_flat.extend_from_slice(&[(99, 98), (97, 96)]);
        scratch.killed.extend_from_slice(&[true, true]);
        scratch.gen_offsets.extend_from_slice(&[11, 12]);
        scratch.gen_cursor.extend_from_slice(&[13, 14]);
        scratch.gen_facts.extend_from_slice(&[15, 16]);
        scratch.cursor.extend_from_slice(&[17, 18]);
        let capacities = (
            row_ptr.capacity(),
            col_idx.capacity(),
            scratch.edges_flat.capacity(),
            scratch.killed.capacity(),
            scratch.gen_offsets.capacity(),
            scratch.gen_cursor.capacity(),
            scratch.gen_facts.capacity(),
            scratch.cursor.capacity(),
        );

        let expected = build_cpu_reference(1, 2, 4, &[(0, 0, 1)], &[], &[(0, 0, 2)], &[(0, 0, 3)]);
        try_build_cpu_reference_into(
            1,
            2,
            4,
            &[(0, 0, 1)],
            &[],
            &[(0, 0, 2)],
            &[(0, 0, 3)],
            &mut row_ptr,
            &mut col_idx,
            &mut scratch,
        )
        .expect("Fix: valid exploded IFDS graph must build with reusable workspace.");

        assert_eq!((row_ptr.clone(), col_idx.clone()), expected);
        assert_eq!(
            (
                row_ptr.capacity(),
                col_idx.capacity(),
                scratch.edges_flat.capacity(),
                scratch.killed.capacity(),
                scratch.gen_offsets.capacity(),
                scratch.gen_cursor.capacity(),
                scratch.gen_facts.capacity(),
                scratch.cursor.capacity(),
            ),
            capacities
        );

        try_build_cpu_reference_into(
            1,
            1,
            1,
            &[],
            &[],
            &[],
            &[],
            &mut row_ptr,
            &mut col_idx,
            &mut scratch,
        )
        .expect("Fix: smaller exploded IFDS graph must reuse the same workspace.");

        assert_eq!(row_ptr, vec![0, 0]);
        assert!(col_idx.is_empty());
        assert_eq!(
            (
                row_ptr.capacity(),
                col_idx.capacity(),
                scratch.edges_flat.capacity(),
                scratch.killed.capacity(),
                scratch.gen_offsets.capacity(),
                scratch.gen_cursor.capacity(),
                scratch.gen_facts.capacity(),
                scratch.cursor.capacity(),
            ),
            capacities
        );
    }

    #[test]
    fn try_build_cpu_reference_into_validates_before_mutating_storage() {
        let mut row_ptr = vec![9, 8, 7];
        let mut col_idx = vec![6, 5, 4];
        let mut scratch = ExplodedIfdsCpuScratch {
            edges_flat: vec![(1, 2)],
            killed: vec![true],
            gen_offsets: vec![3],
            gen_cursor: vec![4],
            gen_facts: vec![5],
            cursor: vec![6],
        };

        let err = try_build_cpu_reference_into(
            0,
            0,
            0,
            &[],
            &[],
            &[],
            &[],
            &mut row_ptr,
            &mut col_idx,
            &mut scratch,
        )
        .expect_err("Fix: empty exploded IFDS domain must be rejected.");

        assert!(err.contains("nonzero"));
        assert_eq!(row_ptr, vec![9, 8, 7]);
        assert_eq!(col_idx, vec![6, 5, 4]);
        assert_eq!(scratch.edges_flat, vec![(1, 2)]);
        assert_eq!(scratch.killed, vec![true]);
        assert_eq!(scratch.gen_offsets, vec![3]);
        assert_eq!(scratch.gen_cursor, vec![4]);
        assert_eq!(scratch.gen_facts, vec![5]);
        assert_eq!(scratch.cursor, vec![6]);
    }

    #[test]
    fn generated_try_build_cpu_reference_into_matches_allocating_reference() {
        let mut row_ptr = Vec::new();
        let mut col_idx = Vec::new();
        let mut scratch = ExplodedIfdsCpuScratch::new();

        for case in 0..1024usize {
            let num_procs = 1 + (case % 3) as u32;
            let blocks_per_proc = 1 + ((case / 3) % 5) as u32;
            let facts_per_proc = 1 + ((case / 15) % 5) as u32;
            let mut intra_edges = Vec::new();
            let mut inter_edges = Vec::new();
            let mut flow_gen = Vec::new();
            let mut flow_kill = Vec::new();

            for p in 0..num_procs {
                for b in 0..blocks_per_proc {
                    let next_b = (b + 1) % blocks_per_proc;
                    let mixed = case
                        .wrapping_mul(37)
                        .wrapping_add((p as usize).wrapping_mul(11))
                        .wrapping_add((b as usize).wrapping_mul(7));
                    if blocks_per_proc > 1 && mixed % 2 == 0 {
                        intra_edges.push((p, b, next_b));
                    }
                    let fact = (mixed as u32) % facts_per_proc;
                    if mixed % 3 == 0 {
                        flow_gen.push((p, b, fact));
                    }
                    if mixed % 5 == 0 && fact != 0 {
                        flow_kill.push((p, b, fact));
                    }
                }
            }
            if num_procs > 1 {
                for p in 0..num_procs - 1 {
                    if (case + p as usize) % 2 == 0 {
                        inter_edges.push((p, 0, p + 1, 0));
                    }
                }
            }

            let expected = try_build_cpu_reference(
                num_procs,
                blocks_per_proc,
                facts_per_proc,
                &intra_edges,
                &inter_edges,
                &flow_gen,
                &flow_kill,
            )
            .expect("Fix: generated exploded IFDS graph must build through allocating oracle.");
            try_build_cpu_reference_into(
                num_procs,
                blocks_per_proc,
                facts_per_proc,
                &intra_edges,
                &inter_edges,
                &flow_gen,
                &flow_kill,
                &mut row_ptr,
                &mut col_idx,
                &mut scratch,
            )
            .expect("Fix: generated exploded IFDS graph must build through reusable oracle.");
            assert_eq!(
                (row_ptr.clone(), col_idx.clone()),
                expected,
                "Fix: reusable exploded IFDS oracle diverged at generated case {case}."
            );
        }
    }
}

/// Convert a dense `(proc, block, fact)` index  -  the space
/// [`build_cpu_reference`] operates in  -  into the packed
/// [`encode_node`] form for reporting or cross-subsystem handoff.
#[must_use]
pub fn dense_to_encoded(dense: u32, blocks_per_proc: u32, facts_per_proc: u32) -> Option<u32> {
    let slots_per_proc = blocks_per_proc.checked_mul(facts_per_proc)?;
    if slots_per_proc == 0 {
        return None;
    }
    let p = dense / slots_per_proc;
    let within_proc = dense % slots_per_proc;
    let b = within_proc / facts_per_proc;
    let f = within_proc % facts_per_proc;
    encode_node(p, b, f)
}

/// Inverse of [`dense_to_encoded`].
#[must_use]
pub fn encoded_to_dense(node_id: u32, blocks_per_proc: u32, facts_per_proc: u32) -> Option<u32> {
    let (p, b, f) = decode_node(node_id);
    let proc_span = blocks_per_proc.checked_mul(facts_per_proc)?;
    let proc_offset = p.checked_mul(proc_span)?;
    let block_offset = b.checked_mul(facts_per_proc)?;
    proc_offset.checked_add(block_offset)?.checked_add(f)
}

#[cfg(test)]
mod tests;
