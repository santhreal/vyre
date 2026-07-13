//! Decode → scan fusion optimizer pass (G5).
//!
//! # Idea
//!
//! When a single Program already contains both a decoder and a
//! scanner  -  the decoder writes some `ReadWrite` storage handoff
//! buffer, the scanner then reads from it  -  the decoded bytes
//! don't need to round-trip through DRAM. Promoting the handoff
//! to workgroup memory keeps the bytes in the SM's shared
//! scratchpad and lets the scanner hit L1 instead of HBM.
//!
//! The companion library API in
//! `vyre_libs::decode::streaming::fuse_decode_scan` does the
//! same transform for a *pair* of Programs (separately-owned
//! decoder + scanner); this pass handles the pre-fused case that
//! already lives in one `Program`.
//!
//! # Transform
//!
//! For every buffer `b` where:
//!   * `b.access() == BufferAccess::ReadWrite` (written then read),
//!   * `b.count() > 0` (static size known  -  workgroup memory
//!     requires a compile-time count), and
//!   * `b` is not marked `pipeline_live_out` (a workgroup buffer
//!     cannot be observed outside the dispatch),
//!
//! the pass rewrites `b` in-place to
//! `BufferDecl::workgroup(name, count, element)`  -  the access mode
//! flips to `Workgroup`, the memory tier flips to `Shared`, and
//! the binding slot is dropped (workgroup buffers do not hold a
//! `@binding`). Entry-body node ops reference buffers by name, so
//! no body rewriting is required.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ir::{BufferAccess, BufferDecl, DataType, Ident, Node, Program};
use crate::optimizer::{fingerprint_program, vyre_pass, PassAnalysis, PassResult};

/// Conservative ceiling on workgroup-promoted buffer size.
///
/// vyre-driver's `DeviceCaps::wgpu_like_default` reports 16 KiB of
/// shared memory on the wgpu fallback path; CUDA/SPIR-V get 48 KiB+.
/// Without a target backend at this stage, we use the wgpu floor so a
/// program that compiles after this pass on any reachable backend.
const MAX_WORKGROUP_PROMOTION_BYTES: u64 = 16 * 1024;

/// Bytes-per-element for the destination workgroup buffer. Delegates
/// to the canonical [`DataType::size_bytes`] table so every variant
/// (U8/I8/Bool/Bytes = 1, U16/I16/F16/BF16 = 2, U32/I32/F32 = 4,
/// U64/I64/F64/Vec2U32 = 8, `Vec4U32` = 16, Vec/Array follow the
/// element/lane math, F8/F4/I4/NF4 = 1) is sized correctly.
///
/// `size_bytes` returns None for dynamically-sized variants (Tensor,
/// `TensorShaped`, SparseCsr/SparseCoo/SparseBsr, Opaque). Those cannot
/// be promoted to fixed-size workgroup storage because any guessed size
/// can understate shared-memory pressure and corrupt dispatch layout.
fn element_bytes(element: &DataType) -> Option<u64> {
    element.size_bytes().map(|bytes| bytes as u64)
}

fn fits_workgroup_budget(buf: &BufferDecl) -> bool {
    let Some(element_bytes) = element_bytes(&buf.element()) else {
        return false;
    };
    let Some(bytes) = u64::from(buf.count()).checked_mul(element_bytes) else {
        return false;
    };
    bytes > 0 && bytes <= MAX_WORKGROUP_PROMOTION_BYTES
}

/// The single promotability predicate shared by [`run`], [`count_opportunities`]
/// and [`candidate_handoffs`] so all three agree exactly on what is a handoff.
///
/// A buffer is a promotable handoff when it is `ReadWrite` (written then read),
/// statically sized (`count > 0`: workgroup allocations must be compile-time
/// sized), not externally observed (`!pipeline_live_out`: workgroup buffers do
/// not survive past dispatch end), fits the workgroup byte budget, AND is not
/// referenced by a cross-workgroup op (see [`cross_workgroup_buffers`]).
fn is_promotable_handoff(buf: &BufferDecl, cross_workgroup: &FxHashSet<Ident>) -> bool {
    buf.access() == BufferAccess::ReadWrite
        && buf.count() > 0
        && !buf.is_pipeline_live_out()
        && fits_workgroup_budget(buf)
        && !cross_workgroup.contains(&Ident::from(buf.name()))
}

/// Collect every buffer referenced by a CROSS-WORKGROUP op anywhere in `nodes`.
///
/// Collectives (`AllReduce`/`Broadcast`/`AllGather`/`ReduceScatter`) move data
/// between workgroups over a `CommGroup`. Workgroup memory is per-workgroup-
/// private, so promoting such a buffer would give each workgroup its own copy
/// and silently destroy the cross-workgroup dataflow (the reduction/gather/
/// broadcast would see only one workgroup's data). `decode_scan_fuse` must
/// therefore never promote a buffer that any collective touches, its decl
/// alone (ReadWrite + sized + not live-out) cannot reveal this; only the body
/// can. (The promotion precondition already excludes externally-observed
/// buffers; a collective is an in-program cross-workgroup observation the
/// `pipeline_live_out` flag does not cover.)
fn cross_workgroup_buffers(nodes: &[Node], out: &mut FxHashSet<Ident>) {
    for node in nodes {
        match node {
            Node::AllReduce { buffer, .. } | Node::Broadcast { buffer, .. } => {
                out.insert(buffer.clone());
            }
            Node::AllGather { input, output, .. } | Node::ReduceScatter { input, output, .. } => {
                out.insert(input.clone());
                out.insert(output.clone());
            }
            Node::If {
                then, otherwise, ..
            } => {
                cross_workgroup_buffers(then, out);
                cross_workgroup_buffers(otherwise, out);
            }
            Node::Loop { body, .. } | Node::Block(body) => cross_workgroup_buffers(body, out),
            Node::Region { body, .. } => cross_workgroup_buffers(body, out),
            _ => {}
        }
    }
}

/// Built-in optimizer pass for in-program decode/scan handoff fusion.
#[derive(Debug, Default)]
#[vyre_pass(
    name = "decode_scan_fuse",
    requires = [],
    invalidates = ["buffer_layout", "fusion"]
)]
pub struct DecodeScanFuse;

impl DecodeScanFuse {
    /// Run only when a program has at least one promotable handoff buffer.
    #[must_use]
    fn analyze_impl(program: &Program) -> PassAnalysis {
        if count_opportunities(program) == 0 {
            PassAnalysis::SKIP
        } else {
            PassAnalysis::RUN
        }
    }

    /// Promote storage handoff buffers to workgroup memory.
    #[must_use]
    pub fn transform(program: Program) -> PassResult {
        let before = fingerprint_program(&program);
        let optimized = run(program);
        PassResult {
            changed: fingerprint_program(&optimized) != before,
            program: optimized,
        }
    }
}

/// Run the decode→scan fusion over a Program.
///
/// Promotes every handoff-looking `ReadWrite` storage buffer to
/// workgroup memory. Returns the rewritten Program. Caller-visible
/// buffers (`pipeline_live_out = true`) are preserved as-is.
#[must_use]
pub fn run(program: Program) -> Program {
    let mut cross_workgroup: FxHashSet<Ident> = FxHashSet::default();
    cross_workgroup_buffers(program.entry(), &mut cross_workgroup);
    let promotable: FxHashSet<Ident> = program
        .buffers
        .iter()
        .filter(|b| is_promotable_handoff(b, &cross_workgroup))
        .map(|b| Ident::from(b.name()))
        .collect();

    if promotable.is_empty() {
        return program;
    }

    let new_buffers: Vec<BufferDecl> = program
        .buffers
        .iter()
        .map(|b| {
            if promotable.contains(&Ident::from(b.name())) {
                BufferDecl::workgroup(b.name(), b.count(), b.element())
            } else {
                b.clone()
            }
        })
        .collect();

    // VYRE_IR_HOTSPOTS audit: avoid the deep-clone of the entry
    // Vec<Node>. When the Arc is unique (the common case  -  we own
    // the only reference after `run()` returns) `try_unwrap` hands
    // back the Vec<Node> directly. Only fall back to cloning when
    // another Arc is still outstanding.
    let entry = std::sync::Arc::try_unwrap(program.entry).unwrap_or_else(|arc| (*arc).clone());
    Program::wrapped(new_buffers, program.workgroup_size, entry)
}

/// Count decode-handoff candidate buffers in `program`  -  the
/// buffers `run` would promote. Identical filter to `run`.
#[must_use]
pub fn count_opportunities(program: &Program) -> usize {
    let mut cross_workgroup: FxHashSet<Ident> = FxHashSet::default();
    cross_workgroup_buffers(program.entry(), &mut cross_workgroup);
    program
        .buffers
        .iter()
        .filter(|b| is_promotable_handoff(b, &cross_workgroup))
        .count()
}

/// Map from candidate handoff buffer name to its declared element
/// count. Parallel to [`count_opportunities`] with names exposed.
#[must_use]
pub fn candidate_handoffs(program: &Program) -> FxHashMap<Ident, u32> {
    let mut cross_workgroup: FxHashSet<Ident> = FxHashSet::default();
    cross_workgroup_buffers(program.entry(), &mut cross_workgroup);
    let mut out = FxHashMap::default();
    for buf in program.buffers.iter() {
        // Same promotability criteria as `run` and `count_opportunities`
        // (shared predicate). NOTE: this previously omitted the workgroup-byte
        // budget check that `run` enforces, so it reported oversize buffers as
        // candidates that `run` would never promote; routing through
        // `is_promotable_handoff` fixes that divergence.
        if is_promotable_handoff(buf, &cross_workgroup) {
            out.insert(Ident::from(buf.name()), buf.count());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferDecl, DataType, Program};

    fn decoder_like() -> Program {
        Program::wrapped(
            vec![
                BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(64),
                BufferDecl::storage("decoded", 1, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(128),
            ],
            [64, 1, 1],
            vec![],
        )
    }

    #[test]
    fn run_promotes_readwrite_handoff_to_workgroup() {
        let p = decoder_like();
        let before_bufs = p.buffers.len();
        let after = run(p);
        assert_eq!(after.buffers.len(), before_bufs);
        let decoded = after
            .buffers
            .iter()
            .find(|b| b.name() == "decoded")
            .unwrap();
        assert_eq!(decoded.access(), BufferAccess::Workgroup);
    }

    #[test]
    fn run_leaves_read_only_buffers_alone() {
        let p = decoder_like();
        let after = run(p);
        let input = after.buffers.iter().find(|b| b.name() == "input").unwrap();
        assert_eq!(input.access(), BufferAccess::ReadOnly);
    }

    #[test]
    fn run_preserves_pipeline_live_out_buffer() {
        // A ReadWrite buffer that is live-out must NOT be demoted
        // to workgroup memory  -  callers expect to read it back.
        let p = Program::wrapped(
            vec![
                BufferDecl::storage("result", 0, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(16)
                    .with_pipeline_live_out(true),
            ],
            [64, 1, 1],
            vec![],
        );
        let after = run(p);
        let r = after.buffers.iter().find(|b| b.name() == "result").unwrap();
        assert_eq!(r.access(), BufferAccess::ReadWrite);
        assert!(r.is_pipeline_live_out());
    }

    #[test]
    fn run_does_not_promote_a_buffer_reduced_across_workgroups() {
        use crate::ir::{CollectiveOp, CommGroup, Node};
        // `b` satisfies every decl criterion (ReadWrite, static count, not
        // live-out, fits the workgroup budget) but it is the target of an
        // AllReduce  -  a CROSS-WORKGROUP reduction over CommGroup::WORLD.
        // Workgroup memory is per-workgroup-private, so promoting `b` would
        // give each workgroup its own copy and silently destroy the reduction.
        // The decl-only filter promoted it; the body must veto the promotion.
        let p = Program::wrapped(
            vec![
                BufferDecl::storage("b", 0, BufferAccess::ReadWrite, DataType::U32).with_count(16),
            ],
            [64, 1, 1],
            vec![Node::AllReduce {
                buffer: "b".into(),
                op: CollectiveOp::Sum,
                group: CommGroup::WORLD,
            }],
        );
        let after = run(p);
        let b = after.buffers.iter().find(|x| x.name() == "b").unwrap();
        assert_eq!(
            b.access(),
            BufferAccess::ReadWrite,
            "a buffer reduced across workgroups by AllReduce must not be promoted to workgroup-private memory"
        );
        assert_eq!(
            count_opportunities(&Program::wrapped(
                vec![
                    BufferDecl::storage("b", 0, BufferAccess::ReadWrite, DataType::U32)
                        .with_count(16)
                ],
                [64, 1, 1],
                vec![Node::AllReduce {
                    buffer: "b".into(),
                    op: CollectiveOp::Sum,
                    group: CommGroup::WORLD,
                }],
            )),
            0,
            "count_opportunities must agree with run and report no promotable handoff"
        );
    }

    #[test]
    fn run_does_not_promote_an_all_gather_input_or_output() {
        use crate::ir::{CommGroup, Node};
        // AllGather moves data ACROSS workgroups from `src` into `dst`; both
        // endpoints are cross-workgroup and must not be promoted to per-
        // workgroup-private memory even though both decls qualify.
        let p = Program::wrapped(
            vec![
                BufferDecl::storage("src", 0, BufferAccess::ReadWrite, DataType::U32).with_count(8),
                BufferDecl::storage("dst", 1, BufferAccess::ReadWrite, DataType::U32).with_count(8),
            ],
            [64, 1, 1],
            vec![Node::AllGather {
                input: "src".into(),
                output: "dst".into(),
                group: CommGroup::WORLD,
            }],
        );
        let after = run(p);
        for name in ["src", "dst"] {
            let b = after.buffers.iter().find(|x| x.name() == name).unwrap();
            assert_eq!(
                b.access(),
                BufferAccess::ReadWrite,
                "an all-gather endpoint must not be promoted to workgroup-private memory"
            );
        }
    }

    #[test]
    fn run_is_identity_when_no_candidates() {
        let p = Program::wrapped(
            vec![
                BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(1),
            ],
            [64, 1, 1],
            vec![],
        );
        let after = run(p);
        assert_eq!(after.buffers.len(), 1);
        assert_eq!(after.buffers[0].access(), BufferAccess::ReadOnly);
    }

    #[test]
    fn run_skips_runtime_sized_buffers() {
        // count=0 means runtime-sized (no `with_count`); workgroup
        // allocations must be static so we can't promote those.
        let p = Program::wrapped(
            vec![BufferDecl::storage(
                "dynamic",
                0,
                BufferAccess::ReadWrite,
                DataType::U32,
            )],
            [64, 1, 1],
            vec![],
        );
        let after = run(p);
        let b = after
            .buffers
            .iter()
            .find(|b| b.name() == "dynamic")
            .unwrap();
        assert_eq!(b.access(), BufferAccess::ReadWrite);
    }

    #[test]
    fn count_opportunities_finds_one_candidate() {
        assert_eq!(count_opportunities(&decoder_like()), 1);
    }

    /// A ReadWrite handoff that exceeds 16 KiB stays in storage memory
    ///  -  wgpu's shared-memory floor would reject the workgroup decl on
    /// the fallback path. 4097 u32 elements = 16388 bytes, just above
    /// the 16384-byte budget.
    #[test]
    fn run_leaves_oversize_handoff_in_storage() {
        let p = Program::wrapped(
            vec![
                BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(64),
                BufferDecl::storage("decoded", 1, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(4097),
            ],
            [64, 1, 1],
            vec![],
        );
        assert_eq!(count_opportunities(&p), 0);
        let after = run(p);
        let decoded = after
            .buffers
            .iter()
            .find(|b| b.name() == "decoded")
            .unwrap();
        assert_eq!(
            decoded.access(),
            BufferAccess::ReadWrite,
            "oversize handoff must not be promoted; would exceed 16 KiB shared-memory floor"
        );
    }

    /// Twin of the above: a 4096-element buffer (exactly at 16 KiB) is
    /// still promotable.
    #[test]
    fn run_promotes_at_workgroup_byte_ceiling() {
        let p = Program::wrapped(
            vec![
                BufferDecl::storage("decoded", 1, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(4096),
            ],
            [64, 1, 1],
            vec![],
        );
        let after = run(p);
        let decoded = after
            .buffers
            .iter()
            .find(|b| b.name() == "decoded")
            .unwrap();
        assert_eq!(decoded.access(), BufferAccess::Workgroup);
    }

    #[test]
    fn count_opportunities_zero_on_read_only_program() {
        let p = Program::wrapped(
            vec![
                BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(1),
            ],
            [64, 1, 1],
            vec![],
        );
        assert_eq!(count_opportunities(&p), 0);
    }

    #[test]
    fn candidate_handoffs_exposes_name_and_count() {
        let p = decoder_like();
        let cands = candidate_handoffs(&p);
        assert_eq!(cands.get(&Ident::from("decoded")).copied(), Some(128));
        assert!(!cands.contains_key(&Ident::from("input")));
    }

    #[test]
    fn multiple_candidates_all_surface() {
        let p = Program::wrapped(
            vec![
                BufferDecl::storage("a", 0, BufferAccess::ReadWrite, DataType::U32).with_count(32),
                BufferDecl::storage("b", 1, BufferAccess::ReadWrite, DataType::U32).with_count(64),
                BufferDecl::storage("c", 2, BufferAccess::ReadOnly, DataType::U32).with_count(16),
            ],
            [64, 1, 1],
            vec![],
        );
        let cands = candidate_handoffs(&p);
        assert_eq!(cands.len(), 2);
        assert_eq!(cands.get(&Ident::from("a")).copied(), Some(32));
        assert_eq!(cands.get(&Ident::from("b")).copied(), Some(64));
    }
}
