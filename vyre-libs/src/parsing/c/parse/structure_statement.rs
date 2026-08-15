use crate::parsing::c::lex::tokens::*;
use crate::parsing::composition::child_phase;
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Tokens one tile of the next-boundary pass resolves.
///
/// This is a TILE WIDTH. It is deliberately not a statement length limit and
/// not a thread count: it only trades the depth of the tile-local passes
/// against the number of tiles the two single-lane composition passes walk.
/// Total work stays linear in the token count for ANY statement length, so no
/// statement is truncated by it.
const STMT_TILE_TOKENS: u32 = 256;

/// Scratch words of per-tile bookkeeping.
const STMT_TILE_WORDS: u32 = 8;

/// Unmatched closers (`)` / `]`) a tile consumes from the incoming depth.
const TILE_PAREN_CLOSERS: u32 = 0;
/// Unmatched openers (`(` / `[`) a tile leaves open.
const TILE_PAREN_OPENERS: u32 = 1;
/// Bracket counterpart of [`TILE_PAREN_CLOSERS`].
const TILE_BRACKET_CLOSERS: u32 = 2;
/// Bracket counterpart of [`TILE_PAREN_OPENERS`].
const TILE_BRACKET_OPENERS: u32 = 3;
/// Absolute paren depth on entry to the tile.
const TILE_PAREN_ENTRY: u32 = 4;
/// Absolute bracket depth on entry to the tile.
const TILE_BRACKET_ENTRY: u32 = 5;
/// First boundary index inside the tile, or [`STMT_NO_BOUNDARY`].
const TILE_FIRST_BOUNDARY: u32 = 6;
/// First boundary index at or after the tile's start, or [`STMT_NO_BOUNDARY`].
const TILE_NEXT_BOUNDARY: u32 = 7;

/// Sentinel for "no statement boundary at or after this position".
///
/// A position carrying this sentinel is reported as the EMPTY span
/// `[pos, pos)`. A terminated statement always contains its own terminator, so
/// its end is always at least `start + 1`; `end == start` is therefore
/// unreachable for a genuine statement and is the unambiguous unterminated
/// signal. See [`c11_statement_bounds`].
const STMT_NO_BOUNDARY: u32 = u32::MAX;
/// Name of the internal scratch buffer holding the per-position next-boundary
/// marks and the per-tile bookkeeping.
///
/// Declared LAST so `out_statements` and `out_counts` keep their output
/// positions. It is kernel-internal: a host caller never supplies its contents.
/// A dispatch path that allocates output buffers from the program declarations
/// (rather than taking them as inputs) must therefore include this name among
/// the buffers it allocates, and should suppress its readback since nothing
/// downstream reads it.
pub const C11_STATEMENT_BOUNDS_SCRATCH: &str = "c11_stmt_boundary_scratch";

/// Words of scratch [`c11_statement_bounds`] needs for a `num_tokens` window.
///
/// Layout: `[0, num_tokens)` holds one next-boundary mark per token position,
/// then one `STMT_TILE_WORDS`-word block per tile, plus one trailing block so
/// the last tile can read its successor's slot without an out-of-bounds load.
///
/// Callers must size the scratch buffer with this, never with a hand-copied
/// formula.
#[must_use]
pub const fn c11_statement_bounds_scratch_words(num_tokens: u32) -> u32 {
    let tokens = if num_tokens == 0 { 1 } else { num_tokens };
    // Ceiling division without `div_ceil`, which is not const on this toolchain.
    let tiles = (tokens - 1) / STMT_TILE_TOKENS + 1;
    tokens
        .saturating_add(tiles.saturating_mul(STMT_TILE_WORDS))
        .saturating_add(STMT_TILE_WORDS)
}

/// Shared index arithmetic for the tiled next-boundary passes.
struct BoundsCtx<'a> {
    tok_types: &'a str,
    tile_base: u32,
    t: Expr,
    active: Expr,
    tile_count: Expr,
}

impl BoundsCtx<'_> {
    /// Scratch index of `offset` inside `tile`'s bookkeeping block.
    fn tile_slot(&self, tile: Expr, offset: u32) -> Expr {
        Expr::add(
            Expr::add(
                Expr::u32(self.tile_base),
                Expr::mul(tile, Expr::u32(STMT_TILE_WORDS)),
            ),
            Expr::u32(offset),
        )
    }

    /// Scratch index of `offset` in the block of the tile this lane owns.
    fn own_slot(&self, offset: u32) -> Expr {
        self.tile_slot(self.t.clone(), offset)
    }

    /// First token index of the tile this lane owns.
    fn own_tile_lo(&self) -> Expr {
        Expr::mul(self.t.clone(), Expr::u32(STMT_TILE_TOKENS))
    }

    /// One past the last token index of the tile this lane owns.
    fn own_tile_hi(&self) -> Expr {
        Expr::min(
            Expr::add(self.own_tile_lo(), Expr::u32(STMT_TILE_TOKENS)),
            self.active.clone(),
        )
    }

    /// Clamped decrement matching the C depth walk: `d > 0 ? d - 1 : 0`.
    fn clamped_dec(name: &str) -> Expr {
        Expr::sub(Expr::max(Expr::var(name), Expr::u32(1)), Expr::u32(1))
    }

    /// Pass 1: reduce each tile to its bracket-matching summary.
    ///
    /// A token run reduces to `)^closers (^openers`, so a tile's whole effect on
    /// an incoming depth `e` is `max(e - closers, 0) + openers`. Both halves are
    /// non-negative, which keeps every intermediate in `u32` range.
    fn pass_tile_reduce(&self) -> Vec<Node> {
        vec![
            Node::let_bind("reduce_lo", self.own_tile_lo()),
            Node::let_bind("reduce_hi", self.own_tile_hi()),
            Node::let_bind("paren_closers", Expr::u32(0)),
            Node::let_bind("paren_openers", Expr::u32(0)),
            Node::let_bind("bracket_closers", Expr::u32(0)),
            Node::let_bind("bracket_openers", Expr::u32(0)),
            Node::loop_for(
                "reduce_tok",
                Expr::var("reduce_lo"),
                Expr::var("reduce_hi"),
                vec![
                    Node::let_bind(
                        "reduce_token",
                        Expr::load(self.tok_types, Expr::var("reduce_tok")),
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var("reduce_token"), Expr::u32(TOK_LPAREN)),
                        vec![Node::assign(
                            "paren_openers",
                            Expr::add(Expr::var("paren_openers"), Expr::u32(1)),
                        )],
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var("reduce_token"), Expr::u32(TOK_RPAREN)),
                        vec![Node::if_then_else(
                            Expr::gt(Expr::var("paren_openers"), Expr::u32(0)),
                            vec![Node::assign(
                                "paren_openers",
                                Expr::sub(Expr::var("paren_openers"), Expr::u32(1)),
                            )],
                            vec![Node::assign(
                                "paren_closers",
                                Expr::add(Expr::var("paren_closers"), Expr::u32(1)),
                            )],
                        )],
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var("reduce_token"), Expr::u32(TOK_LBRACKET)),
                        vec![Node::assign(
                            "bracket_openers",
                            Expr::add(Expr::var("bracket_openers"), Expr::u32(1)),
                        )],
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var("reduce_token"), Expr::u32(TOK_RBRACKET)),
                        vec![Node::if_then_else(
                            Expr::gt(Expr::var("bracket_openers"), Expr::u32(0)),
                            vec![Node::assign(
                                "bracket_openers",
                                Expr::sub(Expr::var("bracket_openers"), Expr::u32(1)),
                            )],
                            vec![Node::assign(
                                "bracket_closers",
                                Expr::add(Expr::var("bracket_closers"), Expr::u32(1)),
                            )],
                        )],
                    ),
                ],
            ),
            Node::store(
                C11_STATEMENT_BOUNDS_SCRATCH,
                self.own_slot(TILE_PAREN_CLOSERS),
                Expr::var("paren_closers"),
            ),
            Node::store(
                C11_STATEMENT_BOUNDS_SCRATCH,
                self.own_slot(TILE_PAREN_OPENERS),
                Expr::var("paren_openers"),
            ),
            Node::store(
                C11_STATEMENT_BOUNDS_SCRATCH,
                self.own_slot(TILE_BRACKET_CLOSERS),
                Expr::var("bracket_closers"),
            ),
            Node::store(
                C11_STATEMENT_BOUNDS_SCRATCH,
                self.own_slot(TILE_BRACKET_OPENERS),
                Expr::var("bracket_openers"),
            ),
        ]
    }

    /// Pass 2: compose the tile summaries into per-tile entry depths.
    ///
    /// One lane walks `tile_count` summaries, so this is `O(tokens / tile)`.
    /// It also seeds the trailing successor slot with the sentinel, which is
    /// what lets pass 5 read `tile + 1` unconditionally and in bounds.
    fn pass_tile_compose(&self) -> Vec<Node> {
        vec![
            Node::let_bind("entry_paren", Expr::u32(0)),
            Node::let_bind("entry_bracket", Expr::u32(0)),
            Node::store(
                C11_STATEMENT_BOUNDS_SCRATCH,
                self.tile_slot(self.tile_count.clone(), TILE_NEXT_BOUNDARY),
                Expr::u32(STMT_NO_BOUNDARY),
            ),
            Node::loop_for(
                "compose_tile",
                Expr::u32(0),
                self.tile_count.clone(),
                vec![
                    Node::store(
                        C11_STATEMENT_BOUNDS_SCRATCH,
                        self.tile_slot(Expr::var("compose_tile"), TILE_PAREN_ENTRY),
                        Expr::var("entry_paren"),
                    ),
                    Node::store(
                        C11_STATEMENT_BOUNDS_SCRATCH,
                        self.tile_slot(Expr::var("compose_tile"), TILE_BRACKET_ENTRY),
                        Expr::var("entry_bracket"),
                    ),
                    Node::let_bind(
                        "compose_pc",
                        Expr::load(
                            C11_STATEMENT_BOUNDS_SCRATCH,
                            self.tile_slot(Expr::var("compose_tile"), TILE_PAREN_CLOSERS),
                        ),
                    ),
                    Node::let_bind(
                        "compose_po",
                        Expr::load(
                            C11_STATEMENT_BOUNDS_SCRATCH,
                            self.tile_slot(Expr::var("compose_tile"), TILE_PAREN_OPENERS),
                        ),
                    ),
                    Node::let_bind(
                        "compose_bc",
                        Expr::load(
                            C11_STATEMENT_BOUNDS_SCRATCH,
                            self.tile_slot(Expr::var("compose_tile"), TILE_BRACKET_CLOSERS),
                        ),
                    ),
                    Node::let_bind(
                        "compose_bo",
                        Expr::load(
                            C11_STATEMENT_BOUNDS_SCRATCH,
                            self.tile_slot(Expr::var("compose_tile"), TILE_BRACKET_OPENERS),
                        ),
                    ),
                    // exit = max(entry - closers, 0) + openers, written so the
                    // subtraction can never wrap.
                    Node::assign(
                        "entry_paren",
                        Expr::add(
                            Expr::sub(
                                Expr::max(Expr::var("entry_paren"), Expr::var("compose_pc")),
                                Expr::var("compose_pc"),
                            ),
                            Expr::var("compose_po"),
                        ),
                    ),
                    Node::assign(
                        "entry_bracket",
                        Expr::add(
                            Expr::sub(
                                Expr::max(Expr::var("entry_bracket"), Expr::var("compose_bc")),
                                Expr::var("compose_bc"),
                            ),
                            Expr::var("compose_bo"),
                        ),
                    ),
                ],
            ),
        ]
    }

    /// Pass 3: mark every position as boundary-or-sentinel.
    ///
    /// The depth walk is the same recurrence the old windowed scan ran, but
    /// seeded from the tile's ABSOLUTE entry depth rather than from zero at an
    /// arbitrary candidate start. For any candidate that begins at absolute
    /// depth zero (which every real statement start does) the two agree
    /// exactly, so the brace predicate is unchanged where it is meaningful.
    fn pass_mark_boundaries(&self) -> Vec<Node> {
        vec![
            Node::let_bind("mark_lo", self.own_tile_lo()),
            Node::let_bind("mark_hi", self.own_tile_hi()),
            Node::let_bind(
                "paren_depth",
                Expr::load(
                    C11_STATEMENT_BOUNDS_SCRATCH,
                    self.own_slot(TILE_PAREN_ENTRY),
                ),
            ),
            Node::let_bind(
                "bracket_depth",
                Expr::load(
                    C11_STATEMENT_BOUNDS_SCRATCH,
                    self.own_slot(TILE_BRACKET_ENTRY),
                ),
            ),
            Node::let_bind("first_boundary", Expr::u32(STMT_NO_BOUNDARY)),
            Node::loop_for(
                "mark_tok",
                Expr::var("mark_lo"),
                Expr::var("mark_hi"),
                vec![
                    Node::let_bind("token", Expr::load(self.tok_types, Expr::var("mark_tok"))),
                    Node::if_then(
                        Expr::eq(Expr::var("token"), Expr::u32(TOK_LPAREN)),
                        vec![Node::assign(
                            "paren_depth",
                            Expr::add(Expr::var("paren_depth"), Expr::u32(1)),
                        )],
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var("token"), Expr::u32(TOK_RPAREN)),
                        vec![Node::assign(
                            "paren_depth",
                            Self::clamped_dec("paren_depth"),
                        )],
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var("token"), Expr::u32(TOK_LBRACKET)),
                        vec![Node::assign(
                            "bracket_depth",
                            Expr::add(Expr::var("bracket_depth"), Expr::u32(1)),
                        )],
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var("token"), Expr::u32(TOK_RBRACKET)),
                        vec![Node::assign(
                            "bracket_depth",
                            Self::clamped_dec("bracket_depth"),
                        )],
                    ),
                    Node::let_bind(
                        "at_top_level_expr",
                        Expr::and(
                            Expr::eq(Expr::var("paren_depth"), Expr::u32(0)),
                            Expr::eq(Expr::var("bracket_depth"), Expr::u32(0)),
                        ),
                    ),
                    Node::let_bind(
                        "is_brace_boundary",
                        Expr::and(
                            Expr::var("at_top_level_expr"),
                            Expr::or(
                                Expr::eq(Expr::var("token"), Expr::u32(TOK_LBRACE)),
                                Expr::eq(Expr::var("token"), Expr::u32(TOK_RBRACE)),
                            ),
                        ),
                    ),
                    Node::let_bind(
                        "is_statement_boundary",
                        Expr::or(
                            Expr::eq(Expr::var("token"), Expr::u32(TOK_SEMICOLON)),
                            Expr::var("is_brace_boundary"),
                        ),
                    ),
                    Node::store(
                        C11_STATEMENT_BOUNDS_SCRATCH,
                        Expr::var("mark_tok"),
                        Expr::select(
                            Expr::var("is_statement_boundary"),
                            Expr::var("mark_tok"),
                            Expr::u32(STMT_NO_BOUNDARY),
                        ),
                    ),
                    Node::if_then(
                        Expr::and(
                            Expr::var("is_statement_boundary"),
                            Expr::eq(Expr::var("first_boundary"), Expr::u32(STMT_NO_BOUNDARY)),
                        ),
                        vec![Node::assign("first_boundary", Expr::var("mark_tok"))],
                    ),
                ],
            ),
            Node::store(
                C11_STATEMENT_BOUNDS_SCRATCH,
                self.own_slot(TILE_FIRST_BOUNDARY),
                Expr::var("first_boundary"),
            ),
        ]
    }

    /// Pass 4: resolve each tile's first boundary at or after its own start.
    ///
    /// One lane walks the tiles back to front, which is `O(tokens / tile)`.
    /// `loop_for` counts up, so the index is mirrored inside the body.
    fn pass_resolve_tiles(&self) -> Vec<Node> {
        vec![
            Node::let_bind("tile_running", Expr::u32(STMT_NO_BOUNDARY)),
            Node::loop_for(
                "resolve_step",
                Expr::u32(0),
                self.tile_count.clone(),
                vec![
                    Node::let_bind(
                        "resolve_tile",
                        Expr::sub(
                            Expr::sub(self.tile_count.clone(), Expr::u32(1)),
                            Expr::var("resolve_step"),
                        ),
                    ),
                    Node::let_bind(
                        "resolve_first",
                        Expr::load(
                            C11_STATEMENT_BOUNDS_SCRATCH,
                            self.tile_slot(Expr::var("resolve_tile"), TILE_FIRST_BOUNDARY),
                        ),
                    ),
                    Node::if_then(
                        Expr::ne(Expr::var("resolve_first"), Expr::u32(STMT_NO_BOUNDARY)),
                        vec![Node::assign("tile_running", Expr::var("resolve_first"))],
                    ),
                    Node::store(
                        C11_STATEMENT_BOUNDS_SCRATCH,
                        self.tile_slot(Expr::var("resolve_tile"), TILE_NEXT_BOUNDARY),
                        Expr::var("tile_running"),
                    ),
                ],
            ),
        ]
    }

    /// Pass 5: walk each tile back to front, resolving every position's next
    /// boundary in amortised `O(1)` and appending its span.
    ///
    /// The running answer is seeded from the successor tile, so a statement may
    /// run to the end of the token stream. Positions with no boundary anywhere
    /// at or after them emit the empty span `[pos, pos)`.
    fn pass_emit_spans(&self, out_statements: &str, out_counts: &str) -> Vec<Node> {
        vec![
            Node::let_bind("emit_lo", self.own_tile_lo()),
            Node::let_bind("emit_hi", self.own_tile_hi()),
            Node::let_bind(
                "emit_running",
                Expr::load(
                    C11_STATEMENT_BOUNDS_SCRATCH,
                    self.tile_slot(Expr::add(self.t.clone(), Expr::u32(1)), TILE_NEXT_BOUNDARY),
                ),
            ),
            Node::loop_for(
                "emit_step",
                Expr::var("emit_lo"),
                Expr::var("emit_hi"),
                vec![
                    Node::let_bind(
                        "emit_pos",
                        Expr::sub(
                            Expr::sub(Expr::var("emit_hi"), Expr::u32(1)),
                            Expr::sub(Expr::var("emit_step"), Expr::var("emit_lo")),
                        ),
                    ),
                    Node::let_bind(
                        "emit_mark",
                        Expr::load(C11_STATEMENT_BOUNDS_SCRATCH, Expr::var("emit_pos")),
                    ),
                    Node::if_then(
                        Expr::ne(Expr::var("emit_mark"), Expr::u32(STMT_NO_BOUNDARY)),
                        vec![Node::assign("emit_running", Expr::var("emit_mark"))],
                    ),
                    Node::let_bind(
                        "stmt_bound_end",
                        Expr::select(
                            Expr::eq(Expr::var("emit_running"), Expr::u32(STMT_NO_BOUNDARY)),
                            Expr::var("emit_pos"),
                            Expr::add(Expr::var("emit_running"), Expr::u32(1)),
                        ),
                    ),
                    Node::let_bind(
                        "stmt_idx",
                        Expr::atomic_add(out_counts, Expr::u32(0), Expr::u32(2)),
                    ),
                    Node::store(out_statements, Expr::var("stmt_idx"), Expr::var("emit_pos")),
                    Node::store(
                        out_statements,
                        Expr::add(Expr::var("stmt_idx"), Expr::u32(1)),
                        Expr::var("stmt_bound_end"),
                    ),
                ],
            ),
        ]
    }
}

/// Compact C11 statement spans for AST construction.
///
/// Emits one candidate span per token position: for position `p` the span is
/// `[p, j + 1)` where `j` is the first statement boundary at or after `p`. A
/// boundary is a semicolon, or a brace at absolute paren and bracket depth
/// zero.
///
/// # Statement length is unbounded
///
/// There is no fixed lookahead. The kernel precomputes each position's next
/// boundary in five grid-synchronised passes whose total work is linear in the
/// token count, then reads the answer directly. An earlier revision scanned a
/// fixed 256-token window from each position and, when that window held no
/// boundary, fell through to an unconditional append that recorded `[p, p + 1)`
/// (santhreal/vyre `C11-STATEMENT-BOUNDS-SILENT-TRUNCATION`). That value is
/// byte for byte what a genuine one-token statement produces, so every C
/// statement longer than 256 tokens parsed wrong with no diagnostic anywhere.
///
/// # Unterminated statements are distinguishable
///
/// If no boundary exists at or after a position, its span is the EMPTY span
/// `[p, p)`. Because a terminated statement always contains its own terminator,
/// a real statement's end is always at least `start + 1`, so `end == start` is
/// unreachable for terminated input. A consumer can therefore always tell an
/// unterminated or truncated statement from a genuine short one.
///
/// # Token count, launch route, and the one real limit
///
/// There is NO correctness ceiling on the token count. The five pass boundaries
/// are `MemoryOrdering::GridSync` barriers, and a large enough grid changes the
/// launch ROUTE rather than failing.
///
/// The dispatch element count follows the largest binding, and `out_statements`
/// is `2 * num_tokens` words, so the launch is `2n` lanes and the grid is
/// `ceil(2n / 256)` blocks: DOUBLE what one lane per token would suggest, which
/// halves every threshold below. OBSERVED through the real inference path: 65536
/// tokens is 512 blocks, 130560 is exactly 1020, 130688 is 1021.
///
/// A cooperative launch needs every block co-resident, and that bound is
/// `(max_threads_per_sm / effective_workgroup) * sm_count` blocks. The bound uses
/// the EFFECTIVE workgroup, which the autotuner picks and which is NOT always the
/// declared one: `VYRE_AUTOTUNER` unset means `NaturalGradient`, and an eligible
/// kernel can be widened to 1024. This kernel is not eligible (it sets
/// `non_composable_with_self`), so the effective width is the declared 256 under
/// both tuner modes, MEASURED via `resolve_launch_workgroup_for_mode` and pinned
/// by `autotuner_does_not_widen_the_declared_workgroup`.
///
/// So on a 170 SM, 1536 thread/SM part (RTX 5090) the bound is
/// `(1536 / 256) * 170 = 1020` blocks, `1020 * 256 = 261_120` lanes, and at two
/// lanes per token `130_560` tokens is where the route changes. Were the kernel
/// ever widened to 1024, integer division gives `1536 / 1024 = 1` block per SM,
/// so `170 * 1024 = 174_080` lanes and the crossing point would fall to `87_040`
/// tokens. The formulas are normative; the numbers are this-device.
///
/// - At or below that, this runs as ONE cooperative launch with native grid
///   barriers.
/// - Above it, `vyre-driver-cuda` routes to a host-orchestrated kernel SPLIT:
///   each top-level `GridSync` becomes a kernel boundary, so the six phases
///   become six regular launches with no co-residency requirement. That is
///   semantically a real barrier and is correct at any grid width. It is not a
///   silent degrade of meaning, but it IS a cost change: six launches instead of
///   one, and no state held live across phases.
///
/// The barriers here are splittable by construction. They sit at the top level
/// of the single `Program::wrapped` region, which `vyre-driver::grid_sync` peels
/// before cutting, and none is nested inside a `Node::Loop` or an inner
/// `Node::Region` (which `validate::barrier` rejects outright).
///
/// The live compiler path never reaches the transition: `C11_AST_MAX_TOK_SCAN`
/// is 65536, which is 512 blocks, exactly half the 1020 limit, so it stays on
/// the single cooperative launch with 2x headroom.
/// `launch_geometry_stays_within_cooperative_residency_at_the_pipeline_cap`
/// fails if a future change spends that headroom, which would silently move the
/// C frontend onto the split route.
///
/// The ONE hard limit is this function's own buffer sizing, and the driver will
/// not surface it: `out_statements` is sized `tok_count.saturating_mul(2)`, so a
/// token count above `2^31 - 1` SATURATES and under-sizes the buffer instead of
/// failing. Unreachable in practice (that buffer would exceed 16 GB and the C
/// pipeline caps three orders of magnitude below it), but it is the real bound.
///
/// # Buffers
///
/// Four buffers in declaration order: `tok_types` (read-only), `out_statements`
/// (`2 * num_tokens` words), `out_counts` (one word), and
/// [`C11_STATEMENT_BOUNDS_SCRATCH`], sized by
/// [`c11_statement_bounds_scratch_words`].
///
/// The scratch buffer is KERNEL-INTERNAL working memory, NOT a result. Nothing
/// downstream should read it and its contents are an implementation detail. It
/// is declared last so `out_statements` and `out_counts` keep output slots 0 and
/// 1, but it still occupies a buffer slot, so a dispatch path that allocates
/// outputs from the program declarations must allocate it (and should suppress
/// its readback) rather than expect the host to supply it. Callers that count
/// buffers must not mistake the third slot for a third output.
///
/// # Panics
/// Panics when the token-window count is not a literal expression. The output
/// buffers are sized at build time; pass `Expr::u32(N)`.
#[must_use]
pub fn c11_statement_bounds(
    tok_types: &str,
    num_tokens: Expr,
    out_statements: &str,
    out_counts: &str,
) -> Program {
    let t = Expr::InvocationId { axis: 0 };
    let tok_count = match &num_tokens {
        Expr::LitU32(0) => 1,
        Expr::LitU32(n) => *n,
        // The statement output buffers are sized at build time from this count
        // (`tok_count.saturating_mul(2)` below). Silently defaulting a
        // runtime-dynamic count to 1 would mis-size those buffers and drop
        // statements with no signal (fail fast instead).
        other => panic!(
            "c11_statement_bounds requires a literal token-window count for build-time output \
             buffer sizing, got a non-literal expression {other:?}. Fix: pass Expr::u32(N)."
        ),
    };

    // Clamp the live range to the build-time window so every scratch and output
    // index stays inside its declared region even if the bound buffer is longer.
    let active = Expr::min(Expr::buf_len(tok_types), Expr::u32(tok_count));
    let tile_count = Expr::div(
        Expr::add(active.clone(), Expr::u32(STMT_TILE_TOKENS - 1)),
        Expr::u32(STMT_TILE_TOKENS),
    );
    let ctx = BoundsCtx {
        tok_types,
        tile_base: tok_count,
        t: t.clone(),
        active,
        tile_count: tile_count.clone(),
    };

    let owns_tile = Expr::lt(t.clone(), tile_count.clone());
    let is_lead_lane = Expr::eq(t.clone(), Expr::u32(0));
    let has_tiles = Expr::gt(tile_count, Expr::u32(0));

    // Every barrier below is GridSync and sits at the top level of the entry
    // sequence. Both are required: V010 rejects a barrier under divergent
    // control flow, and a workgroup-scope fence would let a lane in another
    // workgroup read a pass's scratch before it was written. Backends without a
    // native cooperative grid barrier get an equivalent host-orchestrated
    // kernel split (see `vyre-driver::grid_sync`).
    Program::wrapped(
        vec![
            BufferDecl::storage(tok_types, 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(out_statements, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(tok_count.saturating_mul(2)),
            BufferDecl::storage(out_counts, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
            BufferDecl::storage(
                C11_STATEMENT_BOUNDS_SCRATCH,
                3,
                BufferAccess::ReadWrite,
                DataType::U32,
            )
            .with_count(c11_statement_bounds_scratch_words(tok_count)),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::parsing::c11_statement_bounds",
            vec![
                Node::if_then(
                    is_lead_lane.clone(),
                    vec![Node::store(out_counts, Expr::u32(0), Expr::u32(0))],
                ),
                Node::Barrier {
                    ordering: MemoryOrdering::GridSync,
                },
                Node::if_then(owns_tile.clone(), ctx.pass_tile_reduce()),
                Node::Barrier {
                    ordering: MemoryOrdering::GridSync,
                },
                Node::if_then(is_lead_lane.clone(), ctx.pass_tile_compose()),
                Node::Barrier {
                    ordering: MemoryOrdering::GridSync,
                },
                Node::if_then(owns_tile.clone(), ctx.pass_mark_boundaries()),
                Node::Barrier {
                    ordering: MemoryOrdering::GridSync,
                },
                Node::if_then(Expr::and(is_lead_lane, has_tiles), ctx.pass_resolve_tiles()),
                Node::Barrier {
                    ordering: MemoryOrdering::GridSync,
                },
                child_phase(
                    "vyre-libs::parsing::c11_statement_bounds",
                    vyre_primitives::bitset::select::OP_ID,
                    vec![Node::if_then(
                        owns_tile,
                        ctx.pass_emit_spans(out_statements, out_counts),
                    )],
                ),
            ],
        )],
    )
    .with_entry_op_id("vyre-libs::parsing::c11_statement_bounds")
    .with_non_composable_with_self(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statement_bounds_sizes_outputs_to_full_literal_window_without_fixed_clamp() {
        let token_window = crate::parsing::c::pipeline::stages::C11_AST_MAX_TOK_SCAN + 1;
        let program = c11_statement_bounds(
            "tok_types",
            Expr::u32(token_window),
            "out_statements",
            "out_counts",
        );
        let out_statements = program
            .buffers
            .iter()
            .find(|buffer| buffer.name() == "out_statements")
            .expect("Fix: out_statements buffer must exist");
        assert_eq!(out_statements.count, token_window.saturating_mul(2));
    }

    #[test]
    fn statement_bounds_rejects_non_literal_token_count_for_buffer_sizing() {
        let panic = std::panic::catch_unwind(|| {
            let _ = c11_statement_bounds(
                "tok_types",
                Expr::var("dynamic_tokens"),
                "out_statements",
                "out_counts",
            );
        })
        .expect_err("non-literal statement-bound count must fail");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&'static str>().copied())
            .unwrap_or("<non-string panic>");
        assert!(
            message.contains("requires a literal token-window count"),
            "{message}"
        );
    }

    #[test]
    fn scratch_words_covers_marks_tiles_and_the_successor_slot() {
        // One mark per token, one 8-word block per 256-token tile, plus the
        // trailing block pass 5 reads for `tile + 1`.
        assert_eq!(
            c11_statement_bounds_scratch_words(256),
            256 + STMT_TILE_WORDS + STMT_TILE_WORDS
        );
        assert_eq!(
            c11_statement_bounds_scratch_words(257),
            257 + 2 * STMT_TILE_WORDS + STMT_TILE_WORDS
        );
        // A zero window is sized as one token, matching the buffer sizing above.
        assert_eq!(
            c11_statement_bounds_scratch_words(0),
            c11_statement_bounds_scratch_words(1)
        );
    }

    #[test]
    fn scratch_buffer_is_declared_after_both_reported_outputs() {
        // The frontend reads outputs[0] as statements and outputs[1] as counts.
        // Scratch must stay last or those indices shift silently.
        let program =
            c11_statement_bounds("tok_types", Expr::u32(64), "out_statements", "out_counts");
        let names: Vec<&str> = program.buffers.iter().map(|b| b.name()).collect();
        assert_eq!(
            names,
            vec![
                "tok_types",
                "out_statements",
                "out_counts",
                C11_STATEMENT_BOUNDS_SCRATCH
            ]
        );
    }
}
