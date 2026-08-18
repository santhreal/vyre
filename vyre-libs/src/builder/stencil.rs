//! Canonical 2D grid, coordinate decomposition, stencil, and pixel composer.
//!
//! Unifies 2D index mapping (`y * width + x`), coordinate decomposition (`idx -> (y, x)`),
//! boundary clipping and padding, 3x3/separable stencil walks, character-cell grids,
//! and packed RGBA channel manipulation into a single reusable owner.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::{BufferDecl, Expr, Ident, Node, Program};

/// Default workgroup geometry for per-pixel image processing pipelines.
pub const DEFAULT_2D_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// 2D grid dimensions and element count invariants.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Grid2DShape {
    /// Grid width (columns / horizontal extent).
    pub width: u32,
    /// Grid height (rows / vertical extent).
    pub height: u32,
}

impl Grid2DShape {
    /// Create a new 2D grid shape.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Whether either grid dimension is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Total elements or pixels in the 2D grid, saturating on overflow.
    #[must_use]
    pub const fn pixel_count(&self) -> u32 {
        self.width.saturating_mul(self.height)
    }

    /// Total elements in the 2D grid, returning `None` on u32 multiplication overflow.
    #[cfg(test)]
    #[must_use]
    pub const fn checked_pixel_count(&self) -> Option<u32> {
        self.width.checked_mul(self.height)
    }

    /// Return width floored at 1 to prevent division-by-zero during coordinate mapping.
    #[must_use]
    pub const fn safe_width(&self) -> u32 {
        if self.width == 0 {
            1
        } else {
            self.width
        }
    }

    /// Return height floored at 1 to prevent division-by-zero during coordinate mapping.
    #[must_use]
    pub const fn safe_height(&self) -> u32 {
        if self.height == 0 {
            1
        } else {
            self.height
        }
    }
}

/// Character-cell grid geometry for terminal and text surfaces.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CellGridShape {
    /// Cells across.
    pub cols: u32,
    /// Cells down.
    pub rows: u32,
    /// Pixels across one cell.
    pub cell_width: u32,
    /// Pixels down one cell.
    pub cell_height: u32,
}

impl CellGridShape {
    /// Create a new character-cell grid shape.
    #[must_use]
    pub const fn new(cols: u32, rows: u32, cell_width: u32, cell_height: u32) -> Self {
        Self {
            cols,
            rows,
            cell_width,
            cell_height,
        }
    }

    /// Pixels across the whole surface.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.cols * self.cell_width
    }

    /// Pixels down the whole surface.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.rows * self.cell_height
    }

    /// Cells in the grid.
    #[must_use]
    pub const fn cell_count(&self) -> u32 {
        self.cols * self.rows
    }

    /// Pixels in the surface.
    #[must_use]
    pub const fn pixel_count(&self) -> u32 {
        self.width() * self.height()
    }

    /// Validate dimensions and overflow bounds.
    ///
    /// # Panics
    ///
    /// Panics if `cols` or `rows` is 0, if `cell_width` or `cell_height` is 0,
    /// or if any dimension multiplication (`cols * cell_width`, `rows * cell_height`,
    /// surface pixels, or `cols * rows`) overflows `u32`.
    #[must_use]
    pub fn validated(self) -> Self {
        assert!(
            self.cols > 0 && self.rows > 0,
            "Fix: a cell grid needs at least one row and one column, got {}x{}",
            self.cols,
            self.rows
        );
        assert!(
            self.cell_width > 0 && self.cell_height > 0,
            "Fix: a cell needs a non-zero size, got {}x{} pixels",
            self.cell_width,
            self.cell_height
        );
        let width = self
            .cols
            .checked_mul(self.cell_width)
            .expect("Fix: cols * cell_width overflows u32");
        let height = self
            .rows
            .checked_mul(self.cell_height)
            .expect("Fix: rows * cell_height overflows u32");
        width
            .checked_mul(height)
            .expect("Fix: the surface pixel count overflows u32");
        self.cols
            .checked_mul(self.rows)
            .expect("Fix: cols * rows overflows u32");
        self
    }
}

// ============================================================================
// Canonical 2D Index & Coordinate Calculations
// ============================================================================

/// Flatten 2D coordinates `(y, x)` into a row-major linear index `y * width + x`.
#[must_use]
pub fn flat_index(y: Expr, width: u32, x: Expr) -> Expr {
    Expr::add(Expr::mul(y, Expr::u32(width)), x)
}

/// Flatten 2D coordinates `(y, x)` with a dynamic width expression: `y * width + x`.
#[must_use]
pub fn flat_index_expr(y: Expr, width: Expr, x: Expr) -> Expr {
    Expr::add(Expr::mul(y, width), x)
}

/// Decompose linear index `idx` into 2D coordinates `(py, px)` with `py = idx / width`
/// and `px = idx % width`.
#[must_use]
pub fn decompose_index(idx: &Expr, width: u32) -> (Expr, Expr) {
    let safe_w = width.max(1);
    let py = Expr::div(idx.clone(), Expr::u32(safe_w));
    let px = Expr::rem(idx.clone(), Expr::u32(safe_w));
    (py, px)
}

/// Decompose linear index `idx` into 2D coordinates `(py, px)` reusing `py` to eliminate
/// redundant hardware remainder division: `px = idx - py * width`.
#[must_use]
pub fn decompose_index_fast(idx: &Expr, width: u32) -> (Expr, Expr) {
    let safe_w = width.max(1);
    let py = Expr::div(idx.clone(), Expr::u32(safe_w));
    let px = Expr::sub(idx.clone(), Expr::mul(py.clone(), Expr::u32(safe_w)));
    (py, px)
}

/// Clamp coordinate offset `coord + offset` to boundary interval `[0, extent - 1]`.
#[must_use]
pub fn coord_clamp_offset(coord: &Expr, offset: i32, extent: u32) -> Expr {
    let safe_extent = extent.max(1);
    if offset >= 0 {
        let off_u32 = offset as u32;
        Expr::select(
            Expr::lt(
                Expr::add(coord.clone(), Expr::u32(off_u32)),
                Expr::u32(safe_extent),
            ),
            Expr::add(coord.clone(), Expr::u32(off_u32)),
            Expr::u32(safe_extent - 1),
        )
    } else {
        let neg_u32 = (-offset) as u32;
        Expr::select(
            Expr::ge(coord.clone(), Expr::u32(neg_u32)),
            Expr::sub(coord.clone(), Expr::u32(neg_u32)),
            Expr::u32(0),
        )
    }
}

/// Check whether unsigned coordinate plus offset `coord + delta` is strictly inside `[0, extent)`.
#[must_use]
pub fn axis_in_bounds(coord: &Expr, delta: i32, extent: u32) -> Expr {
    if delta > 0 {
        let d = delta as u32;
        Expr::lt(coord.clone(), Expr::u32(extent.saturating_sub(d)))
    } else if delta < 0 {
        let neg = (-delta) as u32;
        Expr::ge(coord.clone(), Expr::u32(neg))
    } else {
        Expr::lt(coord.clone(), Expr::u32(extent))
    }
}

/// Safe shifted coordinate for unsigned indexing with a signed delta offset.
#[must_use]
pub fn shifted_coord(coord: &Expr, delta: i32) -> Expr {
    if delta >= 0 {
        Expr::add(coord.clone(), Expr::u32(delta as u32))
    } else {
        let neg = (-delta) as u32;
        Expr::select(
            Expr::ge(coord.clone(), Expr::u32(neg)),
            Expr::sub(coord.clone(), Expr::u32(neg)),
            Expr::u32(0),
        )
    }
}

/// Compute linear sample index for separable horizontal or vertical convolution passes
/// with edge clamping.
#[must_use]
pub fn separable_sample_index(
    is_horizontal: bool,
    py: &Expr,
    px: &Expr,
    offset: i32,
    width: u32,
    height: u32,
) -> Expr {
    if is_horizontal {
        let sample_x = coord_clamp_offset(px, offset, width);
        flat_index(py.clone(), width, sample_x)
    } else {
        let sample_y = coord_clamp_offset(py, offset, height);
        flat_index(sample_y, width, px.clone())
    }
}

// ============================================================================
// 3x3 Stencil & Patch Taps (im2col / conv2d)
// ============================================================================

/// One tap of a 3x3 2D neighbourhood patch centered at `(y, x)`.
#[derive(Clone, Debug)]
pub struct Stencil3x3Tap {
    /// Flattened column index in row-major patch order `ky * 3 + kx` (0..9).
    pub column: u32,
    /// Boolean predicate indicating whether the sampled coordinate is in image bounds.
    pub in_bounds: Expr,
    /// Linear row-major buffer index `sample_y * width + sample_x`.
    pub input_index: Expr,
}

/// The nine taps of the zero-padded 3x3 patch centered at `(y, x)` in im2col column order.
#[must_use]
pub fn stencil_3x3_taps(y: &Expr, x: &Expr, height: u32, width: u32) -> Vec<Stencil3x3Tap> {
    let mut taps = Vec::with_capacity(9);
    for ky in 0..3u32 {
        for kx in 0..3u32 {
            let dy = (ky as i32) - 1;
            let dx = (kx as i32) - 1;
            taps.push(Stencil3x3Tap {
                column: ky * 3 + kx,
                in_bounds: Expr::and(axis_in_bounds(y, dy, height), axis_in_bounds(x, dx, width)),
                input_index: flat_index(shifted_coord(y, dy), width, shifted_coord(x, dx)),
            });
        }
    }
    taps
}

/// Sample an F32 value from `input` at `tap`, substituting zero-padding outside image bounds.
#[must_use]
pub fn sample_stencil_3x3_or_zero(input: &str, tap: &Stencil3x3Tap) -> Expr {
    Expr::select(
        tap.in_bounds.clone(),
        Expr::load(input, tap.input_index.clone()),
        Expr::f32(0.0),
    )
}

// ============================================================================
// 2x2 Downsampling & Upsampling Taps
// ============================================================================

/// Derive the 4 source pixel indices for a 2x2 downsample box filter:
/// `[top-left (p00), top-right (p10), bottom-left (p01), bottom-right (p11)]`.
#[must_use]
pub fn downsample_2x_source_indices(oy: &Expr, ox: &Expr, src_width: u32) -> [Expr; 4] {
    let sy = Expr::mul(oy.clone(), Expr::u32(2));
    let sx = Expr::mul(ox.clone(), Expr::u32(2));
    let sy_plus_1 = Expr::add(sy.clone(), Expr::u32(1));
    let sx_plus_1 = Expr::add(sx.clone(), Expr::u32(1));

    [
        flat_index(sy.clone(), src_width, sx.clone()),
        flat_index(sy, src_width, sx_plus_1),
        flat_index(sy_plus_1.clone(), src_width, sx),
        flat_index(sy_plus_1, src_width, sx_plus_1_expr(ox)),
    ]
}

fn sx_plus_1_expr(ox: &Expr) -> Expr {
    Expr::add(Expr::mul(ox.clone(), Expr::u32(2)), Expr::u32(1))
}

/// Map destination coordinates `(oy, ox)` to nearest-neighbour source index for 2x upsampling.
#[must_use]
pub fn upsample_2x_source_index(oy: &Expr, ox: &Expr, in_width: u32) -> Expr {
    let iy = Expr::div(oy.clone(), Expr::u32(2));
    let ix = Expr::div(ox.clone(), Expr::u32(2));
    flat_index(iy, in_width.max(1), ix)
}

// ============================================================================
// Cell Grid Node Decomposition
// ============================================================================

/// Bind `y`, `x`, `col`, `row`, and `cell` from `idx` for a character-cell grid.
#[must_use]
pub fn cell_lookup_nodes(shape: CellGridShape) -> Vec<Node> {
    let width = shape.width();
    vec![
        Node::let_bind("y", Expr::div(Expr::var("idx"), Expr::u32(width.max(1)))),
        Node::let_bind(
            "x",
            Expr::sub(
                Expr::var("idx"),
                Expr::mul(Expr::var("y"), Expr::u32(width.max(1))),
            ),
        ),
        Node::let_bind(
            "col",
            Expr::div(Expr::var("x"), Expr::u32(shape.cell_width.max(1))),
        ),
        Node::let_bind(
            "row",
            Expr::div(Expr::var("y"), Expr::u32(shape.cell_height.max(1))),
        ),
        Node::let_bind(
            "cell",
            Expr::add(
                Expr::mul(Expr::var("row"), Expr::u32(shape.cols.max(1))),
                Expr::var("col"),
            ),
        ),
    ]
}

// ============================================================================
// Packed RGBA Channel Manipulation
// ============================================================================

/// Return `(left * right) >> shift` without losing the high half of the unsigned 32-bit product.
#[must_use]
pub fn wide_mul_shr_u32(left: Expr, right: Expr, shift: u32) -> Expr {
    debug_assert!((1..32).contains(&shift));
    let low = Expr::mul(left.clone(), right.clone());
    let high = Expr::mulhi(left, right);
    Expr::bitor(
        Expr::shr(low, Expr::u32(shift)),
        Expr::shl(high, Expr::u32(32 - shift)),
    )
}

/// Unsigned 16.16 fixed-point multiplication: `(left * right) >> 16`.
#[must_use]
pub fn fixed_mul_16_16(left: Expr, right: Expr) -> Expr {
    wide_mul_shr_u32(left, right, 16)
}

/// Extract one 8-bit channel from a packed `u32` RGBA word by bit shift (0=R, 8=G, 16=B, 24=A).
#[must_use]
pub fn unpack_channel_expr(pixel: Expr, shift: u32) -> Expr {
    let shifted = if shift == 0 {
        pixel
    } else {
        Expr::shr(pixel, Expr::u32(shift))
    };
    if shift == 24 {
        shifted
    } else {
        Expr::bitand(shifted, Expr::u32(0xFF))
    }
}

/// Extract one 8-bit channel from a named variable holding a packed `u32` RGBA word.
#[must_use]
pub fn unpack_channel(pixel_var: &str, shift: u32) -> Expr {
    unpack_channel_expr(Expr::var(pixel_var), shift)
}

/// Unpack all 4 channels `(R, G, B, A)` from a named variable.
#[must_use]
pub fn unpack_rgba(pixel_var: &str) -> (Expr, Expr, Expr, Expr) {
    (
        unpack_channel(pixel_var, 0),
        unpack_channel(pixel_var, 8),
        unpack_channel(pixel_var, 16),
        unpack_channel(pixel_var, 24),
    )
}

/// Pack 4 8-bit channels `(r, g, b, a)` into a single little-endian `u32` RGBA word.
#[must_use]
pub fn pack_rgba(r: Expr, g: Expr, b: Expr, a: Expr) -> Expr {
    Expr::bitor(
        Expr::bitor(r, Expr::shl(g, Expr::u32(8))),
        Expr::bitor(Expr::shl(b, Expr::u32(16)), Expr::shl(a, Expr::u32(24))),
    )
}

/// Pack 4 named channel variables `(r, g, b, a)` into a single little-endian `u32` RGBA word.
#[must_use]
pub fn pack_rgba_named(r_var: &str, g_var: &str, b_var: &str, a_var: &str) -> Expr {
    pack_rgba(
        Expr::var(r_var),
        Expr::var(g_var),
        Expr::var(b_var),
        Expr::var(a_var),
    )
}

/// Clamp value to `[0, 255]`.
#[must_use]
pub fn clamp_u8(val: Expr) -> Expr {
    Expr::select(Expr::gt(val.clone(), Expr::u32(255)), Expr::u32(255), val)
}

/// Porter-Duff "over" channel blend: `out_c = fg_c + ((bg_c * inv_a + 128) * 257 >> 16)`.
#[must_use]
pub fn blend_channel_porter_duff(fg_c: Expr, bg_c: Expr, inv_a: Expr) -> Expr {
    Expr::add(
        fg_c,
        wide_mul_shr_u32(
            Expr::add(Expr::mul(bg_c, inv_a), Expr::u32(128)),
            Expr::u32(257),
            16,
        ),
    )
}

/// Coverage-weighted alpha blend: `out_c = ((fg_c * cov + bg_c * inv_cov + 128) * 257) >> 16`.
#[must_use]
pub fn blend_channel_coverage(fg_c: Expr, bg_c: Expr, cov: Expr, inv_cov: Expr) -> Expr {
    wide_mul_shr_u32(
        Expr::add(
            Expr::add(Expr::mul(fg_c, cov), Expr::mul(bg_c, inv_cov)),
            Expr::u32(128),
        ),
        Expr::u32(257),
        16,
    )
}

/// Compute the rounded box-average of four 8-bit channel samples: `(c0 + c1 + c2 + c3 + 2) >> 2`.
#[must_use]
pub fn avg4_channel(c0: Expr, c1: Expr, c2: Expr, c3: Expr) -> Expr {
    Expr::shr(
        Expr::add(
            Expr::add(
                Expr::add(c0, c1),
                Expr::add(c2, c3),
            ),
            Expr::u32(2),
        ),
        Expr::u32(2),
    )
}

// ============================================================================
// Grid2DComposer
// ============================================================================

/// Composable builder for 2D grid and stencil compute programs.
#[derive(Clone, Debug)]
pub struct Grid2DComposer {
    op_id: &'static str,
    shape: Grid2DShape,
    buffers: Vec<BufferDecl>,
    workgroup_size: [u32; 3],
    child_region: Option<(&'static str, &'static str)>,
}

impl Grid2DComposer {
    /// Create a new 2D grid composer for the given operation id and dimensions.
    #[must_use]
    pub fn new(op_id: &'static str, width: u32, height: u32) -> Self {
        Self {
            op_id,
            shape: Grid2DShape::new(width, height),
            buffers: Vec::new(),
            workgroup_size: DEFAULT_2D_WORKGROUP_SIZE,
            child_region: None,
        }
    }

    /// Set buffer declarations for the program.
    #[must_use]
    pub fn with_buffers(mut self, buffers: Vec<BufferDecl>) -> Self {
        self.buffers = buffers;
        self
    }

    /// Override the dispatch workgroup size.
    #[must_use]
    pub fn with_workgroup_size(mut self, workgroup_size: [u32; 3]) -> Self {
        self.workgroup_size = workgroup_size;
        self
    }

    /// Nest a child region with `(child_op_id, parent_op_id)` inside the outer region.
    #[must_use]
    pub fn with_child_region(
        mut self,
        child_op_id: &'static str,
        parent_op_id: &'static str,
    ) -> Self {
        self.child_region = Some((child_op_id, parent_op_id));
        self
    }

    /// Build the 2D grid program by supplying an inner per-pixel body generator.
    ///
    /// The body callback receives `(shape, idx_expr, py_expr, px_expr)`.
    pub fn build<F>(self, body_fn: F) -> Program
    where
        F: FnOnce(&Grid2DShape, Expr, Expr, Expr) -> Vec<Node>,
    {
        let count = self.shape.pixel_count();
        let idx = Expr::gid_x();
        let (py, px) = decompose_index(&idx, self.shape.safe_width());
        let inner_body = body_fn(&self.shape, idx.clone(), py, px);

        let guarded_body = vec![
            Node::let_bind("idx", idx),
            Node::if_then(Expr::lt(Expr::var("idx"), Expr::u32(count)), inner_body),
        ];

        let root_node = if let Some((child_id, parent_id)) = self.child_region {
            wrap_anonymous_region(
                self.op_id,
                vec![wrap_child_region(
                    child_id,
                    Ident::from(parent_id),
                    guarded_body,
                )],
            )
        } else {
            wrap_anonymous_region(self.op_id, guarded_body)
        };

        Program::wrapped(self.buffers, self.workgroup_size, vec![root_node])
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node};

    #[test]
    fn grid_2d_shape_invariants() {
        let shape = Grid2DShape::new(64, 48);
        assert!(!shape.is_empty());
        assert_eq!(shape.pixel_count(), 64 * 48);
        assert_eq!(shape.checked_pixel_count(), Some(64 * 48));
        assert_eq!(shape.safe_width(), 64);
        assert_eq!(shape.safe_height(), 48);

        let empty_w = Grid2DShape::new(0, 48);
        assert!(empty_w.is_empty());
        assert_eq!(empty_w.pixel_count(), 0);
        assert_eq!(empty_w.safe_width(), 1);
        assert_eq!(empty_w.safe_height(), 48);

        let empty_h = Grid2DShape::new(64, 0);
        assert!(empty_h.is_empty());
        assert_eq!(empty_h.safe_width(), 64);
        assert_eq!(empty_h.safe_height(), 1);
    }

    #[test]
    fn cell_grid_shape_computes_surfaces() {
        let cell_shape = CellGridShape::new(80, 24, 10, 19).validated();
        assert_eq!(cell_shape.width(), 800);
        assert_eq!(cell_shape.height(), 456);
        assert_eq!(cell_shape.cell_count(), 1920);
        assert_eq!(cell_shape.pixel_count(), 800 * 456);
    }

    #[test]
    #[should_panic(expected = "at least one row and one column")]
    fn cell_grid_shape_rejects_empty_grid() {
        let _ = CellGridShape::new(0, 24, 8, 16).validated();
    }

    #[test]
    #[should_panic(expected = "non-zero size")]
    fn cell_grid_shape_rejects_zero_cell_size() {
        let _ = CellGridShape::new(80, 24, 0, 16).validated();
    }

    #[test]
    fn flat_index_constructs_row_major_expression() {
        let y = Expr::var("y");
        let x = Expr::var("x");
        let idx = flat_index(y.clone(), 64, x.clone());
        assert_eq!(idx, Expr::add(Expr::mul(y, Expr::u32(64)), x));
    }

    #[test]
    fn decompose_index_generates_div_and_rem() {
        let idx = Expr::var("idx");
        let (py, px) = decompose_index(&idx, 128);
        assert_eq!(py, Expr::div(idx.clone(), Expr::u32(128)));
        assert_eq!(px, Expr::rem(idx, Expr::u32(128)));
    }

    #[test]
    fn decompose_index_fast_reuses_py() {
        let idx = Expr::var("idx");
        let (py, px) = decompose_index_fast(&idx, 128);
        assert_eq!(py, Expr::div(idx.clone(), Expr::u32(128)));
        assert_eq!(px, Expr::sub(idx, Expr::mul(py, Expr::u32(128))));
    }

    #[test]
    fn coord_clamp_offset_positive_and_negative() {
        let px = Expr::var("px");
        let clamped_pos = coord_clamp_offset(&px, 2, 100);
        assert_eq!(
            clamped_pos,
            Expr::select(
                Expr::lt(Expr::add(px.clone(), Expr::u32(2)), Expr::u32(100)),
                Expr::add(px.clone(), Expr::u32(2)),
                Expr::u32(99),
            )
        );

        let clamped_neg = coord_clamp_offset(&px, -3, 100);
        assert_eq!(
            clamped_neg,
            Expr::select(
                Expr::ge(px.clone(), Expr::u32(3)),
                Expr::sub(px, Expr::u32(3)),
                Expr::u32(0),
            )
        );
    }

    #[test]
    fn separable_sample_index_switches_axis() {
        let py = Expr::var("py");
        let px = Expr::var("px");
        let horiz_idx = separable_sample_index(true, &py, &px, 1, 64, 32);
        let vert_idx = separable_sample_index(false, &py, &px, 1, 64, 32);

        let expected_horiz = flat_index(py.clone(), 64, coord_clamp_offset(&px, 1, 64));
        let expected_vert = flat_index(coord_clamp_offset(&py, 1, 32), 64, px);

        assert_eq!(horiz_idx, expected_horiz);
        assert_eq!(vert_idx, expected_vert);
    }

    #[test]
    fn stencil_3x3_generates_exact_nine_taps() {
        let y = Expr::var("y");
        let x = Expr::var("x");
        let taps = stencil_3x3_taps(&y, &x, 16, 16);
        assert_eq!(taps.len(), 9);
        for (i, tap) in taps.iter().enumerate() {
            assert_eq!(tap.column, i as u32);
        }
        // Center tap (ky=1, kx=1 -> dy=0, dx=0) has column 4
        let center = &taps[4];
        assert_eq!(center.column, 4);
        let sampled = sample_stencil_3x3_or_zero("input", center);
        assert_eq!(
            sampled,
            Expr::select(
                center.in_bounds.clone(),
                Expr::load("input", center.input_index.clone()),
                Expr::f32(0.0),
            )
        );
    }

    #[test]
    fn downsample_and_upsample_indices() {
        let oy = Expr::var("oy");
        let ox = Expr::var("ox");
        let [p00, p10, p01, p11] = downsample_2x_source_indices(&oy, &ox, 64);
        assert_eq!(
            p00,
            flat_index(
                Expr::mul(oy.clone(), Expr::u32(2)),
                64,
                Expr::mul(ox.clone(), Expr::u32(2))
            )
        );
        assert_eq!(
            p10,
            flat_index(
                Expr::mul(oy.clone(), Expr::u32(2)),
                64,
                Expr::add(Expr::mul(ox.clone(), Expr::u32(2)), Expr::u32(1))
            )
        );
        assert_eq!(
            p01,
            flat_index(
                Expr::add(Expr::mul(oy.clone(), Expr::u32(2)), Expr::u32(1)),
                64,
                Expr::mul(ox.clone(), Expr::u32(2))
            )
        );
        assert_eq!(
            p11,
            flat_index(
                Expr::add(Expr::mul(oy.clone(), Expr::u32(2)), Expr::u32(1)),
                64,
                Expr::add(Expr::mul(ox.clone(), Expr::u32(2)), Expr::u32(1))
            )
        );

        let up_idx = upsample_2x_source_index(&oy, &ox, 32);
        assert_eq!(
            up_idx,
            flat_index(Expr::div(oy, Expr::u32(2)), 32, Expr::div(ox, Expr::u32(2)))
        );
    }

    #[test]
    fn rgba_pack_unpack_and_blend() {
        let r = Expr::var("r");
        let g = Expr::var("g");
        let b = Expr::var("b");
        let a = Expr::var("a");
        let packed = pack_rgba(r.clone(), g.clone(), b.clone(), a.clone());
        assert_eq!(
            packed,
            Expr::bitor(
                Expr::bitor(r, Expr::shl(g, Expr::u32(8))),
                Expr::bitor(Expr::shl(b, Expr::u32(16)), Expr::shl(a, Expr::u32(24))),
            )
        );

        let (ur, ug, ub, ua) = unpack_rgba("px");
        assert_eq!(ur, Expr::bitand(Expr::var("px"), Expr::u32(0xFF)));
        assert_eq!(
            ug,
            Expr::bitand(Expr::shr(Expr::var("px"), Expr::u32(8)), Expr::u32(0xFF))
        );
        assert_eq!(
            ub,
            Expr::bitand(Expr::shr(Expr::var("px"), Expr::u32(16)), Expr::u32(0xFF))
        );
        assert_eq!(ua, Expr::shr(Expr::var("px"), Expr::u32(24)));

        let clamped = clamp_u8(Expr::var("val"));
        assert_eq!(
            clamped,
            Expr::select(
                Expr::gt(Expr::var("val"), Expr::u32(255)),
                Expr::u32(255),
                Expr::var("val")
            )
        );
    }

    #[test]
    fn grid_2d_composer_builds_program() {
        let composer = Grid2DComposer::new("test::op", 16, 16)
            .with_buffers(vec![
                BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(256),
                BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(256),
            ])
            .with_workgroup_size([64, 1, 1]);

        let program = composer.build(|_shape, idx, py, px| {
            vec![
                Node::let_bind("py", py),
                Node::let_bind("px", px),
                Node::store("out", idx, Expr::u32(42)),
            ]
        });

        assert_eq!(program.workgroup_size, [64, 1, 1]);
        assert_eq!(program.buffers().len(), 2);
        assert_eq!(program.buffers()[0].count(), 256);
        assert_eq!(program.buffers()[1].count(), 256);
    }
}
