mod glyph_grid {
    use vyre_libs::visual::{glyph_grid_blend, GridShape};
    use vyre_reference::value::Value;

    fn le_bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    fn words(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    struct Grid {
        shape: GridShape,
        glyphs: Vec<u32>,
        fg: Vec<u32>,
        bg: Vec<u32>,
        atlas: Vec<u32>,
        glyph_count: u32,
    }

    fn run(grid: &Grid) -> Vec<u32> {
        let program = glyph_grid_blend(
            "glyphs",
            "fg",
            "bg",
            "atlas",
            "out",
            grid.shape,
            grid.glyph_count,
        );
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(le_bytes(&grid.glyphs)),
                Value::from(le_bytes(&grid.fg)),
                Value::from(le_bytes(&grid.bg)),
                Value::from(le_bytes(&grid.atlas)),
                Value::from(vec![0u8; grid.shape.pixel_count() as usize * 4]),
            ],
        )
        .expect("glyph_grid_blend must execute under the reference oracle");
        assert_eq!(outputs.len(), 1, "only the output buffer is ReadWrite");
        words(&outputs[0].to_bytes())
    }

    /// `(v + 128) * 257 >> 16`, the division by 255 the op uses.
    fn div255(value: u32) -> u32 {
        (value + 128).wrapping_mul(257) >> 16
    }

    /// The mapping and the blend, written independently of the IR.
    fn oracle(grid: &Grid) -> Vec<u32> {
        let shape = grid.shape;
        let width = shape.width();
        let area = shape.cell_width * shape.cell_height;
        (0..shape.pixel_count())
            .map(|idx| {
                let (x, y) = (idx % width, idx / width);
                let (col, row) = (x / shape.cell_width, y / shape.cell_height);
                let cell = (row * shape.cols + col) as usize;
                let (px, py) = (x % shape.cell_width, y % shape.cell_height);
                let texel = grid.glyphs[cell] * area + py * shape.cell_width + px;
                let cov = grid.atlas[texel as usize] & 0xFF;
                let inv = 255 - cov;
                let (fg, bg) = (grid.fg[cell], grid.bg[cell]);
                let mix = |shift: u32| {
                    let f = (fg >> shift) & 0xFF;
                    let b = (bg >> shift) & 0xFF;
                    div255(f * cov + b * inv)
                };
                mix(0) | (mix(8) << 8) | (mix(16) << 16) | (mix(24) << 24)
            })
            .collect()
    }

    const RED: u32 = 0xFF00_00FF;
    const BLUE: u32 = 0xFFFF_0000;

    fn one_cell(atlas: Vec<u32>, glyph_count: u32) -> Grid {
        Grid {
            shape: GridShape {
                cols: 1,
                rows: 1,
                cell_width: 2,
                cell_height: 2,
            },
            glyphs: vec![1],
            fg: vec![RED],
            bg: vec![BLUE],
            atlas,
            glyph_count,
        }
    }

    /// Pinned to hand-computed pixels rather than to the oracle, so a
    /// shared misreading of the blend cannot make the two agree on the
    /// wrong answer. Coverage 0 must leave the background untouched and
    /// coverage 255 must leave the foreground untouched: a blend that is
    /// off by one anywhere shows up at these two endpoints first.
    #[test]
    fn coverage_selects_between_background_and_foreground() {
        let grid = one_cell(vec![0, 0, 0, 0, 0, 255, 128, 255], 2);
        assert_eq!(
            run(&grid),
            vec![BLUE, RED, 0xFF7F_0080, RED],
            "expected bg, fg, a half-covered mix, fg"
        );
    }

    /// A blank glyph must reproduce the background exactly. This is the
    /// common case in a terminal, where most cells are spaces.
    #[test]
    fn a_blank_glyph_is_the_background() {
        let mut grid = one_cell(vec![0, 0, 0, 0], 1);
        grid.glyphs = vec![0];
        assert_eq!(run(&grid), vec![BLUE; 4]);
    }

    /// Full coverage must reproduce the foreground exactly, with no
    /// rounding drift leaking in from the background.
    #[test]
    fn a_fully_covered_glyph_is_the_foreground() {
        let grid = one_cell(vec![0, 0, 0, 0, 255, 255, 255, 255], 2);
        assert_eq!(run(&grid), vec![RED; 4]);
    }

    /// Each cell must sample its own glyph at its own offset. A grid of
    /// distinct glyphs catches an atlas index that ignores the cell.
    #[test]
    fn every_cell_samples_its_own_glyph() {
        let shape = GridShape {
            cols: 3,
            rows: 2,
            cell_width: 2,
            cell_height: 2,
        };
        let atlas: Vec<u32> = (0..6 * 4).map(|n: u32| (n * 11) % 256).collect();
        let grid = Grid {
            shape,
            glyphs: vec![0, 1, 2, 3, 4, 5],
            fg: vec![RED, BLUE, 0xFF00_FF00, 0xFFFF_FFFF, RED, BLUE],
            bg: vec![BLUE, RED, 0xFF12_3456, 0xFF00_0000, 0xFFAB_CDEF, RED],
            atlas,
            glyph_count: 6,
        };
        assert_eq!(run(&grid), oracle(&grid));
    }

    /// Non-square cells make a row-stride error visible: with a square
    /// cell, transposing the within-cell offsets still lands inside the
    /// glyph.
    #[test]
    fn non_square_cells_index_the_atlas_by_row_stride() {
        let shape = GridShape {
            cols: 2,
            rows: 2,
            cell_width: 3,
            cell_height: 5,
        };
        let atlas: Vec<u32> = (0..4 * 15).map(|n: u32| (n * 17) % 256).collect();
        let grid = Grid {
            shape,
            glyphs: vec![3, 2, 1, 0],
            fg: vec![RED, BLUE, 0xFF00_FF00, 0xFFFF_FFFF],
            bg: vec![BLUE, RED, 0xFF12_3456, 0xFF00_0000],
            atlas,
            glyph_count: 4,
        };
        assert_eq!(run(&grid), oracle(&grid));
    }

    /// A full terminal row of distinct cells, the shape the op exists for.
    #[test]
    fn a_terminal_row_matches_the_oracle() {
        let shape = GridShape {
            cols: 80,
            rows: 1,
            cell_width: 2,
            cell_height: 2,
        };
        let glyph_count = 16u32;
        let atlas: Vec<u32> = (0..glyph_count * 4).map(|n| (n * 37) % 256).collect();
        let grid = Grid {
            shape,
            glyphs: (0..80).map(|n| n % glyph_count).collect(),
            fg: (0..80).map(|n: u32| 0xFF00_0000 | n.wrapping_mul(2_654_435_761) >> 8).collect(),
            bg: (0..80).map(|n: u32| 0xFF00_0000 | n.wrapping_mul(40_503) >> 8).collect(),
            atlas,
            glyph_count,
        };
        assert_eq!(run(&grid), oracle(&grid));
    }

    #[test]
    #[should_panic(expected = "at least one glyph")]
    fn an_empty_atlas_is_refused() {
        let _ = glyph_grid_blend(
            "glyphs",
            "fg",
            "bg",
            "atlas",
            "out",
            GridShape {
                cols: 2,
                rows: 2,
                cell_width: 2,
                cell_height: 2,
            },
            0,
        );
    }
}
