    mod cell_grid {
        use vyre_libs::visual::{cell_grid_fill, GridShape};
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

        /// Execute the op and return the surface it wrote.
        fn run(shape: GridShape, cells: &[u32]) -> Vec<u32> {
            assert_eq!(
                cells.len() as u32,
                shape.cell_count(),
                "fixture must supply one colour per cell"
            );
            let program = cell_grid_fill("cells", "out", shape);
            let outputs = vyre_reference::reference_eval(
                &program,
                &[
                    Value::from(le_bytes(cells)),
                    Value::from(vec![0u8; shape.pixel_count() as usize * 4]),
                ],
            )
            .expect("cell_grid_fill must execute under the reference oracle");
            assert_eq!(outputs.len(), 1, "only the output buffer is ReadWrite");
            words(&outputs[0].to_bytes())
        }

        /// The mapping, written independently of the IR: find the pixel's
        /// column and row on the surface, then the cell that covers it.
        fn oracle(shape: GridShape, cells: &[u32]) -> Vec<u32> {
            let width = shape.width();
            (0..shape.pixel_count())
                .map(|idx| {
                    let x = idx % width;
                    let y = idx / width;
                    let cell = (y / shape.cell_height) * shape.cols + (x / shape.cell_width);
                    cells[cell as usize]
                })
                .collect()
        }

        const R: u32 = 0xFF00_00FF;
        const G: u32 = 0xFF00_FF00;
        const B: u32 = 0xFFFF_0000;
        const W: u32 = 0xFFFF_FFFF;

        /// The one case pinned to hand-computed pixels rather than to the
        /// oracle, so that a shared misreading of the mapping cannot make the
        /// oracle and the op agree on the wrong answer.
        #[test]
        fn each_cell_paints_its_own_block_not_its_own_row() {
            let shape = GridShape {
                cols: 2,
                rows: 2,
                cell_width: 2,
                cell_height: 2,
            };
            #[rustfmt::skip]
            let expected = vec![
                R, R, G, G,
                R, R, G, G,
                B, B, W, W,
                B, B, W, W,
            ];
            assert_eq!(run(shape, &[R, G, B, W]), expected);
        }

        #[test]
        fn a_non_square_grid_of_non_square_cells_matches_the_oracle() {
            let shape = GridShape {
                cols: 3,
                rows: 2,
                cell_width: 2,
                cell_height: 3,
            };
            let cells = [R, G, B, W, 0x8012_3456, 0x0000_00FF];
            assert_eq!(run(shape, &cells), oracle(shape, &cells));
        }

        /// A cell one pixel across is the degenerate case where the surface
        /// and the grid are the same size, so the op must be the identity.
        #[test]
        fn one_pixel_cells_copy_the_grid_verbatim() {
            let shape = GridShape {
                cols: 4,
                rows: 3,
                cell_width: 1,
                cell_height: 1,
            };
            let cells: Vec<u32> = (0..12).map(|n| 0xFF00_0000 | n).collect();
            assert_eq!(run(shape, &cells), cells);
        }

        #[test]
        fn a_single_cell_covers_every_pixel() {
            let shape = GridShape {
                cols: 1,
                rows: 1,
                cell_width: 5,
                cell_height: 4,
            };
            assert_eq!(run(shape, &[G]), vec![G; 20]);
        }

        /// The corners are where an off-by-one in the row or column arithmetic
        /// shows up first: the last pixel must come from the last cell, never
        /// from a wrapped index or a neighbour.
        #[test]
        fn the_last_pixel_comes_from_the_last_cell() {
            let shape = GridShape {
                cols: 8,
                rows: 4,
                cell_width: 3,
                cell_height: 2,
            };
            let cells: Vec<u32> = (0..32).map(|n| 0xFF00_0000 | (n * 7 + 1)).collect();
            let surface = run(shape, &cells);
            assert_eq!(surface.len(), 24 * 8);
            assert_eq!(surface[0], cells[0], "first pixel is the first cell");
            assert_eq!(
                *surface.last().expect("surface is non-empty"),
                *cells.last().expect("grid is non-empty"),
                "last pixel is the last cell"
            );
            assert_eq!(surface, oracle(shape, &cells));
        }

        /// The real grid dimensions the op exists for: a full 80x24 terminal,
        /// 1,920 cells. The cells are kept small because the oracle here is a
        /// CPU interpreter, and the mapping under test depends on the grid
        /// shape rather than on how many pixels a cell owns.
        #[test]
        fn a_terminal_grid_matches_the_oracle() {
            let shape = GridShape {
                cols: 80,
                rows: 24,
                cell_width: 2,
                cell_height: 2,
            };
            assert_eq!(shape.cell_count(), 1_920);
            assert_eq!(shape.pixel_count(), 160 * 48);
            let cells: Vec<u32> = (0..shape.cell_count())
                .map(|n| 0xFF00_0000 | n.wrapping_mul(2_654_435_761) >> 8)
                .collect();
            assert_eq!(run(shape, &cells), oracle(shape, &cells));
        }

        #[test]
        #[should_panic(expected = "at least one row and one column")]
        fn an_empty_grid_is_refused() {
            let _ = cell_grid_fill(
                "cells",
                "out",
                GridShape {
                    cols: 0,
                    rows: 4,
                    cell_width: 2,
                    cell_height: 2,
                },
            );
        }

        /// A zero-sized cell would divide by zero on the device, where the
        /// failure is a wrong pixel rather than a message.
        #[test]
        #[should_panic(expected = "non-zero size")]
        fn a_zero_sized_cell_is_refused() {
            let _ = cell_grid_fill(
                "cells",
                "out",
                GridShape {
                    cols: 4,
                    rows: 4,
                    cell_width: 2,
                    cell_height: 0,
                },
            );
        }

        #[test]
        #[should_panic(expected = "overflows u32")]
        fn a_surface_too_large_to_index_is_refused() {
            let _ = cell_grid_fill(
                "cells",
                "out",
                GridShape {
                    cols: 100_000,
                    rows: 100_000,
                    cell_width: 16,
                    cell_height: 16,
                },
            );
        }
    }
