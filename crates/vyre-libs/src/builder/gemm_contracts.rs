//! Contract tests for [`super::ContractionComposer`].
//!
//! Split out of `gemm.rs`, alongside `gemm_algebra.rs` and `gemm_programs.rs`,
//! so the composer and its cases are separate files.

use super::*;

mod cases {
    use super::*;
    use vyre_foundation::ir::{BufferAccess, BufferDecl};
    #[test]
    fn test_contraction_composer_2d_matmul_u32() {
        let a = TensorRef::u32_2d("a", 2, 3);
        let b = TensorRef::u32_2d("b", 3, 2);
        let out = TensorRef::u32_2d("out", 2, 2);
        let program = ContractionComposer::matmul_2d("test_op", a, b, out, 2, 3, 2)
            .build()
            .expect("matmul_2d should build");
        assert_eq!(program.workgroup_size(), [256, 1, 1]);
        assert_eq!(program.buffers().len(), 3);
    }

    #[test]
    fn test_contraction_composer_2d_matmul_bias_u32() {
        let a = TensorRef::u32_2d("a", 2, 3);
        let b = TensorRef::u32_2d("b", 3, 2);
        let bias = TensorRef::u32_1d("bias", 2);
        let out = TensorRef::u32_2d("out", 2, 2);
        let program = ContractionComposer::matmul_bias_2d("test_op", a, b, bias, out, 2, 3, 2)
            .build()
            .expect("matmul_bias_2d should build");
        assert_eq!(program.buffers().len(), 4);
    }

    #[test]
    fn test_contraction_composer_tiled_2d() {
        let a = TensorRef::u32_2d("a", 16, 16);
        let b = TensorRef::u32_2d("b", 16, 16);
        let out = TensorRef::u32_2d("out", 16, 16);
        let program = ContractionComposer::tiled_2d("test_tiled", a, b, None, out, 16, 16, 16, 16)
            .build()
            .expect("tiled_2d should build");
        assert!(program
            .buffers()
            .iter()
            .any(|b| matches!(b.access, BufferAccess::Workgroup)));
    }

    #[test]
    fn test_contraction_composer_semirings() {
        for semiring in [
            Semiring::Real,
            Semiring::MinPlus,
            Semiring::MaxPlus,
            Semiring::BoolOr,
            Semiring::BoolAnd,
            Semiring::MaxTimes,
            Semiring::Lineage,
            Semiring::Gf2,
        ] {
            let a = TensorRef::u32_2d("a", 2, 2);
            let b = TensorRef::u32_2d("b", 2, 2);
            let out = TensorRef::u32_2d("out", 2, 2);
            let program =
                ContractionComposer::semiring_2d("test_semiring", a, b, out, 2, 2, 2, semiring)
                    .build()
                    .expect("semiring_2d should build");
            assert_eq!(program.buffers().len(), 3);
        }
    }

    /// Every `ContractionEpilogue` must be reachable through `build`, and each
    /// one that reads a buffer must declare it.
    ///
    /// The match has no catch-all, so an added epilogue fails to compile here
    /// until its buffer requirement is decided.
    #[test]
    fn each_epilogue_declares_every_buffer_its_expression_reads() {
        let epilogues = [
            ContractionEpilogue::None,
            ContractionEpilogue::Bias {
                buffer: "bias".into(),
                count: 2,
                dtype: DataType::U32,
            },
            ContractionEpilogue::Activation {
                bias: Some("bias".into()),
                activation: Arc::new(|expr| Expr::max(expr, Expr::u32(0))),
            },
            ContractionEpilogue::QuantizedScale {
                row_scales: "r_scale".into(),
                batch_scales: "b_scale".into(),
            },
        ];

        for epilogue in epilogues {
            let reads: Vec<String> = match &epilogue {
                ContractionEpilogue::None => Vec::new(),
                ContractionEpilogue::Bias { buffer, .. } => vec![buffer.clone()],
                ContractionEpilogue::Activation { bias, .. } => bias.iter().cloned().collect(),
                ContractionEpilogue::QuantizedScale {
                    row_scales,
                    batch_scales,
                } => vec![row_scales.clone(), batch_scales.clone()],
            };

            let composer = ContractionComposer::matmul_bias_2d(
                "epilogue_closure",
                TensorRef::u32_2d("a", 2, 3),
                TensorRef::u32_2d("b", 3, 2),
                TensorRef::u32_1d("bias", 2),
                TensorRef::u32_2d("out", 2, 2),
                2,
                3,
                2,
            )
            .with_epilogue(epilogue.clone());
            let program = composer
                .build()
                .unwrap_or_else(|error| panic!("Fix: {epilogue:?} must build: {error}"));

            for buffer in &reads {
                assert!(
                    program.buffers().iter().any(|decl| decl.name() == buffer),
                    "{epilogue:?} reads `{buffer}` but the program declares only {:?}",
                    program
                        .buffers()
                        .iter()
                        .map(BufferDecl::name)
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    /// Every tiling and epilogue combination `build` accepts must produce a
    /// program the workspace validator accepts.
    ///
    /// This is the choke point for the whole class: a composition that reads an
    /// undeclared buffer (V065) or an unbound variable (V066) is caught here
    /// regardless of which arm built it, so a new tiling or epilogue does not
    /// need its own test to be covered.
    #[test]
    fn every_tiling_and_epilogue_builds_a_program_the_validator_accepts() {
        let tilings = [
            ContractionTiling::Linear {
                workgroup_size: [64, 1, 1],
            },
            ContractionTiling::CooperativeShared {
                tile: 16,
                a_tile_name: "a_shared".to_string(),
                b_tile_name: "b_shared".to_string(),
            },
            ContractionTiling::Block1D { tile: 8 },
        ];
        let epilogues = [
            ContractionEpilogue::None,
            ContractionEpilogue::Bias {
                buffer: "bias".into(),
                count: 16,
                dtype: DataType::U32,
            },
            ContractionEpilogue::Activation {
                bias: Some("bias".into()),
                activation: Arc::new(|expr| Expr::max(expr, Expr::u32(0))),
            },
            ContractionEpilogue::Activation {
                bias: None,
                activation: Arc::new(|expr| Expr::max(expr, Expr::u32(0))),
            },
            ContractionEpilogue::QuantizedScale {
                row_scales: "r_scale".into(),
                batch_scales: "b_scale".into(),
            },
        ];

        for tiling in &tilings {
            for epilogue in &epilogues {
                let program = ContractionComposer::matmul_bias_2d(
                    "validator_closure",
                    TensorRef::u32_2d("a", 16, 16),
                    TensorRef::u32_2d("b", 16, 16),
                    TensorRef::u32_1d("bias", 16),
                    TensorRef::u32_2d("out", 16, 16),
                    16,
                    16,
                    16,
                )
                .with_tiling(tiling.clone())
                .with_epilogue(epilogue.clone())
                .build()
                .unwrap_or_else(|error| {
                    panic!("Fix: {tiling:?} with {epilogue:?} must build: {error}")
                });

                let errors = vyre_foundation::validate::validate(&program);
                assert!(
                    errors.is_empty(),
                    "{tiling:?} with {epilogue:?} built an invalid program: {errors:?}"
                );
            }
        }
    }

    /// Every element type the composer accepts must accumulate at that type's
    /// own width, on every tiling.
    ///
    /// A `u32` accumulator seed is silently correct for `u32` and wrong for
    /// every other width, which is why the tiling and epilogue closure above
    /// cannot see it: that test contracts `u32`. The IR validator rejects the
    /// mistyped store the seed feeds.
    #[test]
    fn every_element_type_accumulates_at_its_own_width() {
        let dtypes = [
            DataType::U32,
            DataType::I32,
            DataType::F32,
            DataType::F16,
            DataType::BF16,
        ];
        let tilings = [
            ContractionTiling::Linear {
                workgroup_size: [64, 1, 1],
            },
            ContractionTiling::CooperativeShared {
                tile: 16,
                a_tile_name: "a_shared".to_string(),
                b_tile_name: "b_shared".to_string(),
            },
            ContractionTiling::Block1D { tile: 8 },
        ];

        for dtype in &dtypes {
            for tiling in &tilings {
                let program = ContractionComposer::matmul_2d(
                    "dtype_closure",
                    TensorRef::new("a", dtype.clone(), vec![16, 16]),
                    TensorRef::new("b", dtype.clone(), vec![16, 16]),
                    TensorRef::new("out", dtype.clone(), vec![16, 16]),
                    16,
                    16,
                    16,
                )
                .with_tiling(tiling.clone())
                .build()
                .unwrap_or_else(|error| {
                    panic!("Fix: {dtype:?} with {tiling:?} must build: {error}")
                });

                let errors = vyre_foundation::validate::validate(&program);
                assert!(
                    errors.is_empty(),
                    "{dtype:?} with {tiling:?} built an invalid program: {errors:?}"
                );
            }
        }
    }

    /// Every `ContractionTiling` must state what the 2D geometry does with it:
    /// build with exactly the workgroup tiles it names, or be rejected as a
    /// tiling that geometry has no program for.
    ///
    /// The match has no catch-all, so an added tiling fails to compile until
    /// its memory requirement and its admissibility are decided.
    #[test]
    fn each_tiling_allocates_workgroup_memory_only_when_it_stages_tiles() {
        let tilings = [
            ContractionTiling::Linear {
                workgroup_size: [64, 1, 1],
            },
            ContractionTiling::CooperativeShared {
                tile: 16,
                a_tile_name: "a_shared".to_string(),
                b_tile_name: "b_shared".to_string(),
            },
            ContractionTiling::RegisterTiled {
                rows: 2,
                columns: 2,
                workgroup_size: [64, 1, 1],
            },
            ContractionTiling::Block1D { tile: 8 },
        ];

        for tiling in tilings {
            // `None` states that the 2D geometry has no program for this
            // tiling and must reject it instead of reinterpreting it.
            let expected_shared: Option<Vec<String>> = match &tiling {
                ContractionTiling::Linear { .. } | ContractionTiling::Block1D { .. } => {
                    Some(Vec::new())
                }
                ContractionTiling::CooperativeShared {
                    a_tile_name,
                    b_tile_name,
                    ..
                } => Some(vec![a_tile_name.clone(), b_tile_name.clone()]),
                ContractionTiling::RegisterTiled { .. } => None,
            };

            let built = ContractionComposer::matmul_2d(
                "tiling_closure",
                TensorRef::u32_2d("a", 16, 16),
                TensorRef::u32_2d("b", 16, 16),
                TensorRef::u32_2d("out", 16, 16),
                16,
                16,
                16,
            )
            .with_tiling(tiling.clone())
            .build();

            let Some(expected_shared) = expected_shared else {
                let error = built.expect_err(&format!(
                    "Fix: {tiling:?} has no 2D contraction program and must be rejected"
                ));
                assert!(
                    matches!(error, TensorRefError::UnsupportedTiling { .. }),
                    "Fix: {tiling:?} must be rejected as an unsupported tiling, got {error}"
                );
                continue;
            };

            let program =
                built.unwrap_or_else(|error| panic!("Fix: {tiling:?} must build: {error}"));

            let shared: Vec<String> = program
                .buffers()
                .iter()
                .filter(|decl| matches!(decl.access, BufferAccess::Workgroup))
                .map(|decl| decl.name().to_string())
                .collect();
            assert_eq!(
                shared, expected_shared,
                "{tiling:?} must allocate exactly the workgroup tiles it names"
            );
        }
    }

    /// A linear tiling folds a 3D workgroup declaration into one dimension,
    /// because each invocation computes one output element off a 1D index.
    #[test]
    fn a_linear_tiling_folds_its_workgroup_into_one_dimension() {
        let program = ContractionComposer::matmul_2d(
            "linear_fold",
            TensorRef::u32_2d("a", 2, 3),
            TensorRef::u32_2d("b", 3, 2),
            TensorRef::u32_2d("out", 2, 2),
            2,
            3,
            2,
        )
        .with_tiling(ContractionTiling::Linear {
            workgroup_size: [8, 4, 2],
        })
        .build()
        .expect("Fix: a linear tiling must build");
        assert_eq!(
            program.workgroup_size(),
            [64, 1, 1],
            "8 * 4 * 2 invocations must be declared as a single linear dimension"
        );
    }

    #[test]
    fn test_contraction_composer_batched_3d() {
        let a = TensorRef::new("a", DataType::F32, vec![2, 3, 4]);
        let b = TensorRef::new("b", DataType::F32, vec![2, 4, 5]);
        let out = TensorRef::new("out", DataType::F32, vec![2, 3, 5]);
        let program = ContractionComposer::batched_matmul_3d("test_batch", a, b, out, 2, 3, 4, 5)
            .build()
            .expect("batched_matmul_3d should build");
        assert_eq!(program.buffers().len(), 3);
    }

    #[test]
    fn test_contraction_composer_batched_rows() {
        let x = TensorRef::new("x", DataType::F32, vec![4, 8]);
        let w = TensorRef::new("w", DataType::F32, vec![8, 16]);
        let bias = TensorRef::new("b", DataType::F32, vec![16]);
        let out = TensorRef::new("out", DataType::F32, vec![4, 16]);
        let program = ContractionComposer::batched_rows(
            "test_rows",
            x,
            w,
            Some(bias),
            out,
            4,
            8,
            16,
            DataType::F32,
            false,
        )
        .build()
        .expect("batched_rows should build");
        assert_eq!(program.buffers().len(), 4);
    }

    #[test]
    fn test_contraction_composer_fixed_matvec() {
        let m = TensorRef::u32_2d("matrix", 4, 4);
        let v = TensorRef::u32_1d("vector", 4);
        let out = TensorRef::u32_1d("out", 4);
        let program = ContractionComposer::fixed_u32_matvec("test_matvec", m, v, out, 4, 16)
            .build()
            .expect("fixed_u32_matvec should build");
        assert_eq!(program.buffers().len(), 3);
    }

    #[test]
    fn test_contraction_composer_strassen_2x2() {
        let a = TensorRef::f32_2d("a", 2, 2);
        let b = TensorRef::f32_2d("b", 2, 2);
        let c = TensorRef::f32_2d("c", 2, 2);
        let mut composer = ContractionComposer::matmul_2d("test_strassen", a, b, c, 2, 2, 2);
        composer.geometry = ContractionGeometry::Strassen2x2;
        let program = composer.build().expect("strassen 2x2 should build");
        assert_eq!(program.buffers().len(), 3);
    }

    #[test]
    fn test_contraction_composer_strassen_one_level() {
        let a = TensorRef::f32_2d("a", 4, 4);
        let b = TensorRef::f32_2d("b", 4, 4);
        let c = TensorRef::f32_2d("c", 4, 4);
        let mut composer = ContractionComposer::matmul_2d("test_strassen_4", a, b, c, 4, 4, 4);
        composer.geometry = ContractionGeometry::StrassenOneLevel { n: 4 };
        let program = composer.build().expect("strassen one level should build");
        assert_eq!(program.buffers().len(), 3);
    }

    #[test]
    fn test_contraction_composer_rejects_zero_dims() {
        let a = TensorRef::u32_2d("a", 0, 4);
        let b = TensorRef::u32_2d("b", 4, 4);
        let out = TensorRef::u32_2d("out", 0, 4);
        let err = ContractionComposer::matmul_2d("test_err", a, b, out, 0, 4, 4)
            .build()
            .expect_err("zero dim must fail");
        assert!(matches!(err, TensorRefError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_contraction_composer_rejects_shared_dim_mismatch() {
        let a = TensorRef::u32_2d("a", 4, 3);
        let b = TensorRef::u32_2d("b", 5, 4);
        let out = TensorRef::u32_2d("out", 4, 4);
        let err = ContractionComposer::matmul_2d("test_err", a, b, out, 4, 3, 4)
            .build()
            .expect_err("shared dim mismatch must fail");
        assert!(matches!(err, TensorRefError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_contraction_composer_rejects_tile_zero() {
        let a = TensorRef::u32_2d("a", 4, 4);
        let b = TensorRef::u32_2d("b", 4, 4);
        let out = TensorRef::u32_2d("out", 4, 4);
        let err = ContractionComposer::tiled_2d("test_tiled_zero", a, b, None, out, 4, 4, 4, 0)
            .build()
            .expect_err("tile=0 must fail");
        assert!(matches!(err, TensorRefError::ShapeMismatch { .. }));
    }
}
