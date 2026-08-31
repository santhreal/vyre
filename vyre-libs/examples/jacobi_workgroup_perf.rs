//! Measured performance evidence comparing 1-lane baseline vs 64-lane cooperative Jacobi paths.

use std::time::Instant;
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_libs::math::eigenvector_column_sign::{
    eigenvector_column_sign, eigenvector_column_sign_body, OP_ID as SIGN_OP_ID,
};
use vyre_libs::math::matrix_diagonal_extract::{matrix_diagonal_extract, OP_ID as DIAG_OP_ID};
use vyre_libs::math::matrix_identity_fill::{matrix_identity_fill, OP_ID as IDENTITY_OP_ID};
use vyre_libs::math::symmetric_eigen_jacobi::symmetric_eigen_jacobi;
use vyre_primitives::wire::pack_f32_slice as pack_f32;
use vyre_reference::value::Value;

// Baseline 1-lane serial builders
fn serial_identity_fill(matrix: &str, n: u32) -> Program {
    let cells = n * n;
    Program::wrapped(
        vec![BufferDecl::output(matrix, 0, DataType::F32).with_count(cells)],
        [1, 1, 1],
        vec![wrap_anonymous_region(
            IDENTITY_OP_ID,
            vec![Node::loop_for(
                "mif_row",
                Expr::u32(0),
                Expr::u32(n),
                vec![Node::loop_for(
                    "mif_col",
                    Expr::u32(0),
                    Expr::u32(n),
                    vec![Node::store(
                        matrix,
                        Expr::add(
                            Expr::mul(Expr::var("mif_row"), Expr::u32(n)),
                            Expr::var("mif_col"),
                        ),
                        Expr::select(
                            Expr::eq(Expr::var("mif_row"), Expr::var("mif_col")),
                            Expr::f32(1.0),
                            Expr::f32(0.0),
                        ),
                    )],
                )],
            )],
        )],
    )
}

fn serial_diagonal_extract(matrix: &str, diagonal: &str, n: u32) -> Program {
    let cells = n * n;
    Program::wrapped(
        vec![
            BufferDecl::storage(matrix, 0, BufferAccess::ReadOnly, DataType::F32).with_count(cells),
            BufferDecl::output(diagonal, 1, DataType::F32).with_count(n),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(
            DIAG_OP_ID,
            vec![Node::loop_for(
                "mde_i",
                Expr::u32(0),
                Expr::u32(n),
                vec![Node::store(
                    diagonal,
                    Expr::var("mde_i"),
                    Expr::load(
                        matrix,
                        Expr::add(
                            Expr::mul(Expr::var("mde_i"), Expr::u32(n)),
                            Expr::var("mde_i"),
                        ),
                    ),
                )],
            )],
        )],
    )
}

fn serial_column_sign(eigenvectors: &str, n: u32) -> Program {
    let cells = n * n;
    let mut body = vec![Node::let_bind("local", Expr::u32(0))];
    body.extend(eigenvector_column_sign_body(eigenvectors, n, 1));
    Program::wrapped(
        vec![
            BufferDecl::storage(eigenvectors, 0, BufferAccess::ReadWrite, DataType::F32)
                .with_count(cells),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(SIGN_OP_ID, body)],
    )
}

fn main() {
    println!("=== Jacobi Workgroup Optimization: Measured Performance Evidence ===");
    println!("Workgroup Width: 64 lanes vs 1-lane baseline\n");

    // 1. matrix_identity_fill
    println!("--- 1. matrix_identity_fill ---");
    for &n in &[4u32, 8, 16, 32] {
        let cells = (n * n) as usize;
        let serial_prog = serial_identity_fill("m", n);
        let coop_prog = matrix_identity_fill("m", n);
        let val = [Value::from(pack_f32(&vec![0.0f32; cells]))];
        let iterations = if n <= 8 {
            50
        } else if n <= 16 {
            10
        } else {
            3
        };

        // Warmup
        let _ = vyre_reference::reference_eval(&serial_prog, &val);
        let _ = vyre_reference::reference_eval(&coop_prog, &val);

        let t0 = Instant::now();
        for _ in 0..iterations {
            let _ = vyre_reference::reference_eval(&serial_prog, &val);
        }
        let serial_dur = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..iterations {
            let _ = vyre_reference::reference_eval(&coop_prog, &val);
        }
        let coop_dur = t1.elapsed();

        let serial_us = serial_dur.as_secs_f64() * 1e6 / (iterations as f64);
        let coop_us = coop_dur.as_secs_f64() * 1e6 / (iterations as f64);
        let speedup = serial_us / coop_us;
        println!(
            "n = {:2} ({:4} cells): serial = {:9.2} µs | coop = {:9.2} µs | speedup = {:5.2}x",
            n, cells, serial_us, coop_us, speedup
        );
    }

    // 2. matrix_diagonal_extract
    println!("\n--- 2. matrix_diagonal_extract ---");
    for &n in &[4u32, 8, 16, 32, 64] {
        let cells = (n * n) as usize;
        let mut mat = vec![0.0f32; cells];
        for i in 0..cells {
            mat[i] = (i + 1) as f32;
        }
        let serial_prog = serial_diagonal_extract("m", "diag", n);
        let coop_prog = matrix_diagonal_extract("m", "diag", n);
        let val = [
            Value::from(pack_f32(&mat)),
            Value::from(pack_f32(&vec![0.0f32; n as usize])),
        ];
        let iterations = 50;

        // Warmup
        let _ = vyre_reference::reference_eval(&serial_prog, &val);
        let _ = vyre_reference::reference_eval(&coop_prog, &val);

        let t0 = Instant::now();
        for _ in 0..iterations {
            let _ = vyre_reference::reference_eval(&serial_prog, &val);
        }
        let serial_dur = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..iterations {
            let _ = vyre_reference::reference_eval(&coop_prog, &val);
        }
        let coop_dur = t1.elapsed();

        let serial_us = serial_dur.as_secs_f64() * 1e6 / (iterations as f64);
        let coop_us = coop_dur.as_secs_f64() * 1e6 / (iterations as f64);
        let speedup = serial_us / coop_us;
        println!(
            "n = {:2} ({:4} cells): serial = {:9.2} µs | coop = {:9.2} µs | speedup = {:5.2}x",
            n, cells, serial_us, coop_us, speedup
        );
    }

    // 3. eigenvector_column_sign
    println!("\n--- 3. eigenvector_column_sign ---");
    for &n in &[4u32, 8, 16, 32] {
        let cells = (n * n) as usize;
        let mut mat = vec![0.0f32; cells];
        for c in 0..(n as usize) {
            mat[0 * (n as usize) + c] = -1.0;
            for r in 1..(n as usize) {
                mat[r * (n as usize) + c] = (r + c) as f32;
            }
        }
        let serial_prog = serial_column_sign("evec", n);
        let coop_prog = eigenvector_column_sign("evec", n);
        let val = [Value::from(pack_f32(&mat))];
        let iterations = if n <= 8 {
            50
        } else if n <= 16 {
            10
        } else {
            3
        };

        // Warmup
        let _ = vyre_reference::reference_eval(&serial_prog, &val);
        let _ = vyre_reference::reference_eval(&coop_prog, &val);

        let t0 = Instant::now();
        for _ in 0..iterations {
            let _ = vyre_reference::reference_eval(&serial_prog, &val);
        }
        let serial_dur = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..iterations {
            let _ = vyre_reference::reference_eval(&coop_prog, &val);
        }
        let coop_dur = t1.elapsed();

        let serial_us = serial_dur.as_secs_f64() * 1e6 / (iterations as f64);
        let coop_us = coop_dur.as_secs_f64() * 1e6 / (iterations as f64);
        let speedup = serial_us / coop_us;
        println!(
            "n = {:2} ({:4} cells): serial = {:9.2} µs | coop = {:9.2} µs | speedup = {:5.2}x",
            n, cells, serial_us, coop_us, speedup
        );
    }

    // 4. symmetric_eigen_jacobi
    println!("\n--- 4. symmetric_eigen_jacobi ---");
    for &n in &[2u32, 4, 8] {
        let cells = (n * n) as usize;
        let mut a = vec![0.0f32; cells];
        for i in 0..(n as usize) {
            for j in 0..(n as usize) {
                a[i * (n as usize) + j] = if i == j { (i + 1) as f32 * 3.0 } else { 0.5 };
            }
        }
        let prog = symmetric_eigen_jacobi("a", "evec", "eval", n);
        let val = [
            Value::from(pack_f32(&a)),
            Value::from(pack_f32(&vec![0.0f32; cells])),
            Value::from(pack_f32(&vec![0.0f32; n as usize])),
        ];

        let _ = vyre_reference::reference_eval(&prog, &val);

        let iters = 10;
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = vyre_reference::reference_eval(&prog, &val);
        }
        let dur = t0.elapsed();
        let us = dur.as_secs_f64() * 1e6 / (iters as f64);
        println!(
            "n = {:2}: median execution time = {:9.2} µs across {} sweeps",
            n,
            us,
            vyre_libs::math::symmetric_eigen_jacobi::jacobi_sweeps(n)
        );
    }
}
