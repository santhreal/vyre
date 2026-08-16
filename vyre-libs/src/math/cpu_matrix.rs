//! Shared CPU parity kernels for matrix-shaped primitives.

#[cfg(any(test, feature = "cpu-parity"))]
pub(super) struct MatmulContext<'a> {
    pub operation: &'a str,
    pub left_name: &'a str,
    pub right_name: &'a str,
    pub output_name: &'a str,
    pub fix: &'a str,
    pub allocation_context: &'a str,
}

#[cfg(any(test, feature = "cpu-parity"))]
impl MatmulContext<'static> {
    pub(super) const RANDOMIZED_PROJECTION: Self = Self {
        operation: "randomized_projection_step",
        left_name: "A",
        right_name: "omega",
        output_name: "randomized_projection_step output",
        fix: "shard the randomized SVD matrix before parity evaluation.",
        allocation_context: "randomized SVD CPU oracle",
    };

    pub(super) const TENSOR_CONTRACT: Self = Self {
        operation: "tn_pair_contract",
        left_name: "A",
        right_name: "B",
        output_name: "tn_pair_contract output",
        fix: "shard the tensor before parity evaluation.",
        allocation_context: "tensor-network CPU oracle",
    };
}

#[cfg(any(test, feature = "cpu-parity"))]
pub(super) fn try_f64_matmul_into(
    left: &[f64],
    right: &[f64],
    rows: u32,
    inner: u32,
    columns: u32,
    output: &mut Vec<f64>,
    context: MatmulContext<'_>,
) -> Result<(), String> {
    let rows = rows as usize;
    let inner = inner as usize;
    let columns = columns as usize;
    rows.checked_mul(inner).ok_or_else(|| {
        format!(
            "{} CPU oracle {} shape {rows}x{inner} overflows indexing. Fix: {}",
            context.operation, context.left_name, context.fix
        )
    })?;
    inner.checked_mul(columns).ok_or_else(|| {
        format!(
            "{} CPU oracle {} shape {inner}x{columns} overflows indexing. Fix: {}",
            context.operation, context.right_name, context.fix
        )
    })?;
    let cells = rows.checked_mul(columns).ok_or_else(|| {
        format!(
            "{} CPU oracle output shape {rows}x{columns} overflows indexing. Fix: {}",
            context.operation, context.fix
        )
    })?;
    if cells > output.capacity() {
        crate::plumbing::host::scratch::reserve_items(
            output,
            cells - output.len(),
            context.allocation_context,
            context.output_name,
        )?;
    }
    output.clear();
    output.resize(cells, 0.0);
    for row in 0..rows {
        for column in 0..columns {
            let mut accumulator = 0.0;
            for component in 0..inner {
                let left_value = left.get(row * inner + component).copied().unwrap_or(0.0);
                let right_value = right
                    .get(component * columns + column)
                    .copied()
                    .unwrap_or(0.0);
                accumulator += left_value * right_value;
            }
            output[row * columns + column] = accumulator;
        }
    }
    Ok(())
}
