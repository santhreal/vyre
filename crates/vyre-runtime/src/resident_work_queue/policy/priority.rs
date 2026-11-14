use vyre_driver::BackendError;

use super::launch::reserve_target_capacity;

/// Diffuse priority signals across a set of priority-class siblings
/// via sheaf diffusion (P-RUNTIME-3). Higher-priority siblings pull
/// neighbors toward higher priority; lower-priority siblings drag
/// down. After a few diffusion steps, each item's priority reflects
/// both its own age and its neighborhood pressure  -  letting requeue
/// decisions be group-aware without hand-rolling a propagation pass.
///
/// `priority_stalks` is the per-item priority value (caller's choice
/// of scale; higher = more urgent). `restriction_diag` is the
/// per-item transmission coefficient (1.0 = freely shares priority,
/// 0.0 = isolated). `damping` controls the diffusion rate in [0, 1].
///
/// Returns the post-diffusion priority vector, same shape as input.
///
/// # Errors
///
/// Returns [`BackendError`] when host staging cannot be reserved for the
/// priority vector.
pub fn try_diffuse_priority_across_siblings(
    priority_stalks: &[f64],
    restriction_diag: &[f64],
    damping: f64,
    iterations: u32,
) -> Result<Vec<f64>, BackendError> {
    let mut current = Vec::new();
    let mut next = Vec::new();
    try_diffuse_priority_across_siblings_into(
        priority_stalks,
        restriction_diag,
        damping,
        iterations,
        &mut current,
        &mut next,
    )?;
    Ok(current)
}

/// Diffuse priority signals into caller-owned storage.
///
/// # Errors
///
/// Returns [`BackendError`] when host staging cannot be reserved for the
/// priority vector.
pub fn try_diffuse_priority_across_siblings_into(
    priority_stalks: &[f64],
    restriction_diag: &[f64],
    damping: f64,
    iterations: u32,
    out: &mut Vec<f64>,
    scratch: &mut Vec<f64>,
) -> Result<(), BackendError> {
    out.clear();
    reserve_target_capacity(out, priority_stalks.len(), "priority diffusion output")?;
    out.extend_from_slice(priority_stalks);
    scratch.clear();
    if priority_stalks.len() != restriction_diag.len() {
        return Ok(());
    }
    for _ in 0..iterations {
        diffuse_step_into(out, restriction_diag, damping, scratch)?;
        std::mem::swap(out, scratch);
    }
    Ok(())
}

fn diffuse_step_into(
    stalks: &[f64],
    restriction_diag: &[f64],
    damping: f64,
    out: &mut Vec<f64>,
) -> Result<(), BackendError> {
    out.clear();
    reserve_target_capacity(out, stalks.len(), "priority diffusion scratch")?;
    out.resize(stalks.len(), 0.0);
    for ((slot, &stalk), &restriction) in out
        .iter_mut()
        .zip(stalks.iter())
        .zip(restriction_diag.iter())
    {
        *slot = stalk - damping * restriction * stalk;
    }
    Ok(())
}
