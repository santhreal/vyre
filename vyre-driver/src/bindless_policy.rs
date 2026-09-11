//! D9 substrate: bindless buffers / textures decision policy.
//!
//! When a kernel binds many resources (think 100+ small buffers in a
//! sparse compute graph), the per-binding setup cost  -  bind group
//! creation, descriptor set rebinds  -  dominates dispatch latency.
//! Bindless mode replaces N descriptor entries with one descriptor
//! array indexed at runtime, eliminating the rebind churn.
//!
//! Concrete backends expose bindless access through their own native
//! resource-indexing primitives. Not every adapter supports it; the
//! policy here owns the decision given a probed capability + resource
//! count.
//!
//! Pure decision: no Program walk, no descriptor scan. Caller passes
//! the resource count and the backend's bindless capability bit.

/// Backend support level for bindless resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindlessSupport {
    /// Backend has full bindless support: descriptor arrays plus
    /// dynamic indexing.
    Full,
    /// Backend supports descriptor arrays but with a fixed size and no
    /// runtime indexing of unbound slots. Useful when every slot is
    /// guaranteed bound; not useful for sparse access.
    Static,
    /// Backend has no bindless support. Always use traditional
    /// per-resource bindings.
    Unsupported,
}

/// Inputs to the bindless decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindlessInputs {
    /// Number of resources the kernel binds. Below the threshold,
    /// traditional bindings beat bindless on every backend (the
    /// per-bindless-handle setup cost has its own constant).
    pub resource_count: u32,
    /// Backend's bindless support level (probed once per backend
    /// startup).
    pub support: BindlessSupport,
    /// Whether the kernel's access pattern is dynamic (different
    /// indices per thread / per dispatch). Only `Full` support
    /// handles dynamic indexing; `Static` is wasted on dynamic
    /// access.
    pub dynamic_indexing: bool,
}

/// Verdict from [`decide_bindless`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindlessDecision {
    /// Use bindless  -  N resources go into a single descriptor array.
    Bindless,
    /// Use traditional per-resource bindings.
    TraditionalBindings,
}

/// Threshold above which bindless wins on `Full` support backends.
/// Below this count the per-handle bindless setup overhead dominates.
/// Calibrated from backend microbenchmarks: around two dozen bindings
/// is the crossover on current discrete GPUs.
pub const BINDLESS_RESOURCE_COUNT_THRESHOLD: u32 = 24;

/// Decide whether to use the bindless path for this dispatch.
///
/// Picks `Bindless` when:
///   - support is `Full`, AND
///   - resource_count >= [`BINDLESS_RESOURCE_COUNT_THRESHOLD`]
///
/// `Static` support is treated as `Bindless` only when the access
/// pattern is NOT dynamic (every slot is guaranteed bound) AND the
/// resource count clears the threshold. `Unsupported` always returns
/// `TraditionalBindings`.
#[must_use]
pub fn decide_bindless(inputs: BindlessInputs) -> BindlessDecision {
    if matches!(inputs.support, BindlessSupport::Unsupported) {
        return BindlessDecision::TraditionalBindings;
    }
    if inputs.resource_count < BINDLESS_RESOURCE_COUNT_THRESHOLD {
        return BindlessDecision::TraditionalBindings;
    }
    match inputs.support {
        BindlessSupport::Full => BindlessDecision::Bindless,
        BindlessSupport::Static if !inputs.dynamic_indexing => BindlessDecision::Bindless,
        _ => BindlessDecision::TraditionalBindings,
    }
}
