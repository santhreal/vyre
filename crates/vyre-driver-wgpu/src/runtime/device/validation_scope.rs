//! Validation error scopes that stay with the caller that pushed them.

use std::future::Future;
use std::hash::{BuildHasher, BuildHasherDefault};
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Wake, Waker};

/// Lock stripes for error-scope regions, selected by device hash.
///
/// wgpu 25 keeps one error-scope stack per device, shared by every thread that
/// holds a clone of it. Two threads pushing and popping on one device without
/// coordination pop each other's scopes, so a validation error is reported to
/// the caller that did not cause it and the caller that did returns success.
/// Every device this driver acquires is shared by every caller of its adapter,
/// so each scope region holds its device's stripe from push to pop. Two devices
/// that hash to one stripe serialize their scope regions and stay correct.
static SCOPE_STRIPES: [Mutex<()>; 16] = [const { Mutex::new(()) }; 16];

/// One validation error scope on `device`, popped exactly once.
///
/// [`ValidationScope::pop_now`] pops it and returns the captured error. A scope
/// dropped without `pop_now`, on an early return or an unwind, is popped in
/// `Drop`, so no scope stays on the stack to capture a later caller's error.
///
/// Scope regions do not nest: a second `push` on the same device from inside a
/// region waits on the stripe the first one holds.
pub(crate) struct ValidationScope<'d> {
    device: &'d wgpu::Device,
    popped: bool,
    _region: MutexGuard<'static, ()>,
}

impl<'d> ValidationScope<'d> {
    /// Take `device`'s stripe and push a validation error scope.
    pub(crate) fn push(device: &'d wgpu::Device) -> Self {
        // A panic inside a region still pops its scope in `Drop` before the
        // guard releases, so a poisoned stripe guards a balanced stack.
        let region = stripe(device)
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        Self {
            device,
            popped: false,
            _region: region,
        }
    }

    /// Pop the scope and return the validation error it captured, if any.
    ///
    /// # Errors
    ///
    /// Returns a static reason when the nonblocking device poll before the pop
    /// fails or the pop does not resolve without a host wait. The scope is
    /// popped in both cases.
    pub(crate) fn pop_now(mut self) -> Result<Option<wgpu::Error>, &'static str> {
        self.popped = true;
        pop(self.device)
    }
}

impl Drop for ValidationScope<'_> {
    fn drop(&mut self) {
        if !self.popped {
            drop(pop(self.device));
        }
    }
}

fn stripe(device: &wgpu::Device) -> &'static Mutex<()> {
    let hash = BuildHasherDefault::<rustc_hash::FxHasher>::default().hash_one(device);
    &SCOPE_STRIPES[hash as usize % SCOPE_STRIPES.len()]
}

struct NoopWaker;

impl Wake for NoopWaker {
    fn wake(self: Arc<Self>) {}

    fn wake_by_ref(self: &Arc<Self>) {}
}

/// Pop the top scope. `pop_error_scope` removes the scope when called, so the
/// stack stays balanced whether or not the poll before it succeeded.
fn pop(device: &wgpu::Device) -> Result<Option<wgpu::Error>, &'static str> {
    let polled = device.poll(wgpu::PollType::Poll);
    let mut future = Box::pin(device.pop_error_scope());
    if polled.is_err() {
        return Err("wgpu device poll failed before error-scope pop");
    }
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut context = Context::from_waker(&waker);
    match Future::poll(Pin::as_mut(&mut future), &mut context) {
        Poll::Ready(error) => Ok(error),
        Poll::Pending => Err(
            "wgpu error scope did not resolve after a nonblocking device poll. Fix: inspect the backend event loop; validation must not require a hot-path host wait.",
        ),
    }
}

// Inline: `ValidationScope` and its stripes are crate-private, so an
// integration test cannot name either one.
#[cfg(all(test, feature = "device-tests"))]
mod tests {
    use super::ValidationScope;
    use std::sync::Barrier;

    const INVALID_WGSL: &str = "@compute @workgroup_size(1) fn main() { let x: u32 = 1.5; }";
    const VALID_WGSL: &str = "@compute @workgroup_size(1) fn main() {}";

    fn compile(device: &wgpu::Device, source: &'static str) -> Option<wgpu::Error> {
        let scope = ValidationScope::push(device);
        let _module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vyre validation scope test"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        scope
            .pop_now()
            .expect("Fix: a shader-module scope must resolve without a host wait")
    }

    /// A validation error reaches the caller that caused it, and only that
    /// caller, while other threads open scopes on the same device.
    ///
    /// Without the stripe a valid compile on a second thread pops the scope the
    /// invalid compile pushed, and the invalid compile reports success.
    #[test]
    fn concurrent_scopes_on_one_device_report_to_their_own_caller() {
        const ROUNDS: usize = 64;
        let ((device, _queue), ..) =
            super::super::init_device().expect("Fix: a device-test host must acquire a device");
        let start = Barrier::new(2);
        std::thread::scope(|threads| {
            let invalid = threads.spawn(|| {
                start.wait();
                (0..ROUNDS)
                    .filter(|_| compile(&device, INVALID_WGSL).is_none())
                    .count()
            });
            let valid = threads.spawn(|| {
                start.wait();
                (0..ROUNDS)
                    .filter(|_| compile(&device, VALID_WGSL).is_some())
                    .count()
            });
            let missed = invalid
                .join()
                .expect("Fix: the invalid-compile thread panicked");
            let misattributed = valid
                .join()
                .expect("Fix: the valid-compile thread panicked");
            assert_eq!(
                (missed, misattributed),
                (0, 0),
                "Fix: of {ROUNDS} rounds, {missed} invalid compiles reported no error and {misattributed} valid compiles received another caller's error. Scope regions on one device must not interleave."
            );
        });
    }

    /// A scope dropped without `pop_now`, as on an early return, is popped.
    ///
    /// A scope left on the stack captures the next error raised outside any
    /// scope, and pops in place of a scope pushed after it. With the stripe
    /// held no other caller's scope is on the stack, so an empty stack proves
    /// the drop popped: wgpu panics on a pop with nothing to pop.
    #[test]
    fn a_dropped_scope_is_popped() {
        let ((device, _queue), ..) =
            super::super::init_device().expect("Fix: a device-test host must acquire a device");
        drop(ValidationScope::push(&device));
        let _region = super::stripe(&device)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let leftover = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            drop(device.pop_error_scope());
        }));
        assert!(
            leftover.is_err(),
            "Fix: a scope dropped without pop_now stayed on the device's error-scope stack"
        );
    }
}
