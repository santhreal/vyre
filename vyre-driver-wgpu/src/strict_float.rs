//! Whether this adapter's shader compilation preserves IEEE-754 rounding.
//!
//! `FloatLoweringMode::StrictIeee` is a bit-identity request: every f32
//! operation rounds where IEEE-754 says it does, so a device answer can be
//! compared with the reference oracle word for word. The emitter answers it by
//! expanding every approximable operation and publishing each f32 product
//! through a `u32` reinterpretation emitted as its own statement, which denies
//! the target the multiply-add pair it would otherwise fuse.
//!
//! That is a statement about the module. It is not a statement about the
//! machine code, because the module is compiled a second time by the platform
//! and that second compiler decides whether the barrier survives. On the Metal
//! path `wgpu-hal` builds every library with a default `MTLCompileOptions`,
//! which leaves fast math enabled: it sets only `setPreserveInvariance`, which
//! pins position invariance across passes and says nothing about rounding. Fast
//! math permits contraction, reassociation, and approximate transcendentals, so
//! the `u32` round trip folds and the expansions drift. Measured on a Metal
//! adapter, the strict witness returned the contracted answer and
//! `exp(8.765743)` came back as `4096.001` against a true `6410.825`, which is
//! the range reduction losing its fractional term.
//!
//! Nothing on the WebGPU surface turns that off. So the question is answered by
//! measurement rather than by a vendor list or a backend name. One dispatch of a
//! witness whose two possible answers differ in exactly the bit a fused
//! multiply-add retains decides it, once per adapter per process. An adapter
//! that fails the witness is refused a strict dispatch by name, which is the
//! only alternative to answering a bit-identity request with arithmetic that is
//! not bit-identical.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::runtime::device::{enumerate_adapters, AdapterIdentity};
use crate::WgpuBackend;

/// `1 + 2^-12`, exact in f32.
///
/// Chosen so `w * w - 1` separates the two roundings: the exact product is
/// `1 + 2^-11 + 2^-24`, a rounded multiply discards the `2^-24`, and the
/// subtraction that follows cancels the leading one and leaves whichever of the
/// two the target computed.
const WITNESS: f32 = f32::from_bits(0x3F80_0800);

/// `2^-11`, the answer when the multiply rounds before the add.
const SEPARATELY_ROUNDED: u32 = 0x3A00_0000;

thread_local! {
    /// Set while this thread is measuring, so the admission check the probe's
    /// own dispatch runs does not ask the question it is answering.
    static MEASURING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// One verdict per adapter, shared by every backend instance on it.
///
/// The fact belongs to the adapter and its shader compiler, not to a backend
/// value, and two backends acquired on one adapter must not measure twice.
/// [`AdapterIdentity`] is the single definition of which adapter a fact belongs
/// to, and it already includes the driver strings a compiler update changes.
///
/// The bound is the adapter count. A key is built from the `AdapterInfo` of an
/// adapter a backend was acquired on, every such adapter comes from the same
/// enumeration [`enumerate_adapters`] reports, and an `AdapterInfo` is fixed
/// for the life of the process, so the entry count cannot exceed the number of
/// adapters this process can see. The map is constructed once with exactly that
/// capacity, and the enumeration runs on the first strict dispatch, ahead of
/// the witness dispatch that decides the same adapter's verdict.
static VERDICTS: LazyLock<Mutex<HashMap<AdapterIdentity, Verdict>>> =
    LazyLock::new(|| Mutex::new(HashMap::with_capacity(enumerate_adapters().len())));

/// `w * w + -1.0` over one lane, read back from a read-write output slot.
fn witness_program() -> Program {
    let index = Expr::gid_x();
    Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::F32).with_count(1),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::F32).with_count(1),
        ],
        [64, 1, 1],
        vec![Node::if_then(
            Expr::lt(index.clone(), Expr::u32(1)),
            vec![Node::store(
                "out",
                index.clone(),
                Expr::add(
                    Expr::mul(Expr::load("in", index.clone()), Expr::load("in", index)),
                    Expr::f32(-1.0),
                ),
            )],
        )],
    )
}

/// True when this thread is inside the probe's own dispatch.
pub(crate) fn measuring() -> bool {
    MEASURING.with(std::cell::Cell::get)
}

/// Sets the probe flag for as long as it is held.
///
/// Clearing it on the way out rather than after the dispatch: a panic inside
/// the probe would otherwise leave this thread flagged forever, and every
/// strict dispatch it made afterwards would be admitted without a measurement.
struct Measuring;

impl Measuring {
    fn enter() -> Self {
        MEASURING.with(|flag| flag.set(true));
        Self
    }
}

impl Drop for Measuring {
    fn drop(&mut self) {
        MEASURING.with(|flag| flag.set(false));
    }
}

/// What the witness said about this adapter.
///
/// A dispatch that never ran and a dispatch that came back fused both refuse the
/// mode, and they are not the same fact. Recording one boolean for both made the
/// refusal state that a witness returned the fused answer on an adapter where no
/// witness had run, which is a gate certifying what it never checked.
#[derive(Clone, Debug)]
enum Verdict {
    /// The witness rounded the multiply before the add.
    Separate,
    /// The witness came back with the fused answer.
    Fused,
    /// The witness could not be dispatched, and why.
    Unmeasured(String),
}

/// Run the witness under the strict mode and report what it rounded.
fn measure(backend: &WgpuBackend) -> Verdict {
    let program = witness_program();
    let witness = WITNESS.to_le_bytes();
    let readback = [0u8; 4];
    let inputs: [&[u8]; 2] = [&witness, &readback];
    let mut config = vyre_driver::DispatchConfig::default();
    config.float_lowering = vyre_foundation::fp_parity::FloatLoweringMode::StrictIeee;

    let _probe = Measuring::enter();
    let outcome = backend
        .dispatch_borrowed_async(&program, &inputs, &config)
        .and_then(|pending| pending.await_owned());
    drop(_probe);

    let outputs = match outcome {
        Ok(outputs) => outputs,
        Err(error) => return Verdict::Unmeasured(error.to_string()),
    };
    let Some(word) = outputs
        .first()
        .and_then(|bytes| <[u8; 4]>::try_from(bytes.as_slice()).ok())
    else {
        return Verdict::Unmeasured(
            "the witness returned no four-byte f32 output slot".to_string(),
        );
    };
    if f32::from_le_bytes(word).to_bits() == SEPARATELY_ROUNDED {
        Verdict::Separate
    } else {
        Verdict::Fused
    }
}

/// This adapter's verdict, measured once and shared by every backend on it.
///
/// Measured outside the lock: the probe dispatches, and holding a process wide
/// lock across a GPU round trip would serialize every adapter's first strict
/// dispatch behind one device. Two threads that race both measure the same
/// adapter and the first recorded answer is returned to both, so the decision an
/// admission check reads never changes between two calls.
fn verdict(backend: &WgpuBackend) -> Verdict {
    let key = AdapterIdentity::from_info(&backend.adapter_info);
    {
        let mut verdicts = match VERDICTS.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                guard.clear();
                guard
            }
        };
        if let Some(known) = verdicts.get(&key) {
            return known.clone();
        }
    }
    let measured = measure(backend);
    let mut verdicts = match VERDICTS.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            let mut guard = poisoned.into_inner();
            guard.clear();
            guard
        }
    };
    verdicts.entry(key).or_insert(measured).clone()
}

/// Whether a strict dispatch on this adapter returns IEEE-rounded arithmetic.
///
/// A thread already inside the probe answers `true`, so the probe's own dispatch
/// is admitted rather than asking the question it is answering.
pub(crate) fn honored(backend: &WgpuBackend) -> bool {
    measuring() || matches!(verdict(backend), Verdict::Separate)
}

/// The refusal a strict dispatch receives on an adapter that failed the witness.
///
/// It names the mode, the backend and which of the two refusals this is, because
/// those are what a caller acts on: an adapter that rounds fused is answered by
/// choosing another adapter or the contracted mode, and one the witness could
/// not reach is a device fault to repair.
pub(crate) fn refusal(backend: &WgpuBackend) -> vyre_driver::BackendError {
    let strict = vyre_foundation::fp_parity::FloatLoweringMode::StrictIeee.cache_label();
    let name = match verdict(backend) {
        Verdict::Separate => format!(
            "{strict} f32 lowering: refused after this adapter passed the rounding witness, which \
             is a defect in the admission check rather than in the adapter"
        ),
        Verdict::Fused => format!(
            "{strict} f32 lowering: this adapter's shader compiler does not preserve \
             per-operation rounding. A multiply-add witness returned the fused answer. Fix: \
             dispatch the strict mode on an adapter that preserves it, or state the contracted \
             mode, whose accuracy contract is the ULP budget in `vyre_foundation::fp_parity` \
             rather than bit identity"
        ),
        Verdict::Unmeasured(why) => format!(
            "{strict} f32 lowering: this adapter is uncertified because the rounding witness \
             could not be dispatched ({why}). A bit-identity request is refused rather than \
             answered by an unmeasured device. Fix: repair the device fault above, then retry"
        ),
    };
    vyre_driver::BackendError::UnsupportedFeature {
        name,
        backend: <WgpuBackend as vyre_driver::VyreBackend>::id(backend).to_string(),
    }
}
