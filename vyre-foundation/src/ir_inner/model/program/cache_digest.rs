//! Normalized compiled-pipeline cache digest, and its per-`Program` memo.
//!
//! # What this digest is for
//!
//! Backend pipeline caches key generated primary text or binary on this digest.
//! It therefore has exactly one correctness obligation:
//! two programs that compile to different backend code MUST get different
//! digests. Over-keying (two programs that compile identically getting
//! different digests) only costs a redundant compile; under-keying serves code
//! generated for a different program, which surfaces as wrong output.
//!
//! # Why these inputs and no others
//!
//! The input set is derived from the single Program-to-emitter boundary rather
//! than guessed. `vyre_lower::lower_for_emit` is the only
//! Program-to-descriptor lowering, and every emitter reads only the resulting
//! `KernelDescriptor`. `vyre-lower/src/lower.rs` reads exactly these program
//! inputs into that descriptor:
//!
//! - `Program::workgroup_size` into `Dispatch`
//! - `Program::entry` into the descriptor body
//! - per `BufferDecl`: `name`, `binding`, `access`, `kind`, `element`, `count`
//!
//! Nothing else on `Program` or `BufferDecl` can reach an emitter, so nothing
//! else belongs in the digest. `entry_op_id` is additionally included: it does
//! not reach the descriptor, but it is cheap and keeps distinct certified
//! operations in distinct cache lanes.
//!
//! `count` participates only where it is a static array length, which is
//! [`BufferDecl::has_static_element_count`]. Runtime-sized storage and uniform
//! buffers keep their count erased so that resizing a buffer does not force a
//! shader recompile, an invariance pinned by the driver-owned disk-cache
//! contracts.
//!
//! That erasure is safe only because the two emission paths differ. The primary
//! text emitter reads `element_count` into generated text only under
//! `MemoryClass::Shared`; every other class remains dynamically sized, so a
//! runtime storage length cannot reach primary text. The primary binary emitter
//! can bake a binding count as an immediate for an asynchronous copy. Its
//! owning driver therefore carries a second full-wire-hash cache-key lane
//! beside this digest rather than relying on this digest alone.
//!
//! # Staleness boundary
//!
//! `Program`'s IR fields are `pub`, so writing one directly through a `&mut
//! Program` leaves this memo stale, exactly as it leaves `hash` and
//! `fingerprint` stale. No production path does that: every mutator that can
//! change these inputs (`entry_mut`, `set_workgroup_size`,
//! `set_parallel_region_size`, `with_entry_op_id`,
//! `with_non_composable_with_self`) routes through `invalidate_caches_for`.
//!
//! # Memoization
//!
//! The digest is a pure function of the program value, so it is memoized on
//! the value itself in a `OnceLock`, exactly as [`Program::fingerprint`] is.
//! A memo that lives on the value is keyed by nothing, so it structurally
//! cannot serve another program's digest, and every sanctioned mutator clears
//! it through `invalidate_caches_for`.
//!
//! # Three keys over `BufferDecl`, and why this one is the narrow one
//!
//! There are three independent keys derived from `BufferDecl`, and their
//! coverage differs ON PURPOSE. `to_wire`/`fingerprint` and
//! `buffer_decl_canonical_key` (program equality) both cover
//! `bytes_extraction`, `linear_type` and `shape_predicate`; this digest covers
//! NONE of the three. That is a decision, not an omission: those fields are
//! declaration-level disciplines that feed validation verdicts, and none of
//! them reaches an emitter, so a backend artifact cache must not vary on them.
//! Do not "fix" this to match the other two keys. A field added to
//! `BufferDecl` belongs here only if `vyre_lower::lower` reads it into the
//! descriptor.

use super::{BufferDecl, MemoryKind, Program};

/// Version label for the normalized `Program` cache digest.
///
/// Single source of truth for both the digest's own domain separator and the
/// label recorded in dispatch evidence, so the label can never describe an
/// algorithm the digest no longer implements.
///
/// `v3` dropped `Program::is_structurally_validated` (validation state is
/// provably not a codegen input) and added `BufferDecl::binding` plus the
/// static-array `count`, which are.
pub const NORMALIZED_PROGRAM_CACHE_DIGEST_VERSION: &str = "vyre-pipeline-cache-norm-v3";

impl Program {
    /// Normalized digest used by backend compiled-pipeline caches.
    ///
    /// Computed at most once per `Program` value: the result is memoized on the
    /// program and cleared by every cache-invalidating mutation.
    ///
    /// # Errors
    ///
    /// Returns when the program contains an IR type or node shape that cannot
    /// be serialized into stable cache identity. Dispatch admission surfaces
    /// the error rather than generating a lossy cache key. Failures are not
    /// memoized, so a caller that repairs the program sees the repair.
    pub fn try_normalized_cache_digest(&self) -> Result<[u8; 32], String> {
        if let Some(digest) = self.normalized_cache_digest.get() {
            return Ok(*digest);
        }
        let digest = self.compute_normalized_cache_digest()?;
        let _ = self.normalized_cache_digest.set(digest);
        Ok(digest)
    }

    /// Uncached digest computation.
    ///
    /// `pub(super)` rather than private so the memo-soundness test can compare
    /// a memoized read against a fresh recompute; a memo that returns a wrong
    /// value consistently is invisible to any test that only calls the cached
    /// path.
    pub(super) fn compute_normalized_cache_digest(&self) -> Result<[u8; 32], String> {
        super::record_digest_computation();

        thread_local! {
            static SCRATCH: std::cell::RefCell<Vec<u8>> =
                std::cell::RefCell::new(Vec::with_capacity(1024));
        }
        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.clear();
            scratch.extend_from_slice(NORMALIZED_PROGRAM_CACHE_DIGEST_VERSION.as_bytes());
            scratch.extend_from_slice(b"\0wg\0");
            for axis in self.workgroup_size {
                scratch.extend_from_slice(&axis.to_le_bytes());
            }
            scratch.extend_from_slice(b"\0op\0");
            match self.entry_op_id.as_deref() {
                Some(op) => {
                    // Length-prefixed: a raw name plus a NUL terminator lets an
                    // op id containing an interior NUL impersonate a different
                    // id followed by the next field.
                    scratch.extend_from_slice(&op_len_bytes(op.len())?);
                    scratch.extend_from_slice(op.as_bytes());
                }
                None => scratch.extend_from_slice(&[0u8; 4]),
            }
            scratch.extend_from_slice(b"\0bufs\0");
            for buffer in self.buffers.iter() {
                append_buffer_cache_key(&mut scratch, buffer)?;
            }
            scratch.extend_from_slice(b"\0body\0");
            crate::serial::wire::append_node_list_fingerprint(&mut scratch, self.entry()).map_err(
                |message| {
                    format!(
                        "failed to fingerprint pipeline-cache Program body: {message}. Fix: validate and normalize the Program before computing a compiled-pipeline cache key; invalid IR must not enter cache identity."
                    )
                },
            )?;
            Ok(*blake3::hash(&scratch).as_bytes())
        })
    }
}

fn op_len_bytes(len: usize) -> Result<[u8; 4], String> {
    u32::try_from(len)
        .map(u32::to_le_bytes)
        .map_err(|_| {
            format!(
                "pipeline-cache Program entry op id length {len} exceeds u32. Fix: shorten the certified operation id before computing a compiled-pipeline cache key."
            )
        })
}

fn append_buffer_cache_key(scratch: &mut Vec<u8>, buffer: &BufferDecl) -> Result<(), String> {
    // Length-prefixed name. The v2 encoding wrote the raw name followed by a
    // NUL, so a buffer literally named "a\0<tag bytes>" could produce the same
    // byte stream as a buffer named "a" with different tags.
    let name = buffer.name();
    let name_len = u32::try_from(name.len()).map_err(|_| {
        format!(
            "pipeline-cache buffer name length {} exceeds u32. Fix: shorten the buffer name before computing a compiled-pipeline cache key.",
            name.len()
        )
    })?;
    scratch.extend_from_slice(&name_len.to_le_bytes());
    scratch.extend_from_slice(name.as_bytes());

    // Stable tags, never `enum as u8`: both enums are `#[non_exhaustive]`, so a
    // variant inserted mid-list would silently remap discriminants and could
    // alias a persisted entry recorded under the same version label.
    scratch.push(memory_kind_cache_tag(buffer.kind()));
    let access_tag = crate::serial::wire::tags::access_tag::access_tag(&buffer.access).map_err(
        |message| {
            format!(
                "failed to tag pipeline-cache buffer access for `{name}`: {message}. Fix: validate and normalize the Program before computing a compiled-pipeline cache key; invalid IR must not enter cache identity."
            )
        },
    )?;
    scratch.push(access_tag);

    // Read by vyre_lower::lower as the requested host binding slot, then emitted
    // as the corresponding primary-text binding. Absent from v2, which let two
    // programs with different binding layouts share one compiled-pipeline entry.
    scratch.extend_from_slice(&buffer.binding().to_le_bytes());

    crate::serial::wire::append_data_type_fingerprint(scratch, &buffer.element()).map_err(
        |message| {
            format!(
                "failed to fingerprint pipeline-cache buffer data type `{name}`: {message}. Fix: validate and normalize the Program before computing a compiled-pipeline cache key; invalid IR must not enter cache identity."
            )
        },
    )?;

    // Fixed width so the count lane can never shift the following bytes:
    // zero means "no static element count", matching how vyre_lower maps a
    // zero count to `element_count: None`.
    let static_count = if buffer.has_static_element_count() {
        buffer.count()
    } else {
        0
    };
    scratch.extend_from_slice(&static_count.to_le_bytes());
    Ok(())
}

/// Stable cache-identity tag for a memory tier.
///
/// Single owner for both this digest and `buffer_decl_canonical_key`. The match
/// is exhaustive on purpose: a new `MemoryKind` must fail to compile here so the
/// author decides its cache-identity tag instead of inheriting a wildcard.
pub(super) const fn memory_kind_cache_tag(kind: MemoryKind) -> u8 {
    match kind {
        MemoryKind::Global => 0,
        MemoryKind::Shared => 1,
        MemoryKind::Uniform => 2,
        MemoryKind::Local => 3,
        MemoryKind::Readonly => 4,
        MemoryKind::Persistent => 5,
        MemoryKind::Push => 6,
    }
}
