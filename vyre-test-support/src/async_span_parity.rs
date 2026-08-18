//! Programs, the offset and length matrix, and the reference arm for the async
//! transfer byte-span parity gates.
//!
//! # Why this has one owner
//!
//! `Node::AsyncLoad` and `Node::AsyncStore` carry their offset and length in
//! BYTES: the reference evaluator copies `size` bytes starting at `offset`,
//! reads zeros past the end of the source, and clips at the end of the
//! destination. Every backend addresses whole words, so every backend has to
//! assemble a span whose ends fall inside a word, and every backend owes the
//! same gate: run the span on real silicon and compare byte-for-byte with the
//! reference. The naga emitter turned the byte offset into a word index by
//! dividing it by four, and the PTX emitter did the same, so both copied from
//! the wrong byte for three quarters of the offsets. A matrix per backend means
//! one backend's suite can be trimmed to the aligned cases that always worked
//! while the other keeps the coverage.
//!
//! # What this crate must NOT own
//!
//! The dispatch stays in each backend's suite. The naga word assembly and the
//! PTX word assembly are different implementations of one contract: each needs
//! its own live proof, and neither may be compared against the other in place of
//! it.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};
use vyre_reference::value::Value;

/// Words in the transfer source.
pub const SOURCE_WORDS: u32 = 8;
/// Words in the transfer destination.
pub const DESTINATION_WORDS: u32 = 6;
/// Workgroup widths under test.
///
/// The emitted copy loop carries no invocation guard, so a dispatched program
/// runs the whole copy in every invocation of the workgroup: that is the shape
/// production dispatches, and a span whose ends fall inside a word has every
/// invocation merging the same destination word under the same mask. The merge
/// is idempotent by construction, since the bytes it preserves are the bytes it
/// read, so every ordering must land on the same word. A single invocation
/// proves the arithmetic alone; the wider width proves that claim on silicon,
/// and a divergence between the two is a real defect rather than a fixture
/// artifact.
pub const WORKGROUPS: [[u32; 3]; 2] = [[1, 1, 1], [16, 1, 1]];

/// Byte offsets under test: every residue modulo four, a span that starts past
/// the middle of the source, and one that starts near its end.
pub const OFFSETS: [u32; 7] = [0, 1, 3, 5, 6, 8, 13];

/// Byte lengths under test: every residue modulo four, one whole-word length,
/// one that runs off the end of the source, and one that clips at the end of the
/// destination.
pub const SIZES: [u32; 7] = [1, 4, 6, 7, 16, 20, 28];

/// Which end of the transfer the offset applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Offset into the source; the destination fills from its own start.
    Load,
    /// Offset into the destination; the source is read from its own start.
    Store,
}

impl Direction {
    /// Node name for a failure message.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Load => "AsyncLoad",
            Self::Store => "AsyncStore",
        }
    }

    fn node(self, offset: Expr, size: u32) -> Node {
        let source = Ident::from("src");
        let destination = Ident::from("dst");
        let offset = Box::new(offset);
        let size = Box::new(Expr::u32(size));
        let tag = Ident::from("span");
        match self {
            Self::Load => Node::AsyncLoad {
                source,
                destination,
                offset,
                size,
                tag,
            },
            Self::Store => Node::AsyncStore {
                source,
                destination,
                offset,
                size,
                tag,
            },
        }
    }
}

/// How the offset reaches the node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffsetForm {
    /// A literal an emitter can inspect, so an aligned offset may take a cheaper
    /// word-for-word copy.
    Literal,
    /// A value loaded from a buffer, so alignment is only known at run time.
    Loaded,
}

impl OffsetForm {
    /// Form name for a failure message.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Literal => "literal offset",
            Self::Loaded => "loaded offset",
        }
    }
}

/// One matrix entry: a direction, an offset form, and the span it names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanCase {
    /// Which end of the transfer the offset applies to.
    pub direction: Direction,
    /// How the offset reaches the node.
    pub form: OffsetForm,
    /// Byte offset of the span.
    pub offset: u32,
    /// Byte length of the span.
    pub size: u32,
    /// Workgroup the transfer is dispatched with.
    pub workgroup: [u32; 3],
}

impl SpanCase {
    /// The program that performs this transfer.
    #[must_use]
    pub fn program(&self) -> Program {
        let mut buffers = vec![
            BufferDecl::storage("src", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(SOURCE_WORDS),
            BufferDecl::storage("dst", 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(DESTINATION_WORDS),
        ];
        let offset = match self.form {
            OffsetForm::Literal => Expr::u32(self.offset),
            OffsetForm::Loaded => {
                buffers.push(
                    BufferDecl::storage("offsets", 2, BufferAccess::ReadOnly, DataType::U32)
                        .with_count(1),
                );
                Expr::load("offsets", Expr::u32(0))
            }
        };
        let body = vec![wrap_anonymous_region(
            "vyre-test-support::async_span_parity",
            vec![
                self.direction.node(offset, self.size),
                Node::AsyncWait {
                    tag: Ident::from("span"),
                },
            ],
        )];
        Program::wrapped(buffers, self.workgroup, body)
    }

    /// The dispatch inputs, in declaration order.
    #[must_use]
    pub fn inputs(&self) -> Vec<Vec<u8>> {
        let mut inputs = vec![source_bytes(), destination_bytes()];
        if self.form == OffsetForm::Loaded {
            inputs.push(self.offset.to_le_bytes().to_vec());
        }
        inputs
    }

    /// What the reference evaluator produces for this transfer.
    ///
    /// # Panics
    ///
    /// Panics when the reference evaluator rejects the program, which is a
    /// defect in the fixture rather than in the backend under test.
    #[must_use]
    pub fn reference_outputs(&self) -> Vec<Vec<u8>> {
        let program = self.program();
        let values: Vec<Value> = self.inputs().into_iter().map(Value::from).collect();
        vyre_reference::reference_eval(&program, &values)
            .expect("Fix: the reference evaluator must run the async transfer fixture")
            .into_iter()
            .map(|value| value.to_bytes())
            .collect()
    }

    /// Assert `actual` is byte-for-byte what the reference produces.
    ///
    /// `backend` names where the bytes came from and `lowering` what the backend
    /// lowered the transfer to, so a failure says which implementation of the
    /// contract broke.
    ///
    /// # Panics
    ///
    /// Panics with the first divergent byte when the two disagree.
    pub fn assert_matches_reference(&self, backend: &str, lowering: &str, actual: &[Vec<u8>]) {
        let expected = self.reference_outputs();
        assert_eq!(
            actual.len(),
            expected.len(),
            "{backend} {} {} offset={} size={} workgroup={:?}: {lowering} returned {} buffers, the reference returned {}",
            self.direction.label(),
            self.form.label(),
            self.offset,
            self.size,
            self.workgroup,
            actual.len(),
            expected.len(),
        );
        for (index, (measured, reference)) in actual.iter().zip(&expected).enumerate() {
            if measured == reference {
                continue;
            }
            let first = measured
                .iter()
                .zip(reference)
                .position(|(left, right)| left != right)
                .unwrap_or_else(|| measured.len().min(reference.len()));
            panic!(
                "{backend} {} {} offset={} size={} workgroup={:?}: {lowering} diverges from the reference in \
                 buffer #{index} at byte {first}\n  {backend}: {measured:02x?}\n  reference: \
                 {reference:02x?}",
                self.direction.label(),
                self.form.label(),
                self.offset,
                self.size,
                self.workgroup,
            );
        }
    }
}

/// Source bytes: a distinct value per byte, so a copy that starts one byte off
/// produces different bytes rather than the same ones.
#[must_use]
pub fn source_bytes() -> Vec<u8> {
    (0..SOURCE_WORDS * 4)
        .map(|index| (index + 1) as u8)
        .collect()
}

/// Destination bytes: a pattern no copy can produce, so a byte the transfer must
/// preserve is distinguishable from one it must write.
#[must_use]
pub fn destination_bytes() -> Vec<u8> {
    vec![0xAA; (DESTINATION_WORDS * 4) as usize]
}

/// Every case in the matrix: both directions, both offset forms, every span.
#[must_use]
pub fn cases() -> Vec<SpanCase> {
    let mut cases = Vec::with_capacity(4 * WORKGROUPS.len() * OFFSETS.len() * SIZES.len());
    for workgroup in WORKGROUPS {
        for direction in [Direction::Load, Direction::Store] {
            for form in [OffsetForm::Literal, OffsetForm::Loaded] {
                for offset in OFFSETS {
                    for size in SIZES {
                        cases.push(SpanCase {
                            direction,
                            form,
                            offset,
                            size,
                            workgroup,
                        });
                    }
                }
            }
        }
    }
    cases
}

/// The matrix reaches every alignment case a byte span can present.
///
/// Asserted from the matrix rather than stated in a comment: an offset table
/// trimmed back to multiples of four would leave the emitters' word-assembly
/// path unproven while the suite still passed.
///
/// # Panics
///
/// Panics naming the alignment case the matrix no longer covers.
pub fn assert_matrix_covers_every_alignment() {
    for residue in 0..4 {
        assert!(
            OFFSETS.iter().any(|offset| offset % 4 == residue),
            "Fix: the offset matrix must exercise byte offset residue {residue} modulo four"
        );
        assert!(
            SIZES.iter().any(|size| size % 4 == residue),
            "Fix: the length matrix must exercise byte length residue {residue} modulo four"
        );
    }
    assert!(
        OFFSETS
            .iter()
            .any(|offset| SIZES.iter().any(|size| offset + size > SOURCE_WORDS * 4)),
        "Fix: the matrix must exercise a span that runs off the end of the source"
    );
    assert!(
        SIZES.iter().any(|size| *size > DESTINATION_WORDS * 4),
        "Fix: the matrix must exercise a span that clips at the end of the destination"
    );
    assert!(
        cases().iter().any(|case| case.form == OffsetForm::Loaded)
            && cases().iter().any(|case| case.form == OffsetForm::Literal),
        "Fix: the matrix must exercise both the literal and the loaded offset form"
    );
    assert!(
        cases()
            .iter()
            .any(|case| case.direction == Direction::Store)
            && cases().iter().any(|case| case.direction == Direction::Load),
        "Fix: the matrix must exercise both transfer directions"
    );
    assert!(
        cases().iter().any(|case| case.workgroup[0] == 1)
            && cases().iter().any(|case| case.workgroup[0] > 1),
        "Fix: the matrix must dispatch the transfer both with one invocation and with the wider \
         workgroup a program really dispatches, so a redundant copy that disagrees with itself is \
         a failure rather than an untested shape"
    );
}
