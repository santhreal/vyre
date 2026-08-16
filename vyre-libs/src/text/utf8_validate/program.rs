//! Program entry points and buffer layout for UTF-8 validation.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::sequence_rules::{
    byte_expr, continuation_validation_body, in_range, lead2_validation_body,
    lead3_validation_body, lead4_validation_body,
};
use super::{OP_ID, UTF8_ASCII, UTF8_INVALID, UTF8_VALIDATE_WORKGROUP_SIZE};

/// Build a Program that validates and classifies each `source[i]`
/// byte into one of the `UTF8_*` codes above and writes the result
/// into `classes[i]`.
///
/// This compatibility entry point expects one `DataType::U32` element per
/// source byte and reads the low byte of each word. Use [`utf8_validate_u8`]
/// when the source is packed as one byte per element.
#[must_use]
pub fn utf8_validate(source: &str, classes: &str, n: u32) -> Program {
    utf8_validate_with_source_type(source, classes, n, DataType::U32)
}

/// Build a UTF-8 validation Program over a packed `DataType::U8` source.
///
/// It emits the same per-byte class stream as [`utf8_validate`] while cutting
/// source input bandwidth from four bytes per logical byte to one.
#[must_use]
pub fn utf8_validate_u8(source: &str, classes: &str, n: u32) -> Program {
    utf8_validate_with_source_type(source, classes, n, DataType::U8)
}

fn utf8_validate_with_source_type(
    source: &str,
    classes: &str,
    n: u32,
    source_type: DataType,
) -> Program {
    let idx = Expr::InvocationId { axis: 0 };
    let body = vec![wrap_anonymous_region(
        OP_ID,
        vec![
            Node::let_bind("idx", idx.clone()),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::u32(n)),
                vec![
                    Node::let_bind("byte", byte_expr(source, Expr::var("idx"))),
                    Node::let_bind("class", Expr::u32(UTF8_INVALID)),
                    Node::if_then(
                        Expr::lt(Expr::var("byte"), Expr::u32(0x80)),
                        vec![Node::assign("class", Expr::u32(UTF8_ASCII))],
                    ),
                    Node::if_then(
                        in_range(Expr::var("byte"), 0x80, 0xBF),
                        continuation_validation_body(source),
                    ),
                    Node::if_then(
                        in_range(Expr::var("byte"), 0xC2, 0xDF),
                        lead2_validation_body(source, n),
                    ),
                    Node::if_then(
                        in_range(Expr::var("byte"), 0xE0, 0xEF),
                        lead3_validation_body(source, n),
                    ),
                    Node::if_then(
                        in_range(Expr::var("byte"), 0xF0, 0xF4),
                        lead4_validation_body(source, n),
                    ),
                    Node::store(classes, Expr::var("idx"), Expr::var("class")),
                ],
            ),
        ],
    )];

    let source_decl = if n == 0 {
        BufferDecl::storage(source, 0, BufferAccess::ReadOnly, source_type)
    } else {
        BufferDecl::storage(source, 0, BufferAccess::ReadOnly, source_type).with_count(n)
    };
    Program::wrapped(
        vec![
            source_decl,
            BufferDecl::output(classes, 1, DataType::U32)
                .with_count(n.max(1))
                .with_output_byte_range(0..(n as usize).saturating_mul(4)),
        ],
        UTF8_VALIDATE_WORKGROUP_SIZE,
        body,
    )
}
