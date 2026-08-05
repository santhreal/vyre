//! C source ingestion and backend-neutral typed-IR lowering.

use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;
use tree_sitter::{Node as SyntaxNode, Parser, Point, Tree};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Maximum accepted translation-unit size.
///
/// The bound keeps source spans representable as `u32` and prevents hostile
/// inputs from forcing unbounded parser allocations.
pub const MAX_SOURCE_BYTES: usize = 16 * 1024 * 1024;

/// A successfully ingested C translation unit.
///
/// The syntax tree and its source are owned together. No execution substrate is
/// captured, so the value can be cached or transferred before lowering.
#[derive(Clone)]
pub struct ParsedTranslationUnit {
    source: Arc<str>,
    tree: Tree,
}

impl std::fmt::Debug for ParsedTranslationUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParsedTranslationUnit")
            .field("source_len", &self.source.len())
            .field("root_kind", &self.tree.root_node().kind())
            .finish()
    }
}

impl ParsedTranslationUnit {
    /// Return the exact source accepted by the parser.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Return the concrete syntax tree supplied by the parser substrate.
    #[must_use]
    pub fn syntax_tree(&self) -> &Tree {
        &self.tree
    }
}

/// Deterministic C ingestion or lowering failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CFrontendError {
    /// The translation unit exceeds [`MAX_SOURCE_BYTES`].
    #[error("C frontend source is {actual} bytes, exceeding the {max}-byte limit. Fix: split the translation unit or reduce generated source size.")]
    SourceTooLarge {
        /// Observed byte length.
        actual: usize,
        /// Accepted maximum byte length.
        max: usize,
    },
    /// C source must not contain embedded NUL bytes.
    #[error("C frontend rejected byte 0x00 at byte {offset}. Fix: remove embedded NUL bytes from C source.")]
    EmbeddedNul {
        /// Zero-based byte offset.
        offset: usize,
    },
    /// Byte input was not UTF-8.
    #[error("C frontend source is not UTF-8 at byte {offset}. Fix: provide UTF-8 encoded C source.")]
    InvalidUtf8 {
        /// Zero-based byte offset of the first invalid sequence.
        offset: usize,
    },
    /// The parser substrate could not be initialized.
    #[error("C frontend parser initialization failed. Fix: use the C grammar version shipped with vyre-frontend-c.")]
    ParserInitialization,
    /// The parser rejected malformed syntax.
    #[error("C frontend parse failed at byte {byte} (line {line}, column {column}) near `{fragment}`. Fix: provide a complete C translation unit.")]
    Syntax {
        /// Zero-based source byte offset.
        byte: usize,
        /// One-based source line.
        line: usize,
        /// One-based byte column.
        column: usize,
        /// Bounded source fragment at the failure.
        fragment: String,
    },
    /// No supported `kernel` entry function was present.
    #[error("C frontend lowering found no `kernel` function. Fix: define exactly one `kernel` entry function.")]
    MissingEntrypoint,
    /// More than one `kernel` entry function was present.
    #[error("C frontend lowering found multiple `kernel` functions. Fix: define exactly one `kernel` entry function.")]
    DuplicateEntrypoint,
    /// Syntactically valid C used a construct outside the typed-IR contract.
    #[error("C frontend cannot lower {construct} at byte {byte}. Fix: use the supported scalar kernel subset or lower the construct before this frontend.")]
    Unsupported {
        /// Zero-based source byte offset.
        byte: usize,
        /// Stable construct description.
        construct: String,
    },
}

/// Ingest UTF-8 C source into an owned syntax tree.
pub fn parse_source(source: &str) -> Result<ParsedTranslationUnit, CFrontendError> {
    parse_validated_source(source)
}

/// Validate UTF-8 C bytes and ingest them into an owned syntax tree.
pub fn parse_source_bytes(bytes: &[u8]) -> Result<ParsedTranslationUnit, CFrontendError> {
    if bytes.len() > MAX_SOURCE_BYTES {
        return Err(CFrontendError::SourceTooLarge {
            actual: bytes.len(),
            max: MAX_SOURCE_BYTES,
        });
    }
    if let Some(offset) = bytes.iter().position(|byte| *byte == 0) {
        return Err(CFrontendError::EmbeddedNul { offset });
    }
    let source = std::str::from_utf8(bytes).map_err(|error| CFrontendError::InvalidUtf8 {
        offset: error.valid_up_to(),
    })?;
    parse_validated_source(source)
}

fn parse_validated_source(source: &str) -> Result<ParsedTranslationUnit, CFrontendError> {
    if source.len() > MAX_SOURCE_BYTES {
        return Err(CFrontendError::SourceTooLarge {
            actual: source.len(),
            max: MAX_SOURCE_BYTES,
        });
    }
    if let Some(offset) = source.as_bytes().iter().position(|byte| *byte == 0) {
        return Err(CFrontendError::EmbeddedNul { offset });
    }

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_c::LANGUAGE.into())
        .map_err(|_| CFrontendError::ParserInitialization)?;
    let tree = parser
        .parse(source, None)
        .ok_or(CFrontendError::ParserInitialization)?;
    if let Some(error) = first_syntax_error(tree.root_node()) {
        return Err(syntax_error(source, error));
    }
    Ok(ParsedTranslationUnit {
        source: Arc::from(source),
        tree,
    })
}

/// Parse C source and lower its `kernel` entry to typed Vyre IR.
///
/// This is a source-to-IR convenience only; it never selects or invokes an
/// execution backend.
pub fn lower_source(source: &str) -> Result<Program, CFrontendError> {
    let unit = parse_source(source)?;
    lower_translation_unit(&unit)
}

/// Lower a parsed C translation unit to a backend-neutral typed program.
///
/// The supported executable subset is deliberately explicit:
///
/// - exactly one function named `kernel`;
/// - either a scalar `int`/`unsigned int` return with no parameters, or a
///   `void` function whose parameters are scalar pointers;
/// - integer literals, buffer subscripts, parentheses, and `+`, `-`, `*`,
///   `&`, `|`, `^` expressions;
/// - a scalar return or direct assignments to writable pointer parameters.
pub fn lower_translation_unit(unit: &ParsedTranslationUnit) -> Result<Program, CFrontendError> {
    let source = unit.source.as_bytes();
    let root = unit.tree.root_node();
    let mut cursor = root.walk();
    let mut kernels = root
        .named_children(&mut cursor)
        .filter(|node| node.kind() == "function_definition")
        .filter(|node| function_name(*node, source) == Some("kernel"));
    let kernel = kernels.next().ok_or(CFrontendError::MissingEntrypoint)?;
    if kernels.next().is_some() {
        return Err(CFrontendError::DuplicateEntrypoint);
    }
    lower_kernel(kernel, source)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScalarType {
    U32,
    I32,
}

impl ScalarType {
    fn data_type(self) -> DataType {
        match self {
            Self::U32 => DataType::U32,
            Self::I32 => DataType::I32,
        }
    }

    fn literal(self, value: u32, node: SyntaxNode<'_>) -> Result<Expr, CFrontendError> {
        match self {
            Self::U32 => Ok(Expr::u32(value)),
            Self::I32 => i32::try_from(value).map(Expr::i32).map_err(|_| unsupported(
                node,
                "an integer literal outside the signed 32-bit range",
            )),
        }
    }
}

#[derive(Clone, Debug)]
struct BufferParam {
    name: String,
    access: BufferAccess,
    scalar: ScalarType,
}

fn lower_kernel(kernel: SyntaxNode<'_>, source: &[u8]) -> Result<Program, CFrontendError> {
    let declarator = required_field(kernel, "declarator", "a kernel declarator")?;
    let function_declarator = find_kind(declarator, "function_declarator")
        .ok_or_else(|| unsupported(declarator, "a non-function `kernel` declarator"))?;
    let parameters = function_declarator
        .child_by_field_name("parameters")
        .ok_or_else(|| unsupported(function_declarator, "a kernel without a parameter list"))?;
    let return_type_node = required_field(kernel, "type", "a kernel return type")?;
    let return_text = node_text(return_type_node, source);
    let body = required_field(kernel, "body", "a kernel body")?;

    let params = lower_parameters(parameters, source)?;
    if normalized_type(return_text) == "void" {
        lower_void_kernel(body, source, params)
    } else {
        let scalar = parse_scalar_type(return_type_node, return_text)?;
        if !params.is_empty() {
            return Err(unsupported(
                parameters,
                "parameters on a scalar-returning kernel",
            ));
        }
        lower_scalar_kernel(body, source, scalar)
    }
}

fn lower_parameters(
    parameters: SyntaxNode<'_>,
    source: &[u8],
) -> Result<Vec<BufferParam>, CFrontendError> {
    let mut result = Vec::new();
    let mut cursor = parameters.walk();
    for parameter in parameters.named_children(&mut cursor) {
        if parameter.kind() != "parameter_declaration" {
            return Err(unsupported(parameter, "a variadic or non-parameter declaration"));
        }
        let type_node = required_field(parameter, "type", "a parameter type")?;
        let type_text = node_text(type_node, source);
        if normalized_type(type_text) == "void" && parameter.child_by_field_name("declarator").is_none() {
            continue;
        }
        let declarator = required_field(parameter, "declarator", "an unnamed kernel parameter")?;
        if find_kind(declarator, "pointer_declarator").is_none() {
            return Err(unsupported(parameter, "a non-pointer kernel parameter"));
        }
        let identifier = find_kind(declarator, "identifier")
            .ok_or_else(|| unsupported(declarator, "an unnamed pointer parameter"))?;
        let name = node_text(identifier, source).to_owned();
        if result.iter().any(|existing: &BufferParam| existing.name == name) {
            return Err(unsupported(identifier, "a duplicate kernel parameter name"));
        }
        let scalar = parse_scalar_type(type_node, type_text)?;
        let declaration_text = node_text(parameter, source);
        let access = if declaration_text
            .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
            .any(|word| word == "const")
        {
            BufferAccess::ReadOnly
        } else {
            BufferAccess::ReadWrite
        };
        result.push(BufferParam {
            name,
            access,
            scalar,
        });
    }
    Ok(result)
}

fn lower_scalar_kernel(
    body: SyntaxNode<'_>,
    source: &[u8],
    scalar: ScalarType,
) -> Result<Program, CFrontendError> {
    let statements = named_children(body);
    if statements.len() != 1 || statements[0].kind() != "return_statement" {
        return Err(unsupported(
            body,
            "a scalar kernel body other than one return statement",
        ));
    }
    let return_node = statements[0];
    let expression = return_node
        .named_child(0)
        .ok_or_else(|| unsupported(return_node, "a return statement without a value"))?;
    let buffers = vec![
        BufferDecl::storage("out", 0, BufferAccess::ReadWrite, scalar.data_type()).with_count(1),
    ];
    let value = lower_expression(expression, source, scalar, &HashMap::new())?;
    Ok(Program::wrapped(
        buffers,
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), value)],
    ))
}

fn lower_void_kernel(
    body: SyntaxNode<'_>,
    source: &[u8],
    params: Vec<BufferParam>,
) -> Result<Program, CFrontendError> {
    if params.is_empty() {
        return Err(unsupported(body, "a void kernel without buffer parameters"));
    }
    let mut bindings = HashMap::with_capacity(params.len());
    let mut buffers = Vec::with_capacity(params.len());
    for (binding, parameter) in params.iter().enumerate() {
        bindings.insert(parameter.name.as_str(), parameter);
        buffers.push(
            BufferDecl::storage(
                &parameter.name,
                binding as u32,
                parameter.access.clone(),
                parameter.scalar.data_type(),
            )
            .with_count(1),
        );
    }

    let mut entry = Vec::new();
    for statement in named_children(body) {
        if statement.kind() != "expression_statement" {
            return Err(unsupported(statement, "a non-assignment statement in a void kernel"));
        }
        let assignment = statement
            .named_child(0)
            .ok_or_else(|| unsupported(statement, "an empty expression statement"))?;
        if assignment.kind() != "assignment_expression"
            || assignment_operator(assignment, source) != "="
        {
            return Err(unsupported(assignment, "an expression other than direct assignment"));
        }
        let left = required_field(assignment, "left", "an assignment without a left operand")?;
        let right = required_field(assignment, "right", "an assignment without a right operand")?;
        let (buffer_name, index) = lower_subscript(left, source, ScalarType::U32, &bindings)?;
        let parameter = bindings
            .get(buffer_name.as_str())
            .ok_or_else(|| unsupported(left, "assignment to an unknown buffer"))?;
        if parameter.access == BufferAccess::ReadOnly {
            return Err(unsupported(left, "assignment to a const buffer parameter"));
        }
        let value = lower_expression(right, source, parameter.scalar, &bindings)?;
        entry.push(Node::store(buffer_name, index, value));
    }
    if entry.is_empty() {
        return Err(unsupported(body, "a void kernel with no assignments"));
    }
    Ok(Program::wrapped(buffers, [1, 1, 1], entry))
}

fn lower_expression(
    node: SyntaxNode<'_>,
    source: &[u8],
    scalar: ScalarType,
    buffers: &HashMap<&str, &BufferParam>,
) -> Result<Expr, CFrontendError> {
    match node.kind() {
        "number_literal" => {
            let value = parse_integer_literal(node_text(node, source))
                .ok_or_else(|| unsupported(node, "a non-32-bit integer literal"))?;
            scalar.literal(value, node)
        }
        "parenthesized_expression" => {
            let inner = node
                .named_child(0)
                .ok_or_else(|| unsupported(node, "an empty parenthesized expression"))?;
            lower_expression(inner, source, scalar, buffers)
        }
        "subscript_expression" => {
            let (name, index) = lower_subscript(node, source, ScalarType::U32, buffers)?;
            let parameter = buffers
                .get(name.as_str())
                .ok_or_else(|| unsupported(node, "read from an unknown buffer"))?;
            if parameter.scalar != scalar {
                return Err(unsupported(node, "an implicit conversion between buffer element types"));
            }
            Ok(Expr::load(name, index))
        }
        "binary_expression" => {
            let left_node = required_field(node, "left", "a binary expression without a left operand")?;
            let right_node = required_field(node, "right", "a binary expression without a right operand")?;
            let left = lower_expression(left_node, source, scalar, buffers)?;
            let right = lower_expression(right_node, source, scalar, buffers)?;
            match binary_operator(node, left_node, right_node, source) {
                "+" => Ok(Expr::add(left, right)),
                "-" => Ok(Expr::sub(left, right)),
                "*" => Ok(Expr::mul(left, right)),
                "&" => Ok(Expr::bitand(left, right)),
                "|" => Ok(Expr::bitor(left, right)),
                "^" => Ok(Expr::bitxor(left, right)),
                _ => Err(unsupported(node, "an unsupported binary operator")),
            }
        }
        _ => Err(unsupported(
            node,
            &format!("the `{}` expression", node.kind()),
        )),
    }
}

fn lower_subscript(
    node: SyntaxNode<'_>,
    source: &[u8],
    index_type: ScalarType,
    buffers: &HashMap<&str, &BufferParam>,
) -> Result<(String, Expr), CFrontendError> {
    if node.kind() != "subscript_expression" {
        return Err(unsupported(node, "an assignment target other than a buffer subscript"));
    }
    let argument = required_field(node, "argument", "a subscript without a buffer")?;
    if argument.kind() != "identifier" {
        return Err(unsupported(argument, "a computed buffer expression"));
    }
    let name = node_text(argument, source).to_owned();
    if !buffers.contains_key(name.as_str()) {
        return Err(unsupported(argument, "an unknown buffer parameter"));
    }
    let index_node = required_field(node, "index", "a subscript without an index")?;
    let index = lower_expression(index_node, source, index_type, buffers)?;
    Ok((name, index))
}

fn parse_scalar_type(node: SyntaxNode<'_>, text: &str) -> Result<ScalarType, CFrontendError> {
    let normalized = normalized_type(text);
    match normalized.as_str() {
        "unsigned" | "unsigned int" | "uint32_t" | "uint" => Ok(ScalarType::U32),
        "int" | "signed" | "signed int" | "int32_t" => Ok(ScalarType::I32),
        _ => Err(unsupported(node, "a scalar type other than 32-bit int")),
    }
}

fn normalized_type(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn parse_integer_literal(text: &str) -> Option<u32> {
    let digits = text.trim_end_matches(|character: char| matches!(character, 'u' | 'U' | 'l' | 'L'));
    let (radix, digits) = if let Some(hex) = digits.strip_prefix("0x").or_else(|| digits.strip_prefix("0X")) {
        (16, hex)
    } else if let Some(binary) = digits.strip_prefix("0b").or_else(|| digits.strip_prefix("0B")) {
        (2, binary)
    } else if digits.len() > 1 && digits.starts_with('0') {
        (8, &digits[1..])
    } else {
        (10, digits)
    };
    u32::from_str_radix(digits, radix).ok()
}

fn function_name<'a>(function: SyntaxNode<'a>, source: &'a [u8]) -> Option<&'a str> {
    let declarator = function.child_by_field_name("declarator")?;
    let function_declarator = find_kind(declarator, "function_declarator")?;
    let name_declarator = function_declarator.child_by_field_name("declarator")?;
    let identifier = find_kind(name_declarator, "identifier")?;
    identifier.utf8_text(source).ok()
}

fn required_field<'a>(
    node: SyntaxNode<'a>,
    field: &str,
    construct: &str,
) -> Result<SyntaxNode<'a>, CFrontendError> {
    node.child_by_field_name(field)
        .ok_or_else(|| unsupported(node, construct))
}

fn find_kind<'a>(node: SyntaxNode<'a>, kind: &str) -> Option<SyntaxNode<'a>> {
    if node.kind() == kind {
        return Some(node);
    }
    let mut cursor = node.walk();
    let found = node
        .named_children(&mut cursor)
        .find_map(|child| find_kind(child, kind));
    found
}

fn named_children(node: SyntaxNode<'_>) -> Vec<SyntaxNode<'_>> {
    let mut cursor = node.walk();
    node.named_children(&mut cursor).collect()
}

fn node_text<'a>(node: SyntaxNode<'_>, source: &'a [u8]) -> &'a str {
    node.utf8_text(source)
        .expect("source was validated as UTF-8 before syntax traversal")
}

fn assignment_operator<'a>(node: SyntaxNode<'_>, source: &'a [u8]) -> &'a str {
    let left = node.child_by_field_name("left");
    let right = node.child_by_field_name("right");
    match (left, right) {
        (Some(left), Some(right)) => std::str::from_utf8(&source[left.end_byte()..right.start_byte()])
            .unwrap_or("")
            .trim(),
        _ => "",
    }
}

fn binary_operator<'a>(
    _node: SyntaxNode<'_>,
    left: SyntaxNode<'_>,
    right: SyntaxNode<'_>,
    source: &'a [u8],
) -> &'a str {
    std::str::from_utf8(&source[left.end_byte()..right.start_byte()])
        .unwrap_or("")
        .trim()
}

fn unsupported(node: SyntaxNode<'_>, construct: &str) -> CFrontendError {
    CFrontendError::Unsupported {
        byte: node.start_byte(),
        construct: construct.to_owned(),
    }
}

fn first_syntax_error(node: SyntaxNode<'_>) -> Option<SyntaxNode<'_>> {
    if node.is_error() || node.is_missing() {
        return Some(node);
    }
    if !node.has_error() {
        return None;
    }
    let mut cursor = node.walk();
    let found = node.children(&mut cursor).find_map(first_syntax_error);
    found
}

fn syntax_error(source: &str, node: SyntaxNode<'_>) -> CFrontendError {
    let Point { row, column } = node.start_position();
    let start = node.start_byte().min(source.len());
    let end = node.end_byte().max(start).min(source.len()).min(start.saturating_add(24));
    let mut fragment = source[start..end]
        .chars()
        .flat_map(char::escape_default)
        .collect::<String>();
    if fragment.is_empty() {
        fragment.push_str("<missing>");
    }
    CFrontendError::Syntax {
        byte: start,
        line: row + 1,
        column: column + 1,
        fragment,
    }
}
