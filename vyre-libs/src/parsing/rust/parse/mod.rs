//! Nano-subset Rust parser.
//!
//! Parses the token stream produced by `lex` into a minimal AST.
//! The nano-subset is intentionally tiny so that the parser can be
//! validated exhaustively and GPU-accelerated in v0.1.0.

use crate::parsing::rust::lex::lexer::core::Token;

/// Expression AST nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Integer or boolean literal.
    Literal(u16, String),
    /// Variable reference.
    Var(String),
    /// Binary operation: `lhs op rhs`.
    Binary { op: u16, lhs: Box<Expr>, rhs: Box<Expr> },
    /// Borrow expression: `&expr` or `&mut expr`.
    Borrow { mutable: bool, expr: Box<Expr> },
    /// Dereference: `*expr`.
    Deref(Box<Expr>),
    /// Function call: `name(args)`.
    Call { name: String, args: Vec<Expr> },
    /// Block expression.
    Block(Vec<Stmt>),
    /// Conditional.
    If { cond: Box<Expr>, then_block: Box<Expr>, else_block: Option<Box<Expr>> },
}

/// Statement AST nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// `let mut? name: Type = expr;`
    Let {
        mutable: bool,
        name: String,
        ty: Type,
        init: Expr,
    },
    /// Expression statement (usually a call or assignment).
    Expr(Expr),
    /// `return expr;`
    Return(Option<Expr>),
}

/// Types in the nano-subset.
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    I32,
    Bool,
    Unit,
    Ref { mutable: bool, inner: Box<Type> },
}

/// Function definition.
#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub ret: Type,
    pub body: Vec<Stmt>,
}

/// A parsed module (nano-subset: exactly one file = one function for now).
#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub functions: Vec<Function>,
}

/// Parse error with source span.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub token_index: usize,
}

/// Parse a token stream into a `Module`.
///
/// TODO(v0.0.1): CPU recursive-descent parser.  v0.1.0 target:
/// GPU shunting-yard over the token buffer.
pub fn parse(tokens: &[Token]) -> Result<Module, ParseError> {
    let mut parser = Parser::new(tokens);
    parser.parse_module()
}

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos.min(self.tokens.len() - 1)];
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, kind: u16) -> Result<&Token, ParseError> {
        let tok = self.peek();
        if tok.kind == kind {
            Ok(self.advance())
        } else {
            Err(ParseError {
                message: format!("expected token {}, got {}", kind, tok.kind),
                token_index: self.pos,
            })
        }
    }

    fn parse_module(&mut self) -> Result<Module, ParseError> {
        let mut functions = Vec::new();
        while self.peek().kind != crate::parsing::rust::lex::tokens::RUST_TOK_EOF {
            functions.push(self.parse_function()?);
        }
        Ok(Module { functions })
    }

    fn parse_function(&mut self) -> Result<Function, ParseError> {
        use crate::parsing::rust::lex::tokens::*;
        self.expect(RUST_TOK_FN)?;
        let name_tok = self.expect(RUST_TOK_IDENT)?;
        let name = format!("ident@{}", name_tok.start);
        self.expect(RUST_TOK_LPAREN)?;
        let params = self.parse_param_list()?;
        self.expect(RUST_TOK_RPAREN)?;
        let ret = if self.peek().kind == RUST_TOK_ARROW {
            self.advance();
            self.parse_type()?
        } else {
            Type::Unit
        };
        let body = self.parse_block()?;
        Ok(Function { name, params, ret, body })
    }

    fn parse_param_list(&mut self) -> Result<Vec<(String, Type)>, ParseError> {
        use crate::parsing::rust::lex::tokens::*;
        let mut params = Vec::new();
        if self.peek().kind == RUST_TOK_RPAREN {
            return Ok(params);
        }
        loop {
            let name_tok = self.expect(RUST_TOK_IDENT)?;
            let name = format!("param@{}", name_tok.start);
            self.expect(RUST_TOK_COLON)?;
            let ty = self.parse_type()?;
            params.push((name, ty));
            if self.peek().kind == RUST_TOK_COMMA {
                self.advance();
            } else {
                break;
            }
        }
        Ok(params)
    }

    fn parse_type(&mut self) -> Result<Type, ParseError> {
        use crate::parsing::rust::lex::tokens::*;
        match self.peek().kind {
            RUST_TOK_I32 => { self.advance(); Ok(Type::I32) }
            RUST_TOK_BOOL => { self.advance(); Ok(Type::Bool) }
            RUST_TOK_AMP | RUST_TOK_AMP_MUT => {
                let mutable = self.peek().kind == RUST_TOK_AMP_MUT;
                self.advance();
                let inner = self.parse_type()?;
                Ok(Type::Ref { mutable, inner: Box::new(inner) })
            }
            _ => Err(ParseError { message: "expected type".to_string(), token_index: self.pos }),
        }
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, ParseError> {
        use crate::parsing::rust::lex::tokens::*;
        self.expect(RUST_TOK_LBRACE)?;
        let mut stmts = Vec::new();
        while self.peek().kind != RUST_TOK_RBRACE && self.peek().kind != RUST_TOK_EOF {
            stmts.push(self.parse_stmt()?);
        }
        self.expect(RUST_TOK_RBRACE)?;
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        use crate::parsing::rust::lex::tokens::*;
        match self.peek().kind {
            RUST_TOK_LET => self.parse_let(),
            RUST_TOK_RETURN => self.parse_return(),
            _ => {
                let expr = self.parse_expr()?;
                self.expect(RUST_TOK_SEMI)?;
                Ok(Stmt::Expr(expr))
            }
        }
    }

    fn parse_let(&mut self) -> Result<Stmt, ParseError> {
        use crate::parsing::rust::lex::tokens::*;
        self.expect(RUST_TOK_LET)?;
        let mutable = if self.peek().kind == RUST_TOK_MUT { self.advance(); true } else { false };
        let name_tok = self.expect(RUST_TOK_IDENT)?;
        let name = format!("let@{}", name_tok.start);
        self.expect(RUST_TOK_COLON)?;
        let ty = self.parse_type()?;
        self.expect(RUST_TOK_ASSIGN)?;
        let init = self.parse_expr()?;
        self.expect(RUST_TOK_SEMI)?;
        Ok(Stmt::Let { mutable, name, ty, init })
    }

    fn parse_return(&mut self) -> Result<Stmt, ParseError> {
        use crate::parsing::rust::lex::tokens::*;
        self.expect(RUST_TOK_RETURN)?;
        let expr = if self.peek().kind != RUST_TOK_SEMI { Some(self.parse_expr()?) } else { None };
        self.expect(RUST_TOK_SEMI)?;
        Ok(Stmt::Return(expr))
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> { self.parse_comparison() }

    fn parse_comparison(&mut self) -> Result<Expr, ParseError> {
        use crate::parsing::rust::lex::tokens::*;
        let mut lhs = self.parse_term()?;
        while matches!(self.peek().kind, RUST_TOK_EQ | RUST_TOK_LT) {
            let op = self.advance().kind;
            let rhs = self.parse_term()?;
            lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_term(&mut self) -> Result<Expr, ParseError> {
        use crate::parsing::rust::lex::tokens::*;
        let mut lhs = self.parse_factor()?;
        while matches!(self.peek().kind, RUST_TOK_PLUS | RUST_TOK_MINUS) {
            let op = self.advance().kind;
            let rhs = self.parse_factor()?;
            lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_factor(&mut self) -> Result<Expr, ParseError> {
        use crate::parsing::rust::lex::tokens::*;
        let mut lhs = self.parse_unary()?;
        while matches!(self.peek().kind, RUST_TOK_STAR | RUST_TOK_SLASH) {
            let op = self.advance().kind;
            let rhs = self.parse_unary()?;
            lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        use crate::parsing::rust::lex::tokens::*;
        match self.peek().kind {
            RUST_TOK_AMP | RUST_TOK_AMP_MUT => {
                let mutable = self.peek().kind == RUST_TOK_AMP_MUT;
                self.advance();
                Ok(Expr::Borrow { mutable, expr: Box::new(self.parse_unary()?) })
            }
            RUST_TOK_STAR => { self.advance(); Ok(Expr::Deref(Box::new(self.parse_unary()?))) }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        use crate::parsing::rust::lex::tokens::*;
        match self.peek().kind {
            RUST_TOK_LITERAL_INT | RUST_TOK_LITERAL_BOOL => {
                let tok = self.advance();
                Ok(Expr::Literal(tok.kind, format!("lit@{}", tok.start)))
            }
            RUST_TOK_IDENT => {
                let name_tok = self.advance();
                let name = format!("ident@{}", name_tok.start);
                if self.peek().kind == RUST_TOK_LPAREN {
                    self.advance();
                    let mut args = Vec::new();
                    if self.peek().kind != RUST_TOK_RPAREN {
                        loop {
                            args.push(self.parse_expr()?);
                            if self.peek().kind == RUST_TOK_COMMA { self.advance(); } else { break; }
                        }
                    }
                    self.expect(RUST_TOK_RPAREN)?;
                    Ok(Expr::Call { name, args })
                } else {
                    Ok(Expr::Var(name))
                }
            }
            RUST_TOK_LBRACE => Ok(Expr::Block(self.parse_block()?)),
            RUST_TOK_IF => {
                self.advance();
                let cond = Box::new(self.parse_expr()?);
                let then_block = Box::new(Expr::Block(self.parse_block()?));
                let else_block = if self.peek().kind == RUST_TOK_ELSE {
                    self.advance();
                    if self.peek().kind == RUST_TOK_IF {
                        Some(Box::new(self.parse_expr()?))
                    } else {
                        Some(Box::new(Expr::Block(self.parse_block()?)))
                    }
                } else { None };
                Ok(Expr::If { cond, then_block, else_block })
            }
            _ => Err(ParseError { message: "unexpected token in expression".to_string(), token_index: self.pos }),
        }
    }
}
