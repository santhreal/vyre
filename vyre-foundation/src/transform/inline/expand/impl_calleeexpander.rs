/// Expression expansion: argument substitution, renaming, and call hoisting.
mod expressions;
/// Alpha-renaming of a callee-local name, and of every name in one expression.
mod rename;
/// Statement expansion, as a policy over the one structural node rewrite.
mod statements;
