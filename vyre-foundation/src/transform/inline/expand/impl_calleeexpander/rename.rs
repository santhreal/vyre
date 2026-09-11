use super::super::CalleeExpander;
use crate::ir::Ident;

impl CalleeExpander<'_> {
    /// Give one callee-local name a caller-unique spelling, and record it so
    /// every later use of that name resolves to the same spelling.
    #[inline]
    pub(crate) fn rename_decl(&mut self, name: &Ident) -> String {
        let renamed = format!("{}{name}", self.prefix);
        self.vars.insert(Ident::from(name), renamed.clone());
        renamed
    }

    /// The caller-side spelling of one name a use refers to.
    ///
    /// A name with no recorded declaration is a caller name reached through a
    /// substituted argument, so it is already in the caller's namespace.
    #[inline]
    pub(crate) fn rename_use(&self, name: &Ident) -> String {
        self.vars
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string())
    }
}
