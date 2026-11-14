use crate::validate::{err, Binding, ValidationError, ValidationOptions};
use crate::validate::{ValidationLocation, ValidationPhase};
use rustc_hash::FxHashMap;

#[inline]
pub(crate) fn check_local(
    name: &str,
    scope: &FxHashMap<crate::ir::Ident, Binding>,
    options: ValidationOptions<'_>,
    errors: &mut Vec<ValidationError>,
) {
    if !options.allow_shadowing && scope.contains_key(name) {
        errors.push(err("V008", ValidationPhase::Node, ValidationLocation::Program, format!(
            "duplicate local binding `{name}` shadows an outer scope"
        ), "choose a unique local name, or opt into nested shadowing with ValidationOptions::with_shadowing(true)."));
    }
}

#[cfg(test)]
mod tests {
    use super::super::uniformity::tests::scope_with_binding;
    use super::*;

    #[test]
    fn shadowing_detected_by_default() {
        let scope = scope_with_binding("x", true);
        let mut errors = Vec::new();
        check_local("x", &scope, ValidationOptions::default(), &mut errors);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].code().as_str() == "V008");
    }

    #[test]
    fn shadowing_allowed_with_opt_in() {
        let scope = scope_with_binding("x", true);
        let mut errors = Vec::new();
        let options = ValidationOptions::default().with_shadowing(true);
        check_local("x", &scope, options, &mut errors);
        assert!(errors.is_empty());
    }

    #[test]
    fn new_name_never_errors() {
        let scope = scope_with_binding("x", true);
        let mut errors = Vec::new();
        check_local("y", &scope, ValidationOptions::default(), &mut errors);
        assert!(errors.is_empty());
    }
}
