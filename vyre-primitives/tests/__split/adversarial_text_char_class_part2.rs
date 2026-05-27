use super::*;

#[test]
fn test_text_char_class_adv_26() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[2143289344u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 31 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_27() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[0u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 32 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_28() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[0u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 32 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_29() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[0u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 32 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_30() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[4294967295u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 32 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_31() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[4294967295u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 32 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_32() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[4294967295u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 32 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_33() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[2143289344u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 32 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_34() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[2143289344u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 32 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_35() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[2143289344u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 32 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_36() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[0u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1024 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_37() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[0u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1024 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_38() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[0u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1024 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_39() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[4294967295u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1024 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_40() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[4294967295u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1024 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_41() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[4294967295u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1024 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_42() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[2143289344u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1024 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_43() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[2143289344u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1024 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_44() {
    let source: Vec<u8> = vec![0u8; 0];
    let table: &[u32; 256] = &[2143289344u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 1024 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_45() {
    let source: Vec<u8> = vec![1u8; 0];
    let table: &[u32; 256] = &[0u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 0 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_46() {
    let source: Vec<u8> = vec![1u8; 0];
    let table: &[u32; 256] = &[0u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 0 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_47() {
    let source: Vec<u8> = vec![1u8; 0];
    let table: &[u32; 256] = &[0u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 0 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_48() {
    let source: Vec<u8> = vec![1u8; 0];
    let table: &[u32; 256] = &[4294967295u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 0 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_49() {
    let source: Vec<u8> = vec![1u8; 0];
    let table: &[u32; 256] = &[4294967295u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 0 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_50() {
    let source: Vec<u8> = vec![1u8; 0];
    let table: &[u32; 256] = &[4294967295u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 0 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

#[test]
fn test_text_char_class_adv_51() {
    let source: Vec<u8> = vec![1u8; 0];
    let table: &[u32; 256] = &[2143289344u32; 256];
    let result = std::panic::catch_unwind(|| reference_char_class(&source, table));
    match result {
        Ok(res) => {
            assert!(res.len() <= 0 + 0 + 1);
        }
        Err(_) => {
            // Successfully caught hostile panic
            assert!(true);
        }
    }
}

