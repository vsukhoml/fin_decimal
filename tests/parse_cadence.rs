//! Regression test pinning the 19-digit flush cadence.
//!
//! The parser buffers kept digits in one word and folds them into the wide
//! accumulator once per `CHUNK_DIGITS` digits. That flush is where `Overflow`
//! is *detected*, so the cadence is **observable**: for a string that both
//! overflows and later contains a stray byte, a 19-digit cadence returns
//! `InvalidDigit` where a 16-digit one returns `Overflow`. The values are
//! equal; only the error identity differs.
//!
//! Nothing else in this suite pins that. This file exists because a mutation
//! test found `CHUNK_DIGITS` 19 -> 16 and a swapped flush ordering both pass
//! `cargo test` unnoticed, while changing which error callers see. Any future
//! change to the accumulation cadence must either keep this table green or
//! be an explicit, reviewed decision to change the public error behaviour.
//!
//! The expected values were generated from the implementation as it stood
//! before the parser was rewritten (commit 040d87a), not from the current
//! code, so this pins the original semantics rather than restating the new
//! implementation back to itself.

use fin_decimal::{Amount64, Amount128, Amount256};
use std::str::FromStr;

fn render<T: std::fmt::Display, E: std::fmt::Debug>(r: Result<T, E>) -> String {
    match r {
        Ok(v) => format!("Ok(\"{v}\")"),
        Err(e) => format!("Err({e:?})"),
    }
}

/// `(input, expected Amount64, expected Amount128, expected Amount256)`
#[rustfmt::skip]
const CADENCE: &[(&str, &str, &str, &str)] = &[
        (r#"9x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"9.9999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"9"#, r#"Ok("9")"#, r#"Ok("9")"#, r#"Ok("9")"#),
        (r#"999999999999999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"999999999999999.9999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"999999999999999"#, r#"Err(Overflow)"#, r#"Ok("999999999999999")"#, r#"Ok("999999999999999")"#),
        (r#"9999999999999999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"9999999999999999.9999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"9999999999999999"#, r#"Err(Overflow)"#, r#"Ok("9999999999999999")"#, r#"Ok("9999999999999999")"#),
        (r#"99999999999999999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"99999999999999999.9999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"99999999999999999"#, r#"Err(Overflow)"#, r#"Ok("99999999999999999")"#, r#"Ok("99999999999999999")"#),
        (r#"999999999999999999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"999999999999999999.9999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"999999999999999999"#, r#"Err(Overflow)"#, r#"Ok("999999999999999999")"#, r#"Ok("999999999999999999")"#),
        (r#"9999999999999999999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"9999999999999999999.9999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"9999999999999999999"#, r#"Err(Overflow)"#, r#"Ok("9999999999999999999")"#, r#"Ok("9999999999999999999")"#),
        (r#"99999999999999999999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"99999999999999999999.9999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"99999999999999999999"#, r#"Err(Overflow)"#, r#"Ok("99999999999999999999")"#, r#"Ok("99999999999999999999")"#),
        (r#"999999999999999999999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"999999999999999999999.9999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"999999999999999999999"#, r#"Err(Overflow)"#, r#"Ok("999999999999999999999")"#, r#"Ok("999999999999999999999")"#),
        (r#"9999999999999999999999999999999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"9999999999999999999999999999999.9999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"9999999999999999999999999999999"#, r#"Err(Overflow)"#, r#"Ok("9999999999999999999999999999999")"#, r#"Ok("9999999999999999999999999999999")"#),
        (r#"99999999999999999999999999999999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"99999999999999999999999999999999.9999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"99999999999999999999999999999999"#, r#"Err(Overflow)"#, r#"Ok("99999999999999999999999999999999")"#, r#"Ok("99999999999999999999999999999999")"#),
        (r#"999999999999999999999999999999999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"999999999999999999999999999999999.9999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"999999999999999999999999999999999"#, r#"Err(Overflow)"#, r#"Ok("999999999999999999999999999999999")"#, r#"Ok("999999999999999999999999999999999")"#),
        (r#"9999999999999999999999999999999999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"9999999999999999999999999999999999.9999x"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"9999999999999999999999999999999999"#, r#"Err(Overflow)"#, r#"Ok("9999999999999999999999999999999999")"#, r#"Ok("9999999999999999999999999999999999")"#),
        (r#"99999999999999999999999999999999999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"99999999999999999999999999999999999.9999x"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"99999999999999999999999999999999999"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Ok("99999999999999999999999999999999999")"#),
        (r#"9999999999999999999999999999999999999x"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"9999999999999999999999999999999999999.9999x"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"9999999999999999999999999999999999999"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Ok("9999999999999999999999999999999999999")"#),
        (r#"99999999999999999999999999999999999999x"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"99999999999999999999999999999999999999.9999x"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"99999999999999999999999999999999999999"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Ok("99999999999999999999999999999999999999")"#),
        (r#"999999999999999999999999999999999999999x"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"999999999999999999999999999999999999999.9999x"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"999999999999999999999999999999999999999"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Ok("999999999999999999999999999999999999999")"#),
        (r#"9999999999999999999999999999999999999999x"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"9999999999999999999999999999999999999999.9999x"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#, r#"Err(InvalidDigit)"#),
        (r#"9999999999999999999999999999999999999999"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Ok("9999999999999999999999999999999999999999")"#),
        (r#"999999999999999999999999999999999999999999999999999999999x"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#),
        (r#"999999999999999999999999999999999999999999999999999999999.9999x"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#),
        (r#"999999999999999999999999999999999999999999999999999999999"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Ok("999999999999999999999999999999999999999999999999999999999")"#),
        (r#"9999999999999999999999999999999999999999999999999999999999x"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#),
        (r#"9999999999999999999999999999999999999999999999999999999999.9999x"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#),
        (r#"9999999999999999999999999999999999999999999999999999999999"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Ok("9999999999999999999999999999999999999999999999999999999999")"#),
        (r#"9999999999999999999999999999999999999999999999999999999999999999999999999999x"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#),
        (r#"9999999999999999999999999999999999999999999999999999999999999999999999999999.9999x"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#),
        (r#"9999999999999999999999999999999999999999999999999999999999999999999999999999"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#),
        (r#"99999999999999999999999999999999999999999999999999999999999999999999999999999x"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#),
        (r#"99999999999999999999999999999999999999999999999999999999999999999999999999999.9999x"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#),
        (r#"99999999999999999999999999999999999999999999999999999999999999999999999999999"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#),
        (r#"999999999999999999999999999999999999999999999999999999999999999999999999999999x"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#),
        (r#"999999999999999999999999999999999999999999999999999999999999999999999999999999.9999x"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Err(InvalidDigit)"#),
        (r#"999999999999999999999999999999999999999999999999999999999999999999999999999999"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#, r#"Err(Overflow)"#),
];

#[test]
fn flush_cadence_error_identity_is_stable() {
    let mut bad = Vec::new();
    for &(input, e64, e128, e256) in CADENCE {
        let got64 = render(Amount64::from_str(input));
        let got128 = render(Amount128::from_str(input));
        let got256 = render(Amount256::from_str(input));
        if got64 != e64 {
            bad.push(format!("Amount64  {input:?}: expected {e64}, got {got64}"));
        }
        if got128 != e128 {
            bad.push(format!(
                "Amount128 {input:?}: expected {e128}, got {got128}"
            ));
        }
        if got256 != e256 {
            bad.push(format!(
                "Amount256 {input:?}: expected {e256}, got {got256}"
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "flush cadence changed the observable error identity on {} case(s):\n{}",
        bad.len(),
        bad.join("\n")
    );
}
