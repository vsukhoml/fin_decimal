# fin_decimal - Rust High-Performance Decimal Fixed-point Arithmetic

[![Build Status](https://travis-ci.org/vsukhoml/fin_decimal.svg?branch=master)](https://travis-ci.org/vsukhoml/fin_decimal)

`fin_decimal` is a high-performance, `#![no_std]` compatible decimal fixed-point library tailored specifically for financial and tax computations.

Unlike arbitrary-precision "big decimal" libraries that are slow and heap-allocate, or standard floating-point numbers (`f32`/`f64`) which suffer from rounding errors and precision loss, `fin_decimal` operates on an implicit power-of-10 scaling factor over a fixed-width signed integer.

This means additions and subtractions compile down to a few native CPU instructions, while multiplications and divisions use double-width integer math with strictly compliant decimal rounding and no floating-point artifacts. Re-scaling by the constant `10^DIGITS` is done via reciprocal multiplication with compile-time reciprocals — no division instructions at all on that path, verified by an assembly-inspection script.

## Types

Three backing widths share identical decimal semantics (verified against each other by large differential test suites):

| Backing | Type | Aliases (4 / 8 fractional digits) | Range at 4 digits |
|---|---|---|---|
| `i64` | `Decimal<DIGITS>` | `Amount64` / `Rate64` | ~±922 trillion |
| `i128` | `Decimal128<DIGITS>` | `Amount128` / `Rate128` | ~±1.7 · 10³⁴ |
| 256-bit | `Decimal256<DIGITS>` | `Amount256` / `Rate256` | ~±5.8 · 10⁷² |

Pick `Amount64` for ledgers and carts, `Amount128` when aggregating very large books or working in minor units of high-inflation currencies, and `Amount256` when products of large amounts and high-precision rates must stay exact.

## Features
* **Const Generics (`Decimal<const DIGITS: u8>`)**: Zero-cost abstraction over multiple precision types (`DIGITS ≤ 19` for the wide types).
* **Strict Rounding Modes**: Explicit `.round_to(mode)`, `.mul_rounded()`, and `.div_rounded()` methods to ensure compliance with arbitrary tax codes (`HalfUp`, `HalfEven` Banker's Rounding, `HalfDown`, `Down`, `Up`).
* **Checked Math**: `.checked_add/sub/mul/div()` returning `Option` for mission-critical code paths; symmetric range (`MIN == -MAX`).
* **Compile-Time Evaluation**: parsing, multiplication, rounding, and `powi` are `const fn`, so whole derived constants are computed by the compiler (see below).
* **Zero Heap Allocations**: `no_std` without `alloc`; every buffer is a fixed-size stack array, including string formatting and parsing.
* **Serde Support**: Optional string-based serialization via the `serde` feature, preventing precision loss in transit over APIs. Optional `ufmt` support for embedded targets.
* **(mantissa, exponent) interchange**: `to_decimal_parts` / `from_decimal_parts` (exact-or-error) and `from_decimal_parts_rounded` for codecs, on the `i64` and `i128` types.

## Getting Started

Add `fin_decimal` to your `Cargo.toml`. To enable JSON serialization, include the `serde` feature:

```toml
[dependencies]
fin_decimal = { version = "0.1", features = ["serde"] }
```

## Basic Usage

The library provides trait overloads, so the standard `+`, `-`, `*`, `/`, and `%` operators work ergonomically alongside native primitives. By default, multiplication and division use "Half-Up" financial rounding (half away from zero).

```rust
use fin_decimal::Amount64;
use core::str::FromStr;

fn main() {
    // Initialization
    let a = Amount64::from(3);              // 3.0000
    let b = Amount64::from(2.02f64);        // 2.0200
    let c = Amount64::from_str("1.50").unwrap();

    // Standard Math Traits
    let sum = a + b;                        // 5.0200
    let diff = a - c;                       // 1.5000

    // Interactions with native integers
    let multiplied = sum * 2;               // 10.0400

    println!("Total: {}", multiplied);      // Outputs: "Total: 10.04"
}
```

## Compile-Time Constants

Rates, fees, and even derived values can be evaluated entirely at compile time. Invalid literals become compile errors:

```rust
use fin_decimal::{Amount64, Rounding};

const PRICE: Amount64 = Amount64::from_str_const("19.99");
const TAX_RATE: Amount64 = Amount64::from_str_const("0.0825");
const TAX: Amount64 = PRICE.mul_rounded(TAX_RATE, Rounding::HalfEven);
const GROWTH_10Y: Amount64 = Amount64::from_str_const("1.05").powi(10);
```

## Explicit Rounding & Compliance

When calculating taxes or complex financial algorithms, regulations often dictate exact rounding constraints. Use the `Rounding` enum to dictate exactly how calculations terminate.

```rust
use fin_decimal::{Amount64, Rounding};

fn main() {
    let tax_rate = Amount64::from(0.075); // 7.5%
    let item_price = Amount64::from(19.99);

    // Explicitly round tax down to benefit the consumer,
    // rather than using standard Half-Up.
    let tax_owed = item_price.mul_rounded(tax_rate, Rounding::Down);

    // You can also explicitly round arbitrary values
    let val = Amount64::from(2.5);
    assert_eq!(val.round_to(Rounding::HalfEven), Amount64::from(2)); // Banker's Rounding
    assert_eq!(val.round_to(Rounding::HalfUp), Amount64::from(3));   // Standard Rounding
}
```

## Safety and Overflow (Checked Math)

`Amount64` is backed by a 64-bit integer: values beyond ~±922,337,203,685,477.5807 do not fit. The wide types' `*` and `/` operators panic on overflow in every build profile; for robust backends, use the `checked_*` routines to handle overflow gracefully:

```rust
use fin_decimal::Amount64;

fn main() {
    let massive_balance = Amount64::MAX;
    let deposit = Amount64::from(100);

    match massive_balance.checked_add(deposit) {
        Some(new_balance) => println!("Success: {}", new_balance),
        None => println!("Transaction failed: Account balance overflow!"),
    }
}
```

## Wide Amounts (`Amount128`, `Amount256`)

When 64 bits are not enough, the wider types keep the same semantics and API:

```rust
use fin_decimal::{Amount128, Amount256};
use core::str::FromStr;

fn main() {
    // Exact math far beyond the i64 range:
    let big = Amount128::from(10_000_000_000_000_000i64); // 10^16
    assert_eq!((big * big).to_string(), "100000000000000000000000000000000");

    // 72-digit values round-trip exactly through strings:
    let huge = Amount256::from_str(
        "123456789012345678901234567890123456789012345678901234567890.1234",
    ).unwrap();
    assert_eq!(huge, huge.trunc() + huge.fract());
}
```

## Exchange Rates (`Rate64`)

If you are dealing with fractional percentages or Forex multi-currency pipelines, 4 decimal places is rarely enough. The `Rate64` type provides 8 decimal places of precision automatically (`Rate128`/`Rate256` likewise).

```rust
use fin_decimal::Rate64;

fn main() {
    let usd_to_jpy = Rate64::from(150.12345678);
    println!("Exchange Rate: {}", usd_to_jpy); // Outputs: "150.12345678"
}
```

## Performance

Every arithmetic path is allocation-free, and constant re-scaling is division-instruction-free. Measured on x86-64 (`cargo bench`, ns per operation):

| Operation | `Amount64` | `Amount128` | `Amount256` |
|---|---|---|---|
| add | 0.4 | 0.5 | 1.1 |
| multiply (typical money) | 2.8 | 7.9 | 14.6 |
| divide (small divisor) | 1.7 | 9.2 | 15.1 |
| divide (full-width divisor) | 3.5 | 14.5 | 38–56 |
| round_to | 1.1 | 11.7 | 11.2 |

* `cargo bench` — self-contained performance suite across operand and divisor sizes (no external harness).
* `./scripts/check_asm.sh [asm]` — builds `examples/asm_probe.rs` and verifies the generated assembly: constant re-scaling contains **zero** division instructions, and no path calls the compiler's 128-bit division builtins (`__udivti3`), which are slow software routines on most non-x86 targets.
* The optional `asm` feature uses the native `x86_64` 128÷64 `div` instruction for division by runtime divisors. Measure before enabling: on modern x86 it is roughly a wash.

## Why Not `f64` or Big Numbers?

1. **Floating Point (`f64`)**: Floats cannot perfectly represent base-10 decimals. `0.1 + 0.2` famously equals `0.30000000000000004` in floating-point math, violating strict accounting properties.
2. **Big Numbers (e.g. `num-bigint` or `rust-decimal`)**: These libraries perform heap allocations on virtually every math operation and represent numbers internally as slow arrays or `Vec`s. `fin_decimal` relies purely on fixed-width CPU registers and stack arrays, making it orders of magnitude faster.
3. **The Sweet Spot**: The ~±922 trillion limit of `Amount64` is vastly more than sufficient for most general ledger entries, eCommerce carts, and standard banking — and when it isn't, `Amount128`/`Amount256` extend the range without changing the semantics or the allocation story.
