# fin_decimal - Rust High-Performance Decimal Fixed-point Arithmetic

[![Build Status](https://travis-ci.org/vsukhoml/fin_decimal.svg?branch=master)](https://travis-ci.org/vsukhoml/fin_decimal)

`fin_decimal` is a high-performance, `#![no_std]` compatible decimal fixed-point library tailored specifically for financial and tax computations. 

Unlike arbitrary-precision "big decimal" libraries that are slow and heap-allocate, or standard floating-point numbers (`f32`/`f64`) which suffer from rounding errors and precision loss, `fin_decimal` operates on an implicit scaling factor over a standard 64-bit signed integer (`i64`). 

This means additions and subtractions compile down to single, lightning-fast native CPU instructions, while multiplications and divisions utilize 128-bit math (with platform-specific inline assembly on `x86_64`) to guarantee strictly compliant decimal rounding without floating-point artifacts.

## Features
* **Const Generics (`Decimal<const DIGITS: u8>`)**: Zero-cost abstraction over multiple precision types.
* **`Amount64` (4 Decimal Digits)**: The standard precision for accounting, monetary amounts, and ledgers (up to ~±922 billion).
* **`Rate64` (8 Decimal Digits)**: A higher-precision type used for exact forex/exchange rates and interest calculations.
* **Strict Rounding Modes**: Explicit `.round_to(mode)`, `.mul_rounded()`, and `.div_rounded()` methods to ensure compliance with arbitrary tax codes (e.g. `HalfUp`, `HalfEven` Banker's Rounding, `Down`, `Up`).
* **Checked Math**: Built-in methods like `.checked_add()` to prevent overflow panics in mission-critical applications.
* **Serde Support**: Optional string-based JSON serialization via the `serde` feature, preventing transit precision loss over REST APIs.
* **`no_std` Environment Support**: Works gracefully in embedded contexts or core-only systems.

## Getting Started

Add `fin_decimal` to your `Cargo.toml`. To enable JSON serialization, include the `serde` feature:

```toml
[dependencies]
fin_decimal = { version = "0.1", features = ["serde"] }
```

## Basic Usage

The library provides trait overloads, so the standard `+`, `-`, `*`, `/`, and `%` operators work ergonomically alongside native primitives. By default, standard multiplication and division utilize standard "Half-Up" financial rounding.

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

Because `Amount64` is backed by a 64-bit integer, values exceeding `~922,337,203,685` (for 4 decimal digits) will wrap or panic in debug mode depending on your build settings. 

For robust backends, always use the `checked_*` routines to gracefully handle theoretical overflows:

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

## Exchange Rates (`Rate64`)

If you are dealing with fractional percentages or Forex multi-currency pipelines, 4 decimal places is rarely enough. The `Rate64` type provides 8 decimal places of precision automatically.

```rust
use fin_decimal::{Amount64, Rate64};

fn main() {
    let usd_to_jpy = Rate64::from(150.12345678);
    println!("Exchange Rate: {}", usd_to_jpy); // Outputs: "150.12345678"
}
```

## Why Not `f64` or Big Numbers?

1. **Floating Point (`f64`)**: Floats cannot perfectly represent base-10 decimals. `0.1 + 0.2` famously equals `0.30000000000000004` in floating-point math, violating strict accounting properties.
2. **Big Numbers (e.g. `num-bigint` or `rust-decimal`)**: These libraries perform heap allocations on virtually every math operation and represent numbers internally as slow arrays or `Vec`s. `fin_decimal` relies purely on `i64` CPU registers, making it orders of magnitude faster.
3. **The Sweet Spot**: The ±9.22 trillion limit of `Amount64` is vastly more than sufficient for 99% of general ledger entries, eCommerce carts, and standard banking, while reaping the ultimate performance benefits of fixed-width hardware math.