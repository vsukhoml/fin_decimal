#!/usr/bin/env bash
# Full verification gate for fin_decimal: formatting, lints, the test matrix
# (default / asm / all-features), the assembly codegen contract, and test
# coverage with a regression floor.
#
# Usage: ./test.sh
# Exits non-zero on the first failing stage.
set -euo pipefail
cd "$(dirname "$0")"

# Minimum acceptable line coverage (%, whole crate, lib tests). Current level
# is ~89%; the floor is set below it to catch real regressions, not noise.
# Raise it as coverage improves.
COVERAGE_MIN_LINES=85

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "Format check (cargo fmt --check)"
cargo fmt --all -- --check

step "Lints (cargo clippy --all-features --all-targets, warnings are errors)"
cargo clippy --all-features --all-targets -- -D warnings

step "Tests: default features"
cargo test

step "Tests: asm feature"
cargo test --features asm

step "Tests: all features"
cargo test --all-features

step "Assembly codegen contract (check_asm.sh)"
./scripts/check_asm.sh

step "Assembly codegen contract (check_asm.sh asm)"
./scripts/check_asm.sh asm

if command -v cargo-llvm-cov > /dev/null; then
    step "Coverage (cargo llvm-cov --lib, floor: ${COVERAGE_MIN_LINES}% lines)"
    cargo llvm-cov --lib --summary-only --fail-under-lines "$COVERAGE_MIN_LINES"
else
    step "Coverage"
    echo "error: cargo-llvm-cov not installed (cargo install cargo-llvm-cov)" >&2
    exit 1
fi

printf '\n\033[1;32mAll gates passed.\033[0m\n'
