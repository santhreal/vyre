@echo off
setlocal

if not defined CC set "CC=cl"
if not defined VYRE_CARGO_RUNNER (
    if defined CARGO (
        set "VYRE_CARGO_RUNNER=%CARGO%"
    ) else (
        set "VYRE_CARGO_RUNNER=cargo"
    )
)

rem Nested Rust commands require an executable runner, not this batch file.
cargo %*
