.PHONY: all build clean

# Default target
all: build

# Build the Rust library and Golang binding
build: rust

# Build the Rust library
rust:
	@echo "Building Rust library..."
	cd candle-binding && cargo build --release

# Clean built artifacts
clean:
	@echo "Cleaning build artifacts..."
	cd candle-binding && cargo clean

