.PHONY: all build clean test

# Default target
all: build

# Build the Rust library and Golang binding
build: rust

# Build the Rust library
rust:
	@echo "Building Rust library..."
	cd candle-binding && cargo build --release

# Test the Rust library
test-binding:
	@echo "Running Go tests with static library..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
		cd candle-binding && CGO_ENABLED=1 go test -v

# Test the Rust library and the Go binding
test: test-binding

# Clean built artifacts
clean:
	@echo "Cleaning build artifacts..."
	cd candle-binding && cargo clean

