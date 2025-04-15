.PHONY: all build clean test

# Default target
all: build

# Build the Rust library and Golang binding
build: rust build-router

# Build the Rust library
rust:
	@echo "Building Rust library..."
	cd candle-binding && cargo build --release

# Build router
build-router: rust
	@echo "Building router..."
	@mkdir -p bin
	@cd semantic_router && go build -o ../bin/router cmd/main.go

# Run the router
run-router: build-router
	@echo "Running router..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
		./bin/router -config=config/config.yaml

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
	rm -f bin/router
