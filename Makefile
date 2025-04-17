.PHONY: all build clean test

# Default target
all: build

# vLLM env var
VLLM_ENDPOINT ?= http://192.168.12.175:11434

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

# Run Envoy proxy
run-envoy:
	@echo "Starting Envoy..."
	envoy --config-path config/envoy.yaml --component-log-level ext_proc:debug

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

# Test the Envoy extproc
test-prompt:
	@echo "Testing Envoy extproc with curl..."
	curl -X POST http://localhost:8801/v1/chat/completions \
		-H "Content-Type: application/json" \
		-H "Authorization: Bearer test-token" \
		-d '{"model": "qwen2.5:32b", "messages": [{"role": "assistant", "content": "You are a professional math teacher. Explain math concepts clearly and show step-by-step solutions to problems."}, {"role": "user", "content": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?"}], "temperature": 0.7}'

test-vllm:
	@echo "Testing vLLM at $(VLLM_ENDPOINT)..."
	curl -X POST $(VLLM_ENDPOINT)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "qwen2.5:32b", "messages": [{"role": "assistant", "content": "You are a professional math teacher. Explain math concepts clearly and show step-by-step solutions to problems."}, {"role": "user", "content": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?"}], "temperature": 0.7}'
