# LLM Semantic Router

An Envoy External Processor (ExtProc) that routes OpenAI API requests to optimal model backends based on semantic content matching using BERT similarity matching (current implementation) and classification (future implementation).

The router is implemented in two ways: Golang (with Rust FFI based on Candle) and Python. Benchmarking will be conducted to determine the best implementation.

## Usage

### Go Implementation
```bash
make run-router
```




