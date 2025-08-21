# Installation Guide

This guide will help you set up and install the Semantic Router on your system. The installation process includes setting up dependencies, downloading models, and configuring the routing system.

## System Requirements

### Hardware Requirements

```yaml
minimum_requirements:
  cpu: "4 cores, 2.4GHz"
  memory: "8GB RAM"
  storage: "20GB free space"
  network: "Stable internet connection"
  
recommended_requirements:
  cpu: "8+ cores, 3.0GHz"
  memory: "32GB RAM" 
  storage: "100GB SSD"
  gpu: "NVIDIA RTX 3080 or equivalent (for model training)"
  network: "High-speed internet connection"
```

### Software Dependencies

- **Go**: Version 1.19 or higher
- **Rust**: Version 1.70 or higher (for Candle bindings)
- **Python**: Version 3.8+ (for model training utilities)
- **Envoy Proxy**: Version 1.25+
- **HuggingFace CLI**: For model downloads

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/semantic-router/semantic-router.git
cd semantic-router
```

### 2. Install Go Dependencies

```bash
# Initialize Go modules
go mod download
go mod tidy

# Verify Go installation
go version
```

### 3. Install Rust and Candle Dependencies

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Navigate to candle-binding directory
cd candle-binding

# Build Rust components
cargo build --release

# Build WASM components (optional, for web deployment)
./build-wasm.sh

cd ..
```

### 4. Install Python Dependencies (Optional)

For model training and evaluation utilities:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install training framework dependencies
pip install -r unified_fine_tuning/requirements.txt
pip install -r intent_classification_fine_tuning/requirements.txt
```

### 5. Install Envoy Proxy

#### Option A: Using Package Manager

```bash
# Ubuntu/Debian
curl -sL 'https://deb.dl.getenvoy.io/public/gpg.8115BA8E629CC074.key' | sudo gpg --dearmor -o /usr/share/keyrings/getenvoy-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/getenvoy-keyring.gpg] https://deb.dl.getenvoy.io/public/deb/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/getenvoy.list
sudo apt update && sudo apt install -y getenvoy-envoy

# CentOS/RHEL
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://rpm.dl.getenvoy.io/public/rpm/el/getenvoy-envoy.repo
sudo yum install getenvoy-envoy

# macOS
brew install envoy
```

#### Option B: Using Docker

```bash
# Pull Envoy Docker image
docker pull envoyproxy/envoy:v1.25-latest
```

### 6. Install HuggingFace CLI

```bash
# Install HuggingFace CLI
pip install --upgrade huggingface_hub

# Verify installation
huggingface-cli --help
```

### 7. Download Pre-trained Models

```bash
# Download all required models
make download-models

# Or download individually
huggingface-cli download HuaminChen/category_classifier_modernbert-base_model --local-dir models/category_classifier_modernbert-base_model
huggingface-cli download HuaminChen/pii_classifier_modernbert-base_model --local-dir models/pii_classifier_modernbert-base_model
huggingface-cli download HuaminChen/jailbreak_classifier_modernbert-base_model --local-dir models/jailbreak_classifier_modernbert-base_model
```

## Configuration

### 1. Basic Configuration

Copy the example configuration file:

```bash
cp config/config.yaml.example config/config.yaml
```

Edit the configuration file to match your setup:

```yaml
# config/config.yaml
router:
  # Server configuration
  port: 50051
  log_level: "info"
  
  # Model paths
  models:
    category_classifier: "./models/category_classifier_modernbert-base_model"
    pii_detector: "./models/pii_classifier_modernbert-base_model" 
    jailbreak_guard: "./models/jailbreak_classifier_modernbert-base_model"
    
  # Backend endpoints
  endpoints:
    endpoint1:
      url: "http://localhost:11434"
      model_type: "math"
      timeout: 300
    endpoint2:
      url: "http://localhost:11435"
      model_type: "creative"
      timeout: 300
    endpoint3:
      url: "http://localhost:11436"
      model_type: "general"
      timeout: 300
      
  # Security settings
  security:
    enable_pii_detection: true
    enable_jailbreak_guard: true
    pii_action: "block"
    
  # Cache configuration
  cache:
    enabled: true
    similarity_threshold: 0.85
    ttl_seconds: 3600
    max_entries: 10000
```

### 2. Envoy Configuration

The default Envoy configuration should work for most setups. If needed, customize:

```bash
cp config/envoy.yaml.example config/envoy.yaml
# Edit config/envoy.yaml as needed
```

## Build and Run

### 1. Build the Semantic Router

```bash
# Build the Go application with Rust FFI
make build

# Or build manually
cd candle-binding
cargo build --release
cd ..
go build -o bin/router ./cmd/router
```

### 2. Start the Services

#### Terminal 1: Start Envoy Proxy
```bash
make run-envoy
# Or manually:
# envoy -c config/envoy.yaml
```

#### Terminal 2: Start Semantic Router
```bash
make run-router  
# Or manually:
# ./bin/router -config config/config.yaml
```

### 3. Verify Installation

Test the installation with a sample request:

```bash
# Test basic routing
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "What is 2 + 2?"}
    ]
  }'
```

## Docker Installation (Alternative)

For a containerized setup:

### 1. Using Docker Compose

```bash
# Copy docker compose file
cp docker-compose.yml.example docker-compose.yml

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### 2. Build Custom Docker Image

```bash
# Build the semantic router image
docker build -t semantic-router:latest .

# Run with custom configuration
docker run -d \
  -p 50051:50051 \
  -p 8801:8801 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/models:/app/models \
  semantic-router:latest
```

## Verification and Testing

### 1. Health Checks

```bash
# Check Envoy health
curl http://localhost:8801/stats

# Check router health  
curl http://localhost:50051/health
```

### 2. Run Test Suite

```bash
# Install test dependencies
pip install -r e2e-tests/requirements.txt

# Run comprehensive tests
cd e2e-tests
python run_all_tests.py

# Run specific test
python 00-client-request-test.py
```

### 3. Performance Testing

```bash
# Load test with sample requests
make test-load

# Monitor performance
make monitor
```

## Troubleshooting

### Common Issues

#### 1. Model Download Failures
```bash
# Check HuggingFace CLI authentication
huggingface-cli whoami

# Login if needed
huggingface-cli login

# Retry download
make download-models
```

#### 2. Port Conflicts
```bash
# Check if ports are in use
lsof -i :8801
lsof -i :50051

# Kill conflicting processes or change ports in config
```

#### 3. Memory Issues
```bash
# Check memory usage
free -h

# Reduce model cache size in config
# Or add swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. Rust/Go Build Errors
```bash
# Update Rust
rustup update

# Clean and rebuild
cd candle-binding
cargo clean
cargo build --release

# Update Go modules
go clean -modcache
go mod download
```

### Getting Help

- **Documentation**: Check the full documentation at [docs/](../index.md)
- **Issues**: Report issues on [GitHub Issues](https://github.com/semantic-router/semantic-router/issues)
- **Discussions**: Join discussions on [GitHub Discussions](https://github.com/semantic-router/semantic-router/discussions)
- **Community**: Join our community [Discord/Slack]

## Next Steps

After successful installation:

1. **[Quick Start Guide](quick-start.md)**: Learn basic usage
2. **[Configuration Guide](configuration.md)**: Customize your setup
3. **[Architecture Overview](../architecture/system-architecture.md)**: Understand the system design
4. **[Model Training](../training/training-overview.md)**: Train custom models

Congratulations! You now have a fully functional Semantic Router installation.
