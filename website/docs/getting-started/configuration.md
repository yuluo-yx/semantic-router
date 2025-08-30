# Configuration Guide

This guide covers the configuration options for the Semantic Router. The system uses a single YAML configuration file that controls all aspects of routing, classification, and security.

## Configuration File

The configuration file is located at `config/config.yaml`. Here's the structure based on the actual implementation:

```yaml
# config/config.yaml - Actual configuration structure

# BERT model for semantic similarity
bert_model:
  model_id: sentence-transformers/all-MiniLM-L12-v2
  threshold: 0.6
  use_cpu: true

# Semantic caching
semantic_cache:
  enabled: false
  similarity_threshold: 0.8
  max_entries: 1000
  ttl_seconds: 3600

# Tool auto-selection
tools:
  enabled: false
  top_k: 3
  similarity_threshold: 0.2
  tools_db_path: "config/tools_db.json"
  fallback_to_empty: true

# Jailbreak protection
prompt_guard:
  enabled: false
  use_modernbert: true
  model_id: "models/jailbreak_classifier_modernbert-base_model"
  threshold: 0.7
  use_cpu: true

# vLLM endpoints - your backend models
vllm_endpoints:
  - name: "endpoint1"
    address: "your-server.com"  # Replace with your server
    port: 11434
    models:
      - "your-model"           # Replace with your model
    weight: 1

# Model configuration
model_config:
  "your-model":
    param_count: 7000000000    # Model parameters
    batch_size: 512.0
    context_size: 4096.0
    pii_policy:
      allow_by_default: true
      pii_types_allowed: ["EMAIL_ADDRESS", "PERSON"]
    preferred_endpoints: ["endpoint1"]

# Classification models
classifier:
  category_model:
    model_id: "models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.6
    use_cpu: true
  pii_model:
    model_id: "models/pii_classifier_modernbert-base_presidio_token_model"
    use_modernbert: true
    threshold: 0.7
    use_cpu: true

# Categories and routing rules
categories:
- name: math
  use_reasoning: true  # Enable reasoning for math
  model_scores:
  - model: your-model
    score: 1.0
- name: computer science
  use_reasoning: true  # Enable reasoning for code
  model_scores:
  - model: your-model
    score: 1.0
- name: other
  use_reasoning: false # No reasoning for general queries
  model_scores:
  - model: your-model
    score: 0.8

default_model: your-model
```

## Key Configuration Sections

### Backend Endpoints

Configure your LLM servers:

```yaml
vllm_endpoints:
  - name: "my_endpoint"
    address: "127.0.0.1"  # Your server IP
    port: 8000                # Your server port
    models:
      - "llama2-7b"          # Model name
    weight: 1                 # Load balancing weight
```

### Model Settings

Configure model-specific settings:

```yaml
model_config:
  "llama2-7b":
    param_count: 7000000000     # Model size in parameters
    batch_size: 512.0           # Batch size
    context_size: 4096.0        # Context window
    pii_policy:
      allow_by_default: true    # Allow PII by default
      pii_types_allowed: ["EMAIL_ADDRESS", "PERSON"]
    preferred_endpoints: ["my_endpoint"]
```

### Classification Models

Configure the BERT classification models:

```yaml
classifier:
  category_model:
    model_id: "models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.6            # Classification confidence threshold
    use_cpu: true             # Use CPU (no GPU required)
  pii_model:
    model_id: "models/pii_classifier_modernbert-base_presidio_token_model"
    threshold: 0.7            # PII detection threshold
    use_cpu: true
```

### Categories and Routing

Define how different query types are handled:

```yaml
categories:
- name: math
  use_reasoning: true              # Enable reasoning for math problems
  reasoning_description: "Mathematical problems require step-by-step reasoning"
  model_scores:
  - model: your-model
    score: 1.0                     # Preference score for this model

- name: computer science
  use_reasoning: true              # Enable reasoning for code
  model_scores:
  - model: your-model
    score: 1.0

- name: other
  use_reasoning: false             # No reasoning for general queries
  model_scores:
  - model: your-model
    score: 0.8

default_model: your-model          # Fallback model
```

### Security Features

Configure PII detection and jailbreak protection:

```yaml
# PII Detection
classifier:
  pii_model:
    threshold: 0.7                 # Higher = more strict PII detection

# Jailbreak Protection
prompt_guard:
  enabled: true                    # Enable jailbreak detection
  threshold: 0.7                   # Detection sensitivity
  use_cpu: true                    # Runs on CPU

# Model-level PII policies
model_config:
  "your-model":
    pii_policy:
      allow_by_default: true       # Allow most content
      pii_types_allowed: ["EMAIL_ADDRESS", "PERSON"]  # Specific allowed types
```

### Optional Features

Configure additional features:

```yaml
# Semantic Caching
semantic_cache:
  enabled: true                    # Enable semantic caching
  similarity_threshold: 0.8        # Cache hit threshold
  max_entries: 1000               # Maximum cache entries
  ttl_seconds: 3600               # Cache expiration time

# Tool Auto-Selection
tools:
  enabled: true                    # Enable automatic tool selection
  top_k: 3                        # Number of tools to select
  similarity_threshold: 0.2        # Tool relevance threshold
  tools_db_path: "config/tools_db.json"
  fallback_to_empty: true         # Return empty on failure

# BERT Model for Similarity
bert_model:
  model_id: sentence-transformers/all-MiniLM-L12-v2
  threshold: 0.6                  # Similarity threshold
  use_cpu: true                   # CPU-only inference
```

## Common Configuration Examples

### Enable All Security Features

```yaml
# Enable PII detection
classifier:
  pii_model:
    threshold: 0.8              # Strict PII detection

# Enable jailbreak protection
prompt_guard:
  enabled: true
  threshold: 0.7

# Configure model PII policies
model_config:
  "your-model":
    pii_policy:
      allow_by_default: false   # Block all PII by default
      pii_types_allowed: []     # No PII allowed
```

### Performance Optimization

```yaml
# Enable caching
semantic_cache:
  enabled: true
  similarity_threshold: 0.85    # Higher = more cache hits
  max_entries: 5000
  ttl_seconds: 7200            # 2 hour cache

# Enable tool selection
tools:
  enabled: true
  top_k: 5                     # Select more tools
  similarity_threshold: 0.1    # Lower = more tools selected
```

### Development Setup

```yaml
# Disable security for testing
prompt_guard:
  enabled: false

# Disable caching for consistent results
semantic_cache:
  enabled: false

# Lower classification thresholds
classifier:
  category_model:
    threshold: 0.3             # Lower = more specialized routing
```

## Configuration Validation

### Test Your Configuration

Validate your configuration before starting:

```bash
# Test configuration syntax
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"

# Test the router with your config
make build
make run-router
```

### Common Configuration Patterns

**Multiple Models:**
```yaml
vllm_endpoints:
  - name: "math_endpoint"
    address: "math-server.com"
    port: 8000
    models: ["math-model"]
    weight: 1
  - name: "general_endpoint"
    address: "general-server.com"
    port: 8000
    models: ["general-model"]
    weight: 1

categories:
- name: math
  model_scores:
  - model: math-model
    score: 1.0
- name: other
  model_scores:
  - model: general-model
    score: 1.0
```

**Load Balancing:**
```yaml
vllm_endpoints:
  - name: "endpoint1"
    address: "server1.com"
    port: 8000
    models: ["my-model"]
    weight: 2              # Higher weight = more traffic
  - name: "endpoint2"
    address: "server2.com"
    port: 8000
    models: ["my-model"]
    weight: 1
```

## Best Practices

### Security Configuration

For production environments:

```yaml
# Enable all security features
classifier:
  pii_model:
    threshold: 0.8              # Strict PII detection

prompt_guard:
  enabled: true                 # Enable jailbreak protection
  threshold: 0.7

model_config:
  "your-model":
    pii_policy:
      allow_by_default: false   # Block PII by default
```

### Performance Tuning

For high-traffic scenarios:

```yaml
# Enable caching
semantic_cache:
  enabled: true
  similarity_threshold: 0.85    # Higher = more cache hits
  max_entries: 10000
  ttl_seconds: 3600

# Optimize classification
classifier:
  category_model:
    threshold: 0.7              # Balance accuracy vs speed
```

### Development vs Production

**Development:**
```yaml
# Relaxed settings for testing
classifier:
  category_model:
    threshold: 0.3              # Lower threshold for testing
prompt_guard:
  enabled: false                # Disable for development
semantic_cache:
  enabled: false                # Disable for consistent results
```

**Production:**
```yaml
# Strict settings for production
classifier:
  category_model:
    threshold: 0.7              # Higher threshold for accuracy
prompt_guard:
  enabled: true                 # Enable security
semantic_cache:
  enabled: true                 # Enable for performance
```

## Troubleshooting

### Common Issues

**Invalid YAML syntax:**
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"
```

**Missing model files:**
```bash
# Check if models are downloaded
ls -la models/
# If missing, run: make download-models
```

**Endpoint connectivity:**
```bash
# Test your backend server
curl -f http://your-server:8000/health
```

**Configuration not taking effect:**
```bash
# Restart the router after config changes
make run-router
```

### Testing Configuration

```bash
# Test with different queries
make test-auto-prompt-reasoning      # Math query
make test-auto-prompt-no-reasoning   # General query
make test-pii                        # PII detection
make test-prompt-guard               # Jailbreak protection
```

## Next Steps

- **[Installation Guide](installation.md)** - Setup instructions
- **[Quick Start Guide](installation.md)** - Basic usage examples
- **[API Documentation](../api/router.md)** - Complete API reference

The configuration system is designed to be simple yet powerful. Start with the basic configuration and gradually enable advanced features as needed.
