# Configuration Guide

This guide covers all configuration options available in the Semantic Router, from basic setup to advanced customization for production deployments.

## Configuration File Structure

The main configuration file is located at `config/config.yaml`. Here's the complete structure:

```yaml
# config/config.yaml
router:
  # Server configuration
  host: "0.0.0.0"
  port: 50051
  log_level: "info"  # debug, info, warn, error
  
  # Model paths and configuration
  models:
    category_classifier: "./models/category_classifier_modernbert-base_model"
    pii_detector: "./models/pii_classifier_modernbert-base_model"
    jailbreak_guard: "./models/jailbreak_classifier_modernbert-base_model"
    intent_classifier: "./models/intent_classifier_modernbert-base_model"
    
  # Backend model endpoints  
  endpoints:
    endpoint1:
      url: "http://192.168.12.90:11434"
      model_type: "math"
      model_name: "llama2-math-7b"
      cost_per_token: 0.002
      max_tokens: 4096
      timeout: 300
      health_check_path: "/health"
      
    endpoint2:
      url: "http://192.168.12.91:11434"
      model_type: "creative"
      model_name: "llama2-creative-13b"
      cost_per_token: 0.005
      max_tokens: 8192
      timeout: 600
      
    endpoint3:
      url: "http://192.168.12.92:11434"
      model_type: "code"
      model_name: "codellama-34b"
      cost_per_token: 0.008
      max_tokens: 4096
      timeout: 300
      
    general_endpoint:
      url: "http://192.168.12.93:11434"
      model_type: "general"
      model_name: "llama2-70b"
      cost_per_token: 0.015
      max_tokens: 4096
      timeout: 300
      
  # Classification configuration
  classification:
    confidence_threshold: 0.75
    fallback_model: "general"
    enable_ensemble: false
    ensemble_weights: [0.6, 0.4]  # If ensemble enabled
    
  # Security settings
  security:
    enable_pii_detection: true
    enable_jailbreak_guard: true
    pii_action: "block"  # block, mask, allow
    jailbreak_action: "block"  # block, flag, allow
    pii_confidence_threshold: 0.8
    jailbreak_confidence_threshold: 0.3  # Low threshold for safety
    
  # Semantic cache configuration
  cache:
    enabled: true
    cache_type: "memory"  # memory, redis
    similarity_threshold: 0.85
    ttl_seconds: 3600
    max_entries: 10000
    cleanup_interval: 300
    
  # Redis configuration (if cache_type: redis)
  redis:
    host: "localhost"
    port: 6379
    password: ""
    database: 0
    
  # Tools configuration
  tools:
    auto_selection: true
    max_tools: 5
    relevance_threshold: 0.6
    tools_database_path: "./config/tools_db.json"
    
  # Monitoring and metrics
  monitoring:
    enable_metrics: true
    metrics_port: 9090
    enable_tracing: false
    jaeger_endpoint: "http://localhost:14268/api/traces"
    
  # Performance tuning
  performance:
    max_concurrent_requests: 100
    request_timeout: 30
    classification_timeout: 5
    enable_batching: false
    batch_size: 10
    batch_timeout: 100  # milliseconds
```

## Detailed Configuration Options

### Server Configuration

```yaml
router:
  host: "0.0.0.0"        # Bind address (0.0.0.0 for all interfaces)
  port: 50051            # gRPC server port
  log_level: "info"      # Logging level: debug, info, warn, error
  max_message_size: 4194304  # 4MB max message size
```

### Model Configuration

#### Model Paths
```yaml
models:
  category_classifier: "./models/category_classifier_modernbert-base_model"
  pii_detector: "./models/pii_classifier_modernbert-base_model"
  jailbreak_guard: "./models/jailbreak_classifier_modernbert-base_model"
  intent_classifier: "./models/intent_classifier_modernbert-base_model"
  
  # Optional: Custom model configurations
  custom_models:
    legal_classifier: "./models/legal_classifier_model"
    medical_classifier: "./models/medical_classifier_model"
```

#### Endpoint Configuration

Each endpoint represents a backend LLM that can handle requests:

```yaml
endpoints:
  my_endpoint:
    url: "http://my-model-server:8080"     # Backend URL
    model_type: "specialized_domain"        # Category this model handles
    model_name: "my-custom-model-v1"       # Model identifier
    cost_per_token: 0.001                  # Cost in dollars per token
    max_tokens: 2048                       # Maximum tokens for this model
    timeout: 300                           # Request timeout in seconds
    health_check_path: "/health"           # Health check endpoint
    headers:                               # Custom headers
      Authorization: "Bearer token123"
      X-Custom-Header: "value"
    retry_count: 3                         # Number of retries on failure
    circuit_breaker:                       # Circuit breaker configuration
      failure_threshold: 5
      reset_timeout: 60
```

### Classification Settings

Fine-tune how the router makes routing decisions:

```yaml
classification:
  # Global confidence threshold for routing decisions
  confidence_threshold: 0.75
  
  # Fallback model when confidence is low
  fallback_model: "general"
  
  # Per-category confidence thresholds
  category_thresholds:
    mathematics: 0.85      # Require high confidence for math routing
    creative: 0.70         # Allow lower confidence for creative
    code: 0.80             # High confidence for code generation
    
  # Ensemble classification (multiple models voting)
  enable_ensemble: false
  ensemble_models: ["model1", "model2", "model3"]
  ensemble_weights: [0.5, 0.3, 0.2]
  
  # Advanced options
  enable_confidence_calibration: true
  calibration_temperature: 1.5
```

### Security Configuration

Configure PII detection and jailbreak protection:

```yaml
security:
  # PII Detection
  enable_pii_detection: true
  pii_action: "block"                    # block, mask, allow
  pii_confidence_threshold: 0.8
  pii_entity_types: ["PERSON", "EMAIL", "PHONE", "SSN", "LOCATION"]
  
  # Custom PII patterns (regex)
  custom_pii_patterns:
    credit_card: '\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    api_key: '\b[A-Za-z0-9]{32,}\b'
    
  # Jailbreak Protection  
  enable_jailbreak_guard: true
  jailbreak_action: "block"              # block, flag, allow
  jailbreak_confidence_threshold: 0.3    # Low threshold for safety
  
  # Additional security measures
  rate_limiting:
    enabled: true
    requests_per_minute: 60
    burst_size: 10
    
  ip_whitelist:
    enabled: false
    allowed_ips: ["192.168.1.0/24", "10.0.0.0/8"]
```

### Cache Configuration

Configure semantic caching for performance:

```yaml
cache:
  enabled: true
  cache_type: "memory"                   # memory, redis, hybrid
  
  # Similarity settings
  similarity_threshold: 0.85             # Cosine similarity threshold
  similarity_algorithm: "cosine"         # cosine, euclidean, dot_product
  
  # Memory cache settings
  max_entries: 10000
  ttl_seconds: 3600
  cleanup_interval: 300
  
  # Redis cache settings (if cache_type: redis or hybrid)
  redis:
    host: "localhost"
    port: 6379
    password: "mypassword"
    database: 0
    pool_size: 10
    connection_timeout: 5
    
  # Cache warming
  enable_cache_warming: false
  warm_up_queries: ["common query 1", "common query 2"]
  
  # Cache analytics
  enable_cache_metrics: true
  log_cache_performance: true
```

### Tools Configuration

Configure automatic tool selection:

```yaml
tools:
  auto_selection: true
  max_tools: 5
  relevance_threshold: 0.6
  
  # Tools database
  tools_database_path: "./config/tools_db.json"
  
  # Tool categories and weights
  tool_categories:
    calculation: 
      weight: 1.0
      max_tools: 3
    web_search:
      weight: 0.8
      max_tools: 2
    file_operations:
      weight: 0.9
      max_tools: 2
      
  # Custom tool scoring
  custom_scoring:
    enable_semantic_scoring: true
    enable_keyword_scoring: true
    enable_category_scoring: true
    weights: [0.4, 0.4, 0.2]  # semantic, keyword, category
```

## Environment-Specific Configurations

### Development Configuration

```yaml
# config/development.yaml
router:
  log_level: "debug"
  
  classification:
    confidence_threshold: 0.5  # Lower for testing
    
  security:
    enable_pii_detection: false  # Disable for testing
    enable_jailbreak_guard: false
    
  cache:
    ttl_seconds: 300  # Shorter cache for development
    
  monitoring:
    enable_metrics: true
    enable_tracing: true
```

### Production Configuration

```yaml
# config/production.yaml
router:
  log_level: "warn"
  
  classification:
    confidence_threshold: 0.8  # Higher for production
    enable_ensemble: true
    
  security:
    enable_pii_detection: true
    enable_jailbreak_guard: true
    pii_action: "block"
    jailbreak_action: "block"
    
    rate_limiting:
      enabled: true
      requests_per_minute: 1000
      
  cache:
    cache_type: "redis"
    ttl_seconds: 7200  # Longer cache
    
  performance:
    max_concurrent_requests: 1000
    enable_batching: true
    
  monitoring:
    enable_metrics: true
    enable_tracing: true
```

### Testing Configuration

```yaml
# config/testing.yaml
router:
  log_level: "debug"
  
  endpoints:
    mock_endpoint:
      url: "http://localhost:8080/mock"
      model_type: "general"
      
  classification:
    confidence_threshold: 0.1  # Very low for testing all paths
    
  security:
    enable_pii_detection: true
    pii_action: "flag"  # Don't block in tests
    enable_jailbreak_guard: true
    jailbreak_action: "flag"
    
  cache:
    enabled: false  # Disable cache for consistent test results
```

## Dynamic Configuration Updates

### Hot Reloading

Enable configuration hot reloading for production environments:

```yaml
router:
  config:
    enable_hot_reload: true
    reload_interval: 60  # Check for changes every 60 seconds
    reload_signal: "SIGHUP"  # Signal to trigger reload
```

### Configuration Management

Use environment variables for sensitive values:

```bash
# Environment variables
export ROUTER_REDIS_PASSWORD="secure_password"
export ROUTER_API_KEY="your_api_key"
export ROUTER_LOG_LEVEL="info"
```

```yaml
# In config file
router:
  redis:
    password: "${ROUTER_REDIS_PASSWORD}"
  api:
    key: "${ROUTER_API_KEY}"
  log_level: "${ROUTER_LOG_LEVEL:info}"  # Default to "info"
```

## Configuration Validation

### Built-in Validation

The router validates configuration on startup:

```bash
# Test configuration
./bin/router -config config/config.yaml -validate-only

# Check specific section
./bin/router -config config/config.yaml -validate-section=endpoints
```

### Configuration Schema

Use JSON Schema validation:

```bash
# Install schema validator
npm install -g ajv-cli

# Validate configuration
ajv validate -s config/schema.json -d config/config.yaml
```

## Advanced Configuration Patterns

### Multi-Tenant Configuration

```yaml
# config/multi-tenant.yaml
router:
  tenants:
    tenant_a:
      classification:
        confidence_threshold: 0.8
      endpoints: ["endpoint1", "endpoint2"]
      security:
        enable_pii_detection: true
        
    tenant_b:
      classification:
        confidence_threshold: 0.6
      endpoints: ["endpoint3", "endpoint4"]
      security:
        enable_pii_detection: false
```

### Load Balancing Configuration

```yaml
router:
  endpoints:
    math_cluster:
      type: "cluster"
      load_balancing: "round_robin"  # round_robin, weighted, least_connections
      members:
        - url: "http://math1:8080"
          weight: 1
        - url: "http://math2:8080"
          weight: 2
        - url: "http://math3:8080"  
          weight: 1
      health_check:
        enabled: true
        interval: 30
        timeout: 5
        healthy_threshold: 2
        unhealthy_threshold: 3
```

### A/B Testing Configuration

```yaml
router:
  experiments:
    model_comparison:
      enabled: true
      traffic_split: 0.1  # 10% to experimental model
      control_endpoint: "endpoint1"
      experimental_endpoint: "endpoint2"
      metrics_collection: true
      
  feature_flags:
    enable_new_classifier: false
    enable_advanced_caching: true
    enable_multi_model_routing: false
```

## Configuration Best Practices

### 1. Security Best Practices

```yaml
# Use strong security settings in production
security:
  enable_pii_detection: true
  pii_action: "block"
  enable_jailbreak_guard: true
  jailbreak_action: "block"
  
  # Enable rate limiting
  rate_limiting:
    enabled: true
    requests_per_minute: 100
    
  # Use IP whitelisting if applicable
  ip_whitelist:
    enabled: true
    allowed_ips: ["trusted_network/24"]
```

### 2. Performance Best Practices

```yaml
# Optimize for performance
performance:
  max_concurrent_requests: 500
  enable_batching: true
  batch_size: 20
  
cache:
  enabled: true
  cache_type: "redis"  # Use Redis for distributed caching
  max_entries: 50000
  ttl_seconds: 3600
  
classification:
  confidence_threshold: 0.75  # Balance accuracy and speed
```

### 3. Monitoring Best Practices

```yaml
# Comprehensive monitoring
monitoring:
  enable_metrics: true
  enable_tracing: true
  enable_logging: true
  
  # Detailed metrics collection
  detailed_metrics:
    classification_latency: true
    cache_performance: true
    security_events: true
    endpoint_health: true
```

## Troubleshooting Configuration

### Common Configuration Issues

1. **Invalid YAML syntax**
   ```bash
   # Validate YAML syntax
   python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"
   ```

2. **Missing model files**
   ```bash
   # Check model paths
   ls -la models/
   ```

3. **Unreachable endpoints**
   ```bash
   # Test endpoint connectivity
   curl -f http://your-endpoint:8080/health
   ```

4. **Port conflicts**
   ```bash
   # Check port usage
   lsof -i :50051
   ```

### Configuration Debugging

Enable debug logging for configuration issues:

```bash
# Run with verbose configuration logging
./bin/router -config config/config.yaml -log-level debug -config-debug
```

## Next Steps

- **[API Reference](../api/router.md)**: Detailed API documentation
- **[Architecture Guide](../architecture/system-architecture.md)**: Understand the system design and monitoring
- **[Installation Guide](installation.md)**: Deployment setup and requirements

For more advanced configuration options, refer to the specific component documentation or join our community discussions.
