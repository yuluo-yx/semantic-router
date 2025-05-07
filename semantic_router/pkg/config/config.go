package config

import (
	"fmt"
	"os"
	"sync"

	"gopkg.in/yaml.v3"
)

// RouterConfig represents the main configuration for the LLM Router
type RouterConfig struct {
	// BERT model configuration for Candle BERT similarity comparison
	BertModel struct {
		ModelID   string  `yaml:"model_id"`
		Threshold float32 `yaml:"threshold"`
		UseCPU    bool    `yaml:"use_cpu"`
	} `yaml:"bert_model"`

	// Classifier configuration for text classification
	Classifier struct {
		ModelID             string  `yaml:"model_id"`
		Threshold           float32 `yaml:"threshold"`
		UseCPU              bool    `yaml:"use_cpu"`
		CategoryMappingPath string  `yaml:"category_mapping_path"`
	} `yaml:"classifier"`

	// Categories for routing queries
	Categories []Category `yaml:"categories"`

	// Default LLM model to use if no match is found
	DefaultModel string `yaml:"default_model"`

	// Semantic cache configuration
	SemanticCache SemanticCacheConfig `yaml:"semantic_cache"`
}

// SemanticCacheConfig represents configuration for the semantic cache
type SemanticCacheConfig struct {
	// Enable semantic caching
	Enabled bool `yaml:"enabled"`

	// Similarity threshold for cache hits (0.0-1.0)
	// If not specified, will use the BertModel.Threshold
	SimilarityThreshold *float32 `yaml:"similarity_threshold,omitempty"`

	// Maximum number of cache entries to keep
	MaxEntries int `yaml:"max_entries,omitempty"`

	// Time-to-live for cache entries in seconds (0 means no expiration)
	TTLSeconds int `yaml:"ttl_seconds,omitempty"`
}

// GetCacheSimilarityThreshold returns the effective threshold for the semantic cache
func (c *RouterConfig) GetCacheSimilarityThreshold() float32 {
	if c.SemanticCache.SimilarityThreshold != nil {
		return *c.SemanticCache.SimilarityThreshold
	}
	return c.BertModel.Threshold
}

// Category represents a category for routing queries
type Category struct {
	Name        string   `yaml:"name"`
	Description string   `yaml:"description,omitempty"`
	Models      []string `yaml:"models"` // Ranked list of LLM models
}

var (
	config     *RouterConfig
	configOnce sync.Once
	configErr  error
)

// LoadConfig loads the configuration from the specified YAML file
func LoadConfig(configPath string) (*RouterConfig, error) {
	configOnce.Do(func() {
		data, err := os.ReadFile(configPath)
		if err != nil {
			configErr = fmt.Errorf("failed to read config file: %w", err)
			return
		}

		config = &RouterConfig{}
		if err := yaml.Unmarshal(data, config); err != nil {
			configErr = fmt.Errorf("failed to parse config file: %w", err)
			return
		}
	})

	if configErr != nil {
		return nil, configErr
	}
	return config, nil
}

// GetConfig returns the current configuration
func GetConfig() *RouterConfig {
	return config
}

// GetCategoryDescriptions returns all category descriptions for similarity matching
func (c *RouterConfig) GetCategoryDescriptions() []string {
	var descriptions []string
	for _, category := range c.Categories {
		if category.Description != "" {
			descriptions = append(descriptions, category.Description)
		} else {
			// Use category name if no description is available
			descriptions = append(descriptions, category.Name)
		}
	}
	return descriptions
}

// GetModelForCategoryIndex returns the best LLM model name for the category at the given index
func (c *RouterConfig) GetModelForCategoryIndex(index int) string {
	if index < 0 || index >= len(c.Categories) {
		return c.DefaultModel
	}

	category := c.Categories[index]
	if len(category.Models) > 0 {
		return category.Models[0]
	}

	// Fall back to default model if category has no models
	return c.DefaultModel
}
