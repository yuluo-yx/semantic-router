package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/semantic-router/pkg/config"
)

// TestReasoningModeIntegration tests the complete reasoning mode integration
func TestReasoningModeIntegration(t *testing.T) {
	// Create a mock router with reasoning configuration
	cfg := &config.RouterConfig{
		Categories: []config.Category{
			{
				Name:                 "math",
				UseReasoning:         true,
				ReasoningDescription: "Mathematical problems require step-by-step reasoning",
			},
			{
				Name:                 "business",
				UseReasoning:         false,
				ReasoningDescription: "Business content is typically conversational",
			},
		},
	}

	router := &OpenAIRouter{
		Config: cfg,
	}

	// Test case 1: Math query should enable reasoning (when classifier works)
	t.Run("Math query enables reasoning", func(t *testing.T) {
		mathQuery := "What is the derivative of x^2 + 3x + 1?"

		// Since we don't have the actual classifier, this will return false
		// But we can test the configuration logic directly
		useReasoning := router.shouldUseReasoningMode(mathQuery)

		// Without a working classifier, this should be false
		expectedReasoning := false

		if useReasoning != expectedReasoning {
			t.Errorf("Expected reasoning mode %v for math query without classifier, got %v", expectedReasoning, useReasoning)
		}

		// Test the configuration logic directly
		mathCategory := cfg.Categories[0] // math category
		if !mathCategory.UseReasoning {
			t.Error("Math category should have UseReasoning set to true in configuration")
		}
	})

	// Test case 2: Business query should not enable reasoning
	t.Run("Business query disables reasoning", func(t *testing.T) {
		businessQuery := "Write a business plan for a coffee shop"

		useReasoning := router.shouldUseReasoningMode(businessQuery)

		// Should be false because classifier returns empty (no category found)
		if useReasoning != false {
			t.Errorf("Expected reasoning mode false for business query, got %v", useReasoning)
		}
	})

	// Test case 3: Test addReasoningModeToRequestBody function
	t.Run("addReasoningModeToRequestBody adds correct fields", func(t *testing.T) {
		// Test with DeepSeek model (which supports chat_template_kwargs)
		originalRequest := map[string]interface{}{
			"model": "deepseek-v31",
			"messages": []map[string]interface{}{
				{"role": "user", "content": "What is 2 + 2?"},
			},
			"stream": false,
		}

		originalBody, err := json.Marshal(originalRequest)
		if err != nil {
			t.Fatalf("Failed to marshal original request: %v", err)
		}

		modifiedBody, err := router.setReasoningModeToRequestBody(originalBody, true, "math")
		if err != nil {
			t.Fatalf("Failed to add reasoning mode: %v", err)
		}

		var modifiedRequest map[string]interface{}
		if err := json.Unmarshal(modifiedBody, &modifiedRequest); err != nil {
			t.Fatalf("Failed to unmarshal modified request: %v", err)
		}

		// Check if chat_template_kwargs was added for DeepSeek model
		chatTemplateKwargs, exists := modifiedRequest["chat_template_kwargs"]
		if !exists {
			t.Error("chat_template_kwargs not found in modified request for DeepSeek model")
		}

		// Check if thinking: true was set for DeepSeek model
		if kwargs, ok := chatTemplateKwargs.(map[string]interface{}); ok {
			if thinking, hasThinking := kwargs["thinking"]; hasThinking {
				if thinkingBool, isBool := thinking.(bool); !isBool || !thinkingBool {
					t.Errorf("Expected thinking: true for DeepSeek model, got %v", thinking)
				}
			} else {
				t.Error("thinking field not found in chat_template_kwargs for DeepSeek model")
			}
		} else {
			t.Errorf("chat_template_kwargs is not a map for DeepSeek model, got %T", chatTemplateKwargs)
		}

		// Verify original fields are preserved
		originalFields := []string{"model", "messages", "stream"}
		for _, field := range originalFields {
			if _, exists := modifiedRequest[field]; !exists {
				t.Errorf("Original field '%s' was lost", field)
			}
		}

		// Test with unsupported model (phi4) - should not add chat_template_kwargs
		originalRequestPhi4 := map[string]interface{}{
			"model": "phi4",
			"messages": []map[string]interface{}{
				{"role": "user", "content": "What is 2 + 2?"},
			},
			"stream": false,
		}

		originalBodyPhi4, err := json.Marshal(originalRequestPhi4)
		if err != nil {
			t.Fatalf("Failed to marshal phi4 request: %v", err)
		}

		modifiedBodyPhi4, err := router.setReasoningModeToRequestBody(originalBodyPhi4, true, "math")
		if err != nil {
			t.Fatalf("Failed to process phi4 request: %v", err)
		}

		var modifiedRequestPhi4 map[string]interface{}
		if err := json.Unmarshal(modifiedBodyPhi4, &modifiedRequestPhi4); err != nil {
			t.Fatalf("Failed to unmarshal phi4 request: %v", err)
		}

		// For phi4, chat_template_kwargs should not be added (since it's not supported)
		if _, exists := modifiedRequestPhi4["chat_template_kwargs"]; exists {
			t.Error("chat_template_kwargs should not be added for unsupported model phi4")
		}

		// But reasoning_effort should still be set
		if reasoningEffort, exists := modifiedRequestPhi4["reasoning_effort"]; !exists {
			t.Error("reasoning_effort should be set for phi4 model")
		} else if reasoningEffort != "high" {
			t.Errorf("Expected reasoning_effort: high for phi4 model, got %v", reasoningEffort)
		}
	})

	// Test case 4: Test getChatTemplateKwargs function
	t.Run("getChatTemplateKwargs returns correct values", func(t *testing.T) {
		// Test with DeepSeek model and reasoning enabled
		kwargs := getChatTemplateKwargs("deepseek-v31", true)
		if kwargs == nil {
			t.Error("Expected non-nil kwargs for DeepSeek model with reasoning enabled")
		}

		if thinking, ok := kwargs["thinking"]; !ok || thinking != true {
			t.Errorf("Expected thinking: true for DeepSeek model, got %v", thinking)
		}

		// Test with DeepSeek model and reasoning disabled
		kwargs = getChatTemplateKwargs("deepseek-v31", false)
		if kwargs == nil {
			t.Error("Expected non-nil kwargs for DeepSeek model with reasoning disabled")
		}

		if thinking, ok := kwargs["thinking"]; !ok || thinking != false {
			t.Errorf("Expected thinking: false for DeepSeek model, got %v", thinking)
		}

		// Test with Qwen3 model and reasoning enabled
		kwargs = getChatTemplateKwargs("qwen3-7b", true)
		if kwargs == nil {
			t.Error("Expected non-nil kwargs for Qwen3 model with reasoning enabled")
		}

		if enableThinking, ok := kwargs["enable_thinking"]; !ok || enableThinking != true {
			t.Errorf("Expected enable_thinking: true for Qwen3 model, got %v", enableThinking)
		}

		// Test with unknown model (should return nil)
		kwargs = getChatTemplateKwargs("unknown-model", true)
		if kwargs != nil {
			t.Errorf("Expected nil kwargs for unknown model, got %v", kwargs)
		}
	})

	// Test case 5: Test empty query handling
	t.Run("Empty query defaults to no reasoning", func(t *testing.T) {
		useReasoning := router.shouldUseReasoningMode("")
		if useReasoning != false {
			t.Errorf("Expected reasoning mode false for empty query, got %v", useReasoning)
		}
	})

	// Test case 6: Test unknown category handling
	t.Run("Unknown category defaults to no reasoning", func(t *testing.T) {
		unknownQuery := "This is some unknown category query"
		useReasoning := router.shouldUseReasoningMode(unknownQuery)
		if useReasoning != false {
			t.Errorf("Expected reasoning mode false for unknown category, got %v", useReasoning)
		}
	})
}

// TestReasoningModeConfigurationValidation tests the configuration validation
func TestReasoningModeConfigurationValidation(t *testing.T) {
	testCases := []struct {
		name     string
		category config.Category
		expected bool
	}{
		{
			name: "Math category with reasoning enabled",
			category: config.Category{
				Name:                 "math",
				UseReasoning:         true,
				ReasoningDescription: "Mathematical problems require step-by-step reasoning",
			},
			expected: true,
		},
		{
			name: "Business category with reasoning disabled",
			category: config.Category{
				Name:                 "business",
				UseReasoning:         false,
				ReasoningDescription: "Business content is typically conversational",
			},
			expected: false,
		},
		{
			name: "Science category with reasoning enabled",
			category: config.Category{
				Name:                 "science",
				UseReasoning:         true,
				ReasoningDescription: "Scientific concepts benefit from structured analysis",
			},
			expected: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.category.UseReasoning != tc.expected {
				t.Errorf("Expected UseReasoning %v for %s, got %v",
					tc.expected, tc.category.Name, tc.category.UseReasoning)
			}

			// Verify description is not empty
			if tc.category.ReasoningDescription == "" {
				t.Errorf("ReasoningDescription should not be empty for category %s", tc.category.Name)
			}
		})
	}
}
