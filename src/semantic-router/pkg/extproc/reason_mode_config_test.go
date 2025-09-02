package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/semantic-router/pkg/config"
)

// TestReasoningModeConfiguration demonstrates how the reasoning mode works with the new config-based approach
func TestReasoningModeConfiguration(t *testing.T) {
	fmt.Println("=== Configuration-Based Reasoning Mode Test ===")

	// Create a mock configuration for testing
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
			{
				Name:                 "biology",
				UseReasoning:         true,
				ReasoningDescription: "Biological processes benefit from structured analysis",
			},
		},
	}

	fmt.Printf("Loaded configuration with %d categories\n\n", len(cfg.Categories))

	// Display reasoning configuration for each category
	fmt.Println("--- Reasoning Mode Configuration ---")
	for _, category := range cfg.Categories {
		reasoningStatus := "DISABLED"
		if category.UseReasoning {
			reasoningStatus = "ENABLED"
		}

		fmt.Printf("Category: %-15s | Reasoning: %-8s | %s\n",
			category.Name, reasoningStatus, category.ReasoningDescription)
	}

	// Test queries with expected categories
	testQueries := []struct {
		query    string
		category string
	}{
		{"What is the derivative of x^2 + 3x + 1?", "math"},
		{"Implement a binary search algorithm in Python", "computer science"},
		{"Explain the process of photosynthesis", "biology"},
		{"Write a business plan for a coffee shop", "business"},
		{"Tell me about World War II", "history"},
		{"What are Newton's laws of motion?", "physics"},
		{"How does chemical bonding work?", "chemistry"},
		{"Design a bridge structure", "engineering"},
	}

	fmt.Printf("\n--- Test Query Reasoning Decisions ---\n")
	for _, test := range testQueries {
		// Find the category configuration
		var useReasoning bool
		var reasoningDesc string
		var found bool

		for _, category := range cfg.Categories {
			if strings.EqualFold(category.Name, test.category) {
				useReasoning = category.UseReasoning
				reasoningDesc = category.ReasoningDescription
				found = true
				break
			}
		}

		if !found {
			fmt.Printf("Query: %s\n", test.query)
			fmt.Printf("  Expected Category: %s (NOT FOUND IN CONFIG)\n", test.category)
			fmt.Printf("  Reasoning: DISABLED (default)\n\n")
			continue
		}

		reasoningStatus := "DISABLED"
		if useReasoning {
			reasoningStatus = "ENABLED"
		}

		fmt.Printf("Query: %s\n", test.query)
		fmt.Printf("  Category: %s\n", test.category)
		fmt.Printf("  Reasoning: %s - %s\n", reasoningStatus, reasoningDesc)

		// // Generate example request body
		// messages := []map[string]string{
		// 	{"role": "system", "content": "You are an AI assistant"},
		// 	{"role": "user", "content": test.query},
		// }

		// requestBody := buildRequestBody("deepseek-v31", messages, useReasoning, true)

		// Show key differences in request
		if useReasoning {
			fmt.Printf("  Request includes: chat_template_kwargs: {thinking: true}\n")
		} else {
			fmt.Printf("  Request: Standard mode (no reasoning)\n")
		}
		fmt.Println()
	}

	// Show example configuration section
	fmt.Println("--- Example Config.yaml Section ---")
	fmt.Print(`
categories:
- name: math
  use_reasoning: true
  reasoning_description: "Mathematical problems require step-by-step reasoning"
  model_scores:
  - model: phi4
    score: 1.0

- name: business
  use_reasoning: false
  reasoning_description: "Business content is typically conversational"
  model_scores:
  - model: phi4
    score: 0.8
`)
}

// GetReasoningConfigurationSummary returns a summary of the reasoning configuration
func GetReasoningConfigurationSummary(cfg *config.RouterConfig) map[string]interface{} {
	summary := make(map[string]interface{})

	reasoningEnabled := 0
	reasoningDisabled := 0

	categoriesWithReasoning := []string{}
	categoriesWithoutReasoning := []string{}

	for _, category := range cfg.Categories {
		if category.UseReasoning {
			reasoningEnabled++
			categoriesWithReasoning = append(categoriesWithReasoning, category.Name)
		} else {
			reasoningDisabled++
			categoriesWithoutReasoning = append(categoriesWithoutReasoning, category.Name)
		}
	}

	summary["total_categories"] = len(cfg.Categories)
	summary["reasoning_enabled_count"] = reasoningEnabled
	summary["reasoning_disabled_count"] = reasoningDisabled
	summary["categories_with_reasoning"] = categoriesWithReasoning
	summary["categories_without_reasoning"] = categoriesWithoutReasoning

	return summary
}

// DemonstrateConfigurationUsage shows how to use the configuration-based reasoning
func DemonstrateConfigurationUsage() {
	fmt.Println("=== Configuration Usage Example ===")
	fmt.Println()

	fmt.Println("1. Configure reasoning in config.yaml:")
	fmt.Print(`
categories:
- name: math
  use_reasoning: true
  reasoning_description: "Mathematical problems require step-by-step reasoning"

- name: creative_writing
  use_reasoning: false
  reasoning_description: "Creative content flows better without structured reasoning"
`)

	fmt.Println("\n2. Use in Go code:")
	fmt.Print(`
// The reasoning decision now comes from configuration
useReasoning := router.shouldUseReasoningMode(query)

// Build request with appropriate reasoning mode
requestBody := buildRequestBody(model, messages, useReasoning, stream)
`)

	fmt.Println("\n3. Benefits of configuration-based approach:")
	fmt.Println("   - Easy to modify reasoning settings without code changes")
	fmt.Println("   - Consistent with existing category configuration")
	fmt.Println("   - Supports different reasoning strategies per category")
	fmt.Println("   - Can be updated at runtime by reloading configuration")
	fmt.Println("   - Documentation is embedded in the config file")
}

// TestAddReasoningModeToRequestBody tests the addReasoningModeToRequestBody function
func TestAddReasoningModeToRequestBody(t *testing.T) {
	fmt.Println("=== Testing addReasoningModeToRequestBody Function ===")

	// Create a mock router with minimal config
	router := &OpenAIRouter{}

	// Test case 1: Basic request body
	originalRequest := map[string]interface{}{
		"model": "phi4",
		"messages": []map[string]interface{}{
			{"role": "user", "content": "What is 2 + 2?"},
		},
		"stream": false,
	}

	originalBody, err := json.Marshal(originalRequest)
	if err != nil {
		fmt.Printf("Error marshaling original request: %v\n", err)
		return
	}

	fmt.Printf("Original request body:\n%s\n\n", string(originalBody))

	// Add reasoning mode
	modifiedBody, err := router.setReasoningModeToRequestBody(originalBody, true, "math")
	if err != nil {
		fmt.Printf("Error adding reasoning mode: %v\n", err)
		return
	}

	fmt.Printf("Modified request body with reasoning mode:\n%s\n\n", string(modifiedBody))

	// Verify the modification
	var modifiedRequest map[string]interface{}
	if err := json.Unmarshal(modifiedBody, &modifiedRequest); err != nil {
		fmt.Printf("Error unmarshaling modified request: %v\n", err)
		return
	}

	// Check if chat_template_kwargs was added
	if chatTemplateKwargs, exists := modifiedRequest["chat_template_kwargs"]; exists {
		if kwargs, ok := chatTemplateKwargs.(map[string]interface{}); ok {
			if thinking, hasThinking := kwargs["thinking"]; hasThinking {
				if thinkingBool, isBool := thinking.(bool); isBool && thinkingBool {
					fmt.Println("✅ SUCCESS: chat_template_kwargs with thinking: true was correctly added")
				} else {
					fmt.Printf("❌ ERROR: thinking value is not true, got: %v\n", thinking)
				}
			} else {
				fmt.Println("❌ ERROR: thinking field not found in chat_template_kwargs")
			}
		} else {
			fmt.Printf("❌ ERROR: chat_template_kwargs is not a map, got: %T\n", chatTemplateKwargs)
		}
	} else {
		fmt.Println("❌ ERROR: chat_template_kwargs not found in modified request")
	}

	// Test case 2: Request with existing fields
	fmt.Println("\n--- Test Case 2: Request with existing fields ---")
	complexRequest := map[string]interface{}{
		"model": "phi4",
		"messages": []map[string]interface{}{
			{"role": "system", "content": "You are a helpful assistant"},
			{"role": "user", "content": "Solve x^2 + 5x + 6 = 0"},
		},
		"stream":      true,
		"temperature": 0.7,
		"max_tokens":  1000,
	}

	complexBody, err := json.Marshal(complexRequest)
	if err != nil {
		fmt.Printf("Error marshaling complex request: %v\n", err)
		return
	}

	modifiedComplexBody, err := router.setReasoningModeToRequestBody(complexBody, true, "chemistry")
	if err != nil {
		fmt.Printf("Error adding reasoning mode to complex request: %v\n", err)
		return
	}

	var modifiedComplexRequest map[string]interface{}
	if err := json.Unmarshal(modifiedComplexBody, &modifiedComplexRequest); err != nil {
		fmt.Printf("Error unmarshaling modified complex request: %v\n", err)
		return
	}

	// Verify all original fields are preserved
	originalFields := []string{"model", "messages", "stream", "temperature", "max_tokens"}
	allFieldsPreserved := true
	for _, field := range originalFields {
		if _, exists := modifiedComplexRequest[field]; !exists {
			fmt.Printf("❌ ERROR: Original field '%s' was lost\n", field)
			allFieldsPreserved = false
		}
	}

	if allFieldsPreserved {
		fmt.Println("✅ SUCCESS: All original fields preserved")
	}

	// Verify chat_template_kwargs was added
	if _, exists := modifiedComplexRequest["chat_template_kwargs"]; exists {
		fmt.Println("✅ SUCCESS: chat_template_kwargs added to complex request")
		fmt.Printf("Final modified request:\n%s\n", string(modifiedComplexBody))
	} else {
		fmt.Println("❌ ERROR: chat_template_kwargs not added to complex request")
	}
}
