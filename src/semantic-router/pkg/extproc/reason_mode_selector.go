package extproc

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"
)

// shouldUseReasoningMode determines if reasoning mode should be enabled based on the query category
func (r *OpenAIRouter) shouldUseReasoningMode(query string) bool {
	// Get the category for this query using the existing classification system
	categoryName := r.findCategoryForClassification(query)

	// If no category was determined (empty string), default to no reasoning
	if categoryName == "" {
		log.Printf("No category determined for query, defaulting to no reasoning mode")
		return false
	}

	// Normalize category name for consistent lookup
	normalizedCategory := strings.ToLower(strings.TrimSpace(categoryName))

	// Look up the category in the configuration
	for _, category := range r.Config.Categories {
		if strings.EqualFold(category.Name, normalizedCategory) {
			reasoningStatus := "DISABLED"
			if category.UseReasoning {
				reasoningStatus = "ENABLED"
			}
			log.Printf("Reasoning mode decision: Category '%s' → %s",
				categoryName, reasoningStatus)
			return category.UseReasoning
		}
	}

	// If category not found in config, default to no reasoning
	log.Printf("Category '%s' not found in configuration, defaulting to no reasoning mode", categoryName)
	return false
}

// getChatTemplateKwargs returns the appropriate chat template kwargs based on reasoning mode and streaming
func getChatTemplateKwargs(useReasoning bool) map[string]interface{} {
	if useReasoning {
		return map[string]interface{}{
			"thinking": useReasoning,
		}
	}
	return nil
}

// setReasoningModeToRequestBody adds chat_template_kwargs to the JSON request body
func (r *OpenAIRouter) setReasoningModeToRequestBody(requestBody []byte, enabled bool) ([]byte, error) {
	// Parse the JSON request body
	var requestMap map[string]interface{}
	if err := json.Unmarshal(requestBody, &requestMap); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}

	// Add chat_template_kwargs for reasoning mode
	requestMap["chat_template_kwargs"] = getChatTemplateKwargs(enabled)
	// Also set Reasoning-Effort in openai request
	// This is a hack to get the reasoning mode for openai/gpt-oss-20b to work
	originalReasoningEffort, ok := requestMap["reasoning_effort"]
	if !ok {
		// This seems to be the default for openai/gpt-oss models
		originalReasoningEffort = "low"
	}
	if enabled {
		// TODO: make this configurable
		requestMap["reasoning_effort"] = "high"
	} else {
		requestMap["reasoning_effort"] = originalReasoningEffort
	}

	// Get the model name for logging
	model := "unknown"
	if modelValue, ok := requestMap["model"]; ok {
		if modelStr, ok := modelValue.(string); ok {
			model = modelStr
		}
	}

	log.Printf("Original reasoning effort: %s", originalReasoningEffort)
	log.Printf("Added reasoning mode (thinking: %v) and reasoning effort (%s) to request for model: %s", enabled, requestMap["reasoning_effort"], model)

	// Serialize back to JSON
	modifiedBody, err := json.Marshal(requestMap)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize modified request: %w", err)
	}

	return modifiedBody, nil
}

// logReasoningConfiguration logs the reasoning mode configuration for all categories during startup
func (r *OpenAIRouter) logReasoningConfiguration() {
	if len(r.Config.Categories) == 0 {
		log.Printf("No categories configured for reasoning mode")
		return
	}

	reasoningEnabled := []string{}
	reasoningDisabled := []string{}

	for _, category := range r.Config.Categories {
		if category.UseReasoning {
			reasoningEnabled = append(reasoningEnabled, category.Name)
		} else {
			reasoningDisabled = append(reasoningDisabled, category.Name)
		}
	}

	log.Printf("Reasoning configuration - Total categories: %d", len(r.Config.Categories))

	if len(reasoningEnabled) > 0 {
		log.Printf("Reasoning ENABLED for categories (%d): %v", len(reasoningEnabled), reasoningEnabled)
	}

	if len(reasoningDisabled) > 0 {
		log.Printf("Reasoning DISABLED for categories (%d): %v", len(reasoningDisabled), reasoningDisabled)
	}
}

// ClassifyAndDetermineReasoningMode performs category classification and returns both the best model and reasoning mode setting
func (r *OpenAIRouter) ClassifyAndDetermineReasoningMode(query string) (string, bool) {
	// Get the best model using existing logic
	bestModel := r.classifyAndSelectBestModel(query)

	// Determine if reasoning mode should be used
	useReasoning := r.shouldUseReasoningMode(query)

	reasoningStatus := "disabled"
	if useReasoning {
		reasoningStatus = "enabled"
	}
	log.Printf("Model selection complete: model=%s, reasoning=%s", bestModel, reasoningStatus)

	return bestModel, useReasoning
}

// LogReasoningConfigurationSummary provides a compact summary of reasoning configuration
func (r *OpenAIRouter) LogReasoningConfigurationSummary() {
	if len(r.Config.Categories) == 0 {
		return
	}

	enabledCount := 0
	for _, category := range r.Config.Categories {
		if category.UseReasoning {
			enabledCount++
		}
	}

	log.Printf("Reasoning mode summary: %d/%d categories have reasoning enabled", enabledCount, len(r.Config.Categories))
}
