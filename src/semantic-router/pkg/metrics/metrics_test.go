package metrics

import (
	"testing"
	"time"
)

// TestBatchClassificationMetrics tests the batch classification metrics recording
func TestBatchClassificationMetrics(t *testing.T) {
	tests := []struct {
		name           string
		processingType string
		batchSize      int
		duration       float64
		errorType      string
		expectError    bool
	}{
		{
			name:           "Sequential processing metrics",
			processingType: "sequential",
			batchSize:      3,
			duration:       0.5,
			errorType:      "",
			expectError:    false,
		},
		{
			name:           "Concurrent processing metrics",
			processingType: "concurrent",
			batchSize:      10,
			duration:       1.2,
			errorType:      "",
			expectError:    false,
		},
		{
			name:           "Error case metrics",
			processingType: "sequential",
			batchSize:      5,
			duration:       0.3,
			errorType:      "classification_failed",
			expectError:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Record metrics
			RecordBatchClassificationRequest(tt.processingType)
			RecordBatchSizeDistribution(tt.processingType, tt.batchSize)
			RecordBatchClassificationDuration(tt.processingType, tt.batchSize, tt.duration)
			RecordBatchClassificationTexts(tt.processingType, tt.batchSize)

			if tt.expectError {
				RecordBatchClassificationError(tt.processingType, tt.errorType)
			}

			// Test passes if no panic occurs during metric recording
			// In a real production environment, you would verify the actual metric values
			// using prometheus test utilities or by checking the metric registry
		})
	}
}

// TestGetBatchSizeRange tests the batch size range helper function
func TestGetBatchSizeRange(t *testing.T) {
	tests := []struct {
		name     string
		size     int
		expected string
	}{
		{"Single text", 1, "1"},
		{"Small batch", 3, "2-5"},
		{"Medium batch", 8, "6-10"},
		{"Large batch", 15, "11-20"},
		{"Very large batch", 35, "21-50"},
		{"Maximum batch", 100, "50+"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetBatchSizeRange(tt.size)
			if result != tt.expected {
				t.Errorf("GetBatchSizeRange(%d) = %s, want %s", tt.size, result, tt.expected)
			}
		})
	}
}

// TestConcurrentGoroutineTracking tests goroutine tracking functionality
func TestConcurrentGoroutineTracking(t *testing.T) {
	batchID := "test_batch_123"

	// Simulate goroutine start
	ConcurrentGoroutines.WithLabelValues(batchID).Inc()

	// Simulate some work
	time.Sleep(10 * time.Millisecond)

	// Simulate goroutine end
	ConcurrentGoroutines.WithLabelValues(batchID).Dec()

	// Test passes if no panic occurs during goroutine tracking
	// In production, you would verify the gauge values
}

// BenchmarkBatchClassificationMetrics benchmarks the performance impact of metrics recording
func BenchmarkBatchClassificationMetrics(b *testing.B) {
	processingType := "concurrent"
	batchSize := 10
	duration := 0.5

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		RecordBatchClassificationRequest(processingType)
		RecordBatchSizeDistribution(processingType, batchSize)
		RecordBatchClassificationDuration(processingType, batchSize, duration)
		RecordBatchClassificationTexts(processingType, batchSize)
	}
}

// TestMetricsIntegration tests the integration of all batch classification metrics
func TestMetricsIntegration(t *testing.T) {
	// Simulate a complete batch processing scenario
	processingType := "concurrent"
	batchSize := 8
	batchID := "integration_test_batch"

	// Start of batch processing
	RecordBatchClassificationRequest(processingType)
	RecordBatchSizeDistribution(processingType, batchSize)

	// Simulate concurrent goroutines
	for i := 0; i < batchSize; i++ {
		ConcurrentGoroutines.WithLabelValues(batchID).Inc()
	}

	// Simulate processing completion
	duration := 1.5
	RecordBatchClassificationDuration(processingType, batchSize, duration)
	RecordBatchClassificationTexts(processingType, batchSize)

	// Simulate goroutines completion
	for i := 0; i < batchSize; i++ {
		ConcurrentGoroutines.WithLabelValues(batchID).Dec()
	}

	// Test passes if no panic occurs during the complete workflow
}
