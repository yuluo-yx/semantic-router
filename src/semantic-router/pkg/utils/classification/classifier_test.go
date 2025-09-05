package classification

import (
	"errors"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/config"
)

func TestClassifier(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Classifier Suite")
}

// MockModelInference implements ModelInference interface for testing
type MockModelInference struct {
	classifyTextResult       candle_binding.ClassResult
	classifyTextError        error
	classifyModernBertResult candle_binding.ClassResult
	classifyModernBertError  error
}

func (m *MockModelInference) ClassifyText(text string) (candle_binding.ClassResult, error) {
	return m.classifyTextResult, m.classifyTextError
}

func (m *MockModelInference) ClassifyModernBertText(text string) (candle_binding.ClassResult, error) {
	return m.classifyModernBertResult, m.classifyModernBertError
}

var _ = Describe("Classifier", func() {
	var (
		classifier *Classifier
		mockModel  *MockModelInference
	)

	BeforeEach(func() {
		mockModel = &MockModelInference{}
		cfg := &config.RouterConfig{}
		cfg.Classifier.CategoryModel.Threshold = 0.5 // Set threshold for testing

		classifier = &Classifier{
			modelInference: mockModel,
			Config:         cfg,
			CategoryMapping: &CategoryMapping{
				CategoryToIdx: map[string]int{"technology": 0, "sports": 1, "politics": 2},
				IdxToCategory: map[string]string{"0": "technology", "1": "sports", "2": "politics"},
			},
		}
	})

	Describe("ClassifyCategory", func() {
		Context("when classification succeeds with high confidence", func() {
			It("should return the correct category", func() {
				mockModel.classifyTextResult = candle_binding.ClassResult{
					Class:      2,
					Confidence: 0.95,
				}

				category, score, err := classifier.ClassifyCategory("This is about politics")

				Expect(err).To(BeNil())
				Expect(category).To(Equal("politics"))
				Expect(score).To(BeNumerically("~", 0.95, 0.001))
			})
		})

		Context("when classification has low confidence below threshold", func() {
			It("should return empty category", func() {
				mockModel.classifyTextResult = candle_binding.ClassResult{
					Class:      0,
					Confidence: 0.3,
				}

				category, score, err := classifier.ClassifyCategory("Ambiguous text")

				Expect(err).To(BeNil())
				Expect(category).To(Equal(""))
				Expect(score).To(BeNumerically("~", 0.3, 0.001))
			})
		})

		Context("when BERT model returns error", func() {
			It("should return unknown category with zero score", func() {
				mockModel.classifyTextError = errors.New("model inference failed")

				category, score, err := classifier.ClassifyCategory("Some text")

				Expect(err).ToNot(BeNil())
				Expect(category).To(Equal(""))
				Expect(score).To(BeNumerically("~", 0.0, 0.001))
			})
		})

		Context("when input is empty or invalid", func() {
			It("should handle empty text gracefully", func() {
				mockModel.classifyTextResult = candle_binding.ClassResult{
					Class:      0,
					Confidence: 0.8,
				}

				category, score, err := classifier.ClassifyCategory("")

				// Should still attempt classification
				Expect(err).To(BeNil())
				Expect(category).To(Equal("technology"))
				Expect(score).To(BeNumerically("~", 0.8, 0.001))
			})
		})
	})
})
