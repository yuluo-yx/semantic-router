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

type MockCategoryInference struct {
	classifyResult candle_binding.ClassResult
	classifyError  error
}

func (m *MockCategoryInference) Classify(text string) (candle_binding.ClassResult, error) {
	return m.classifyResult, m.classifyError
}

var _ = Describe("ClassifyCategory", func() {
	var (
		classifier        *Classifier
		mockCategoryModel *MockCategoryInference
	)

	BeforeEach(func() {
		mockCategoryModel = &MockCategoryInference{}
		cfg := &config.RouterConfig{}
		cfg.Classifier.CategoryModel.Threshold = 0.5 // Set threshold for testing

		classifier = &Classifier{
			categoryInference: mockCategoryModel,
			Config:            cfg,
			CategoryMapping: &CategoryMapping{
				CategoryToIdx: map[string]int{"technology": 0, "sports": 1, "politics": 2},
				IdxToCategory: map[string]string{"0": "technology", "1": "sports", "2": "politics"},
			},
		}
	})

	Context("when classification succeeds with high confidence", func() {
		It("should return the correct category", func() {
			mockCategoryModel.classifyResult = candle_binding.ClassResult{
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
			mockCategoryModel.classifyResult = candle_binding.ClassResult{
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
		It("should return empty category with zero score", func() {
			mockCategoryModel.classifyError = errors.New("model inference failed")

			category, score, err := classifier.ClassifyCategory("Some text")

			Expect(err).ToNot(BeNil())
			Expect(category).To(Equal(""))
			Expect(score).To(BeNumerically("~", 0.0, 0.001))
		})
	})

	Context("when input is empty or invalid", func() {
		It("should handle empty text gracefully", func() {
			mockCategoryModel.classifyResult = candle_binding.ClassResult{
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

	Context("when category mapping is invalid", func() {
		It("should handle invalid category mapping gracefully", func() {
			mockCategoryModel.classifyResult = candle_binding.ClassResult{
				Class:      9,
				Confidence: 0.8,
			}

			category, score, err := classifier.ClassifyCategory("Some text")

			Expect(err).To(BeNil())
			Expect(category).To(Equal(""))
			Expect(score).To(BeNumerically("~", 0.8, 0.001))
		})
	})
})
