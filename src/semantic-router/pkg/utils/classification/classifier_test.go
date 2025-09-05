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

	Context("when classification confidence is below threshold", func() {
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

	Context("when model inference fails", func() {
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

	Context("when class index is not found in category mapping", func() {
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

type MockJailbreakInference struct {
	classifyResult candle_binding.ClassResult
	classifyError  error
}

func (m *MockJailbreakInference) Classify(text string) (candle_binding.ClassResult, error) {
	return m.classifyResult, m.classifyError
}

var _ = Describe("CheckForJailbreak", func() {
	var (
		classifier         *Classifier
		mockJailbreakModel *MockJailbreakInference
	)

	BeforeEach(func() {
		mockJailbreakModel = &MockJailbreakInference{}
		cfg := &config.RouterConfig{}
		cfg.PromptGuard.Enabled = true
		cfg.PromptGuard.ModelID = "test-model"
		cfg.PromptGuard.JailbreakMappingPath = "test-mapping"
		cfg.PromptGuard.Threshold = 0.7

		classifier = &Classifier{
			jailbreakInference: mockJailbreakModel,
			Config:             cfg,
			JailbreakMapping: &JailbreakMapping{
				LabelToIdx: map[string]int{"jailbreak": 0, "benign": 1},
				IdxToLabel: map[string]string{"0": "jailbreak", "1": "benign"},
			},
			JailbreakInitialized: true,
		}
	})

	Context("when jailbreak is detected with high confidence", func() {
		It("should return true with jailbreak type", func() {
			mockJailbreakModel.classifyResult = candle_binding.ClassResult{
				Class:      0,
				Confidence: 0.9,
			}

			isJailbreak, jailbreakType, confidence, err := classifier.CheckForJailbreak("This is a jailbreak attempt")

			Expect(err).To(BeNil())
			Expect(isJailbreak).To(BeTrue())
			Expect(jailbreakType).To(Equal("jailbreak"))
			Expect(confidence).To(BeNumerically("~", 0.9, 0.001))
		})
	})

	Context("when text is benign with high confidence", func() {
		It("should return false with benign type", func() {
			mockJailbreakModel.classifyResult = candle_binding.ClassResult{
				Class:      1,
				Confidence: 0.9,
			}

			isJailbreak, jailbreakType, confidence, err := classifier.CheckForJailbreak("This is a normal question")

			Expect(err).To(BeNil())
			Expect(isJailbreak).To(BeFalse())
			Expect(jailbreakType).To(Equal("benign"))
			Expect(confidence).To(BeNumerically("~", 0.9, 0.001))
		})
	})

	Context("when jailbreak confidence is below threshold", func() {
		It("should return false even if classified as jailbreak", func() {
			mockJailbreakModel.classifyResult = candle_binding.ClassResult{
				Class:      0,
				Confidence: 0.5,
			}

			isJailbreak, jailbreakType, confidence, err := classifier.CheckForJailbreak("Ambiguous text")

			Expect(err).To(BeNil())
			Expect(isJailbreak).To(BeFalse())
			Expect(jailbreakType).To(Equal("jailbreak"))
			Expect(confidence).To(BeNumerically("~", 0.5, 0.001))
		})
	})

	Context("when model inference fails", func() {
		It("should return error", func() {
			mockJailbreakModel.classifyError = errors.New("model inference failed")

			isJailbreak, jailbreakType, confidence, err := classifier.CheckForJailbreak("Some text")

			Expect(err).ToNot(BeNil())
			Expect(err.Error()).To(ContainSubstring("jailbreak classification failed"))
			Expect(isJailbreak).To(BeFalse())
			Expect(jailbreakType).To(Equal(""))
			Expect(confidence).To(BeNumerically("~", 0.0, 0.001))
		})
	})

	Context("when class index is not found in jailbreak mapping", func() {
		It("should return error for unknown class", func() {
			mockJailbreakModel.classifyResult = candle_binding.ClassResult{
				Class:      9,
				Confidence: 0.9,
			}

			isJailbreak, jailbreakType, confidence, err := classifier.CheckForJailbreak("Some text")

			Expect(err).ToNot(BeNil())
			Expect(isJailbreak).To(BeFalse())
			Expect(jailbreakType).To(Equal(""))
			Expect(confidence).To(BeNumerically("~", 0.0, 0.001))
		})
	})
})
