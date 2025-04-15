package candle_binding

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

/*
#cgo LDFLAGS: -L${SRCDIR}/target/release -lcandle_semantic_router -ldl -lm
#include <stdlib.h>
#include <stdbool.h>

extern bool init_similarity_model(const char* model_id, bool use_cpu);
extern float calculate_similarity(const char* text1, const char* text2);

// Similarity result structure
typedef struct {
    int index;
    float score;
} SimilarityResult;

extern SimilarityResult find_most_similar(const char* query, const char** candidates, int num_candidates);
extern void free_cstring(char* s);
*/
import "C"

var (
	initOnce sync.Once
	initErr  error
)

// SimResult represents the result of a similarity search
type SimResult struct {
	Index int     // Index of the most similar text
	Score float32 // Similarity score
}

// InitModel initializes the BERT model with the specified model path
func InitModel(modelPath string, useCPU bool) error {
	var err error
	initOnce.Do(func() {
		if modelPath == "" {
			modelPath = "sentence-transformers/all-MiniLM-L6-v2"
		}

		fmt.Println("Initializing BERT similarity model:", modelPath)

		// Initialize BERT directly using CGO
		cModelID := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelID))

		success := C.init_similarity_model(cModelID, C.bool(useCPU))
		if !bool(success) {
			err = fmt.Errorf("failed to initialize BERT similarity model")
		}
	})
	return err
}

// CalculateSimilarity calculates the similarity between two texts
func CalculateSimilarity(text1, text2 string) float32 {
	cText1 := C.CString(text1)
	defer C.free(unsafe.Pointer(cText1))

	cText2 := C.CString(text2)
	defer C.free(unsafe.Pointer(cText2))

	result := C.calculate_similarity(cText1, cText2)
	return float32(result)
}

// FindMostSimilar finds the most similar text from a list of candidates
func FindMostSimilar(query string, candidates []string) SimResult {
	if len(candidates) == 0 {
		return SimResult{Index: -1, Score: -1.0}
	}

	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))

	// Convert the candidates to C strings
	cCandidates := make([]*C.char, len(candidates))
	for i, candidate := range candidates {
		cCandidates[i] = C.CString(candidate)
		defer C.free(unsafe.Pointer(cCandidates[i]))
	}

	// Create a C array of C strings
	cCandidatesPtr := (**C.char)(unsafe.Pointer(&cCandidates[0]))

	result := C.find_most_similar(cQuery, cCandidatesPtr, C.int(len(candidates)))

	return SimResult{
		Index: int(result.index),
		Score: float32(result.score),
	}
}

// SetMemoryCleanupHandler sets up a finalizer to clean up memory when the Go GC runs
func SetMemoryCleanupHandler() {
	runtime.GC()
}
