#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

std::string decodeJsonEscapes(const std::string &input);

struct MarianConfig {
    int decoderStartTokenId;
    int eosTokenId;
    int padTokenId;
    int maxLength;
    int vocabSize;
};

class MemoryMappedWeights {
  public:
    MemoryMappedWeights(const std::string &path);
    ~MemoryMappedWeights();

    MemoryMappedWeights(const MemoryMappedWeights &) = delete;
    MemoryMappedWeights &operator=(const MemoryMappedWeights &) = delete;

    MemoryMappedWeights(MemoryMappedWeights &&other) noexcept;

    const float *data() const;
    size_t size() const;
    bool isValid() const;

    inline const float &operator[](size_t index) const;
    inline void prefetchRange(size_t startIdx, size_t count) const;
    inline const float *getAlignedPtr(size_t offset = 0) const;

  private:
    int fd_;
    float *data_;
    size_t size_;
};

void computeLogitsOptimized(const std::vector<float> &hiddenState,
                            const MemoryMappedWeights &weights,
                            const MemoryMappedWeights &bias,
                            std::vector<float> &logits,
                            int vocabSize,
                            int hiddenSize);

void computeBatchedLogitsOptimized(const std::vector<std::vector<float>> &hiddenStates,
                                   const MemoryMappedWeights &weights,
                                   const MemoryMappedWeights &bias,
                                   std::vector<std::vector<float>> &batchLogits,
                                   int vocabSize,
                                   int hiddenSize);

#ifdef USE_SIMD
void computeLogitsSimd(const std::vector<float> &hiddenState,
                       const MemoryMappedWeights &weights,
                       const MemoryMappedWeights &bias,
                       std::vector<float> &logits,
                       int vocabSize,
                       int hiddenSize);

void computeBatchedLogitsSimd(const std::vector<std::vector<float>> &hiddenStates,
                              const MemoryMappedWeights &weights,
                              const MemoryMappedWeights &bias,
                              std::vector<std::vector<float>> &batchLogits,
                              int vocabSize,
                              int hiddenSize);

void computeBatchedLogitsSimdUltra(const std::vector<std::vector<float>> &hiddenStates,
                                   const MemoryMappedWeights &weights,
                                   const MemoryMappedWeights &bias,
                                   std::vector<std::vector<float>> &batchLogits,
                                   int vocabSize,
                                   int hiddenSize);
#endif

class TranslationModel {
  public:
    explicit TranslationModel(const std::string &modelDir);
    
    std::vector<std::string> translate(const std::vector<std::string> &sentences);
    std::string translateSingle(const std::string &inputSentence);

  private:
    struct BatchedInputs {
        std::vector<std::vector<int64_t>> inputIds;
        std::vector<std::vector<int64_t>> attentionMasks;
        std::vector<size_t> originalLengths;
        std::vector<std::string> sentences;
        size_t batchSize;
        size_t maxSequenceLength;
    };

    struct BatchedEncoderOutput {
        std::vector<float> encoderOutputs;
        std::vector<int64_t> shape;
        size_t batchSize;
        size_t sequenceLength;
        size_t hiddenSize;
    };

    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> idToToken_;
    MarianConfig config_;
    std::unique_ptr<MemoryMappedWeights> lmWeights_;
    std::unique_ptr<MemoryMappedWeights> lmBias_;

    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> encoderSession_;
    std::unique_ptr<Ort::Session> decoderSession_;

    sentencepiece::SentencePieceProcessor spProcessor_;
    std::string modelDir_;

    static constexpr int HIDDEN_SIZE = 512;

    std::unordered_map<std::string, int> loadVocab(const std::string &vocabPath);
    MarianConfig loadConfig(const std::string &configPath);
    void buildReverseVocab();

    std::vector<int> tokenize(const std::string &sentence);
    std::string detokenize(const std::vector<int> &ids);
    
    BatchedInputs createBatch(const std::vector<std::string> &sentences);
    BatchedEncoderOutput runBatchedEncoder(const BatchedInputs &batch);
    std::vector<std::vector<int>> runBatchedDecoder(const BatchedEncoderOutput &encoderOutput, 
                                                   const BatchedInputs &batchInputs);

    std::vector<float> runEncoder(const std::vector<int> &tokenIds);
    std::vector<float> runDecoderStep(const std::vector<int64_t> &decoderInputIds,
                                      const std::vector<float> &encoderOutput,
                                      const std::vector<int64_t> &attentionMask,
                                      const std::vector<int64_t> &decoderShape,
                                      const std::vector<int64_t> &encoderShape);
    std::vector<int> runDecoder(const std::vector<float> &encoderOutput, 
                               const std::vector<int64_t> &attentionMask);
}; 