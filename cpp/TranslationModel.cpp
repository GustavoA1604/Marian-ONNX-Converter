#include "TranslationModel.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iomanip>

#ifdef USE_SIMD
#include <immintrin.h>
#endif

std::string decodeUnicodeEscape(const std::string &input, size_t &i) {
    if (i + 5 >= input.length()) {
        return std::string(1, input[i]);
    }

    std::string hexCode = input.substr(i + 2, 4);

    for (char c : hexCode) {
        if (!std::isxdigit(c)) {
            return std::string(1, input[i]);
        }
    }

    try {
        int codePoint = std::stoi(hexCode, nullptr, 16);
        i += 5;

        if (codePoint <= 0x7F) {
            return std::string(1, static_cast<char>(codePoint));
        } else if (codePoint <= 0x7FF) {
            std::string result;
            result += static_cast<char>(0xC0 | (codePoint >> 6));
            result += static_cast<char>(0x80 | (codePoint & 0x3F));
            return result;
        } else if (codePoint <= 0xFFFF) {
            std::string result;
            result += static_cast<char>(0xE0 | (codePoint >> 12));
            result += static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (codePoint & 0x3F));
            return result;
        } else {
            std::string result;
            result += static_cast<char>(0xF0 | (codePoint >> 18));
            result += static_cast<char>(0x80 | ((codePoint >> 12) & 0x3F));
            result += static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (codePoint & 0x3F));
            return result;
        }
    } catch (const std::exception &) {
        return std::string(1, input[i]);
    }
}

std::string decodeJsonEscapes(const std::string &input) {
    std::string result;
    result.reserve(input.length());

    for (size_t i = 0; i < input.length(); ++i) {
        if (input[i] != '\\' || i + 1 >= input.length()) {
            result += input[i];
            continue;
        }

        char next = input[i + 1];
        // clang-format off
        switch (next) {
            case '"': result += '"'; i++; break;
            case '\\': result += '\\'; i++; break;
            case '/': result += '/'; i++; break;
            case 'b': result += '\b'; i++; break;
            case 'f': result += '\f'; i++; break;
            case 'n': result += '\n'; i++; break;
            case 'r': result += '\r'; i++; break;
            case 't': result += '\t'; i++; break;
            case 'u': result += decodeUnicodeEscape(input, i); break;
            default: result += input[i]; break;
        }
        // clang-format on
    }

    return result;
}

MemoryMappedWeights::MemoryMappedWeights(const std::string &path) : fd_(-1), data_(nullptr), size_(0) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ == -1) {
        std::cerr << "Failed to open weights file: " << path << std::endl;
        return;
    }

    struct stat st;
    if (fstat(fd_, &st) == -1) {
        std::cerr << "Failed to get file size: " << path << std::endl;
        close(fd_);
        fd_ = -1;
        return;
    }

    size_ = st.st_size;

    int flags = MAP_PRIVATE | MAP_POPULATE;

    data_ = static_cast<float *>(mmap(nullptr, size_, PROT_READ, flags, fd_, 0));
    if (data_ == MAP_FAILED) {
        data_ = static_cast<float *>(mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (data_ == MAP_FAILED) {
            std::cerr << "Failed to memory map file: " << path << std::endl;
            close(fd_);
            fd_ = -1;
            data_ = nullptr;
            return;
        }
    }

    madvise(data_, size_, MADV_SEQUENTIAL | MADV_WILLNEED);

    volatile float prefetchSum = 0;
    size_t prefetchSize = std::min(size_, static_cast<size_t>(64 * 1024));
    for (size_t i = 0; i < prefetchSize / sizeof(float); i += 16) {
        prefetchSum += data_[i];
    }

    double sizeInMB = static_cast<double>(size_) / (1024.0 * 1024.0);
    if (sizeInMB >= 1.0) {
        std::cout << "Memory mapped " << (size_ / sizeof(float)) << " weights from " << path 
                  << " (size: " << std::fixed << std::setprecision(1) << sizeInMB << " MB)" << std::endl;
    } else {
        double sizeInKB = static_cast<double>(size_) / 1024.0;
        std::cout << "Memory mapped " << (size_ / sizeof(float)) << " weights from " << path 
                  << " (size: " << std::fixed << std::setprecision(1) << sizeInKB << " kB)" << std::endl;
    }
}

MemoryMappedWeights::~MemoryMappedWeights() {
    if (data_ != nullptr && data_ != MAP_FAILED) {
        munmap(data_, size_);
    }
    if (fd_ != -1) {
        close(fd_);
    }
}

MemoryMappedWeights::MemoryMappedWeights(MemoryMappedWeights &&other) noexcept : fd_(other.fd_), data_(other.data_), size_(other.size_) {
    other.fd_ = -1;
    other.data_ = nullptr;
    other.size_ = 0;
}

const float *MemoryMappedWeights::data() const {
    return data_;
}

size_t MemoryMappedWeights::size() const {
    return size_ / sizeof(float);
}

bool MemoryMappedWeights::isValid() const {
    return data_ != nullptr && fd_ != -1;
}

inline const float &MemoryMappedWeights::operator[](size_t index) const {
    return data_[index];
}

inline void MemoryMappedWeights::prefetchRange(size_t startIdx, size_t count) const {
    const char *startAddr = reinterpret_cast<const char *>(data_ + startIdx);
    const char *endAddr = reinterpret_cast<const char *>(data_ + startIdx + count);

    for (const char *addr = startAddr; addr < endAddr; addr += 64) {
        __builtin_prefetch(addr, 0, 3);
    }
}

inline const float *MemoryMappedWeights::getAlignedPtr(size_t offset) const {
    return data_ + offset;
}

void computeLogits(const std::vector<float> &hiddenState,
                   const MemoryMappedWeights &weights,
                   const MemoryMappedWeights &bias,
                   std::vector<float> &logits,
                   int vocabSize,
                   int hiddenSize) {

    const float *hiddenPtr = hiddenState.data();
    const float *weightsPtr = weights.getAlignedPtr();
    const float *biasPtr = bias.getAlignedPtr();

    constexpr int BLOCK_SIZE = 64;

    for (int blockStart = 0; blockStart < vocabSize; blockStart += BLOCK_SIZE) {
        int blockEnd = std::min(blockStart + BLOCK_SIZE, vocabSize);

        if (blockEnd < vocabSize) {
            weights.prefetchRange(blockEnd * hiddenSize, BLOCK_SIZE * hiddenSize);
            bias.prefetchRange(blockEnd, BLOCK_SIZE);
        }

        for (int i = blockStart; i < blockEnd; ++i) {
            const float *weightRow = weightsPtr + i * hiddenSize;

            float sum = biasPtr[i];

            int j = 0;
            for (; j <= hiddenSize - 8; j += 8) {
                sum += hiddenPtr[j] * weightRow[j];
                sum += hiddenPtr[j + 1] * weightRow[j + 1];
                sum += hiddenPtr[j + 2] * weightRow[j + 2];
                sum += hiddenPtr[j + 3] * weightRow[j + 3];
                sum += hiddenPtr[j + 4] * weightRow[j + 4];
                sum += hiddenPtr[j + 5] * weightRow[j + 5];
                sum += hiddenPtr[j + 6] * weightRow[j + 6];
                sum += hiddenPtr[j + 7] * weightRow[j + 7];
            }

            for (; j < hiddenSize; ++j) {
                sum += hiddenPtr[j] * weightRow[j];
            }

            logits[i] = sum;
        }
    }
}

#ifdef USE_SIMD
void computeLogitsSimd(const std::vector<float> &hiddenState,
                       const MemoryMappedWeights &weights,
                       const MemoryMappedWeights &bias,
                       std::vector<float> &logits,
                       int vocabSize,
                       int hiddenSize) {

    const float *hiddenPtr = hiddenState.data();
    const float *weightsPtr = weights.getAlignedPtr();
    const float *biasPtr = bias.getAlignedPtr();

    for (int i = 0; i < vocabSize; ++i) {
        const float *weightRow = weightsPtr + i * hiddenSize;

        __m256 sumVec = _mm256_setzero_ps();

        int j = 0;
        for (; j <= hiddenSize - 8; j += 8) {
            __m256 hiddenVec = _mm256_loadu_ps(&hiddenPtr[j]);
            __m256 weightVec = _mm256_loadu_ps(&weightRow[j]);
            sumVec = _mm256_fmadd_ps(hiddenVec, weightVec, sumVec);
        }

        float sumArray[8];
        _mm256_storeu_ps(sumArray, sumVec);
        float sum = sumArray[0] + sumArray[1] + sumArray[2] + sumArray[3] + sumArray[4] + sumArray[5] + sumArray[6] + sumArray[7];

        for (; j < hiddenSize; ++j) {
            sum += hiddenPtr[j] * weightRow[j];
        }

        logits[i] = sum + biasPtr[i];
    }
}
#endif

TranslationModel::TranslationModel(const std::string &modelDir) : modelDir_(modelDir), env_(ORT_LOGGING_LEVEL_WARNING, "MarianTranslationModel") {
    vocab_ = loadVocab(modelDir_ + "/vocab.json");
    buildReverseVocab();
    config_ = loadConfig(modelDir_ + "/config.json");
    lmWeights_ = std::make_unique<MemoryMappedWeights>(modelDir_ + "/lm_weight_raw.bin");
    lmBias_ = std::make_unique<MemoryMappedWeights>(modelDir_ + "/lm_bias_raw.bin");
    if (!lmWeights_->isValid() || !lmBias_->isValid()) {
        throw std::runtime_error("Failed to memory map weight files");
    }

    if (!spProcessor_.Load(modelDir_ + "/source.spm").ok()) {
        throw std::runtime_error("Failed to load SentencePiece model");
    }

    sessionOptions_.SetIntraOpNumThreads(1);
    encoderSession_ = std::make_unique<Ort::Session>(env_, (modelDir_ + "/encoder.onnx").c_str(), sessionOptions_);
    decoderSession_ = std::make_unique<Ort::Session>(env_, (modelDir_ + "/decoder.onnx").c_str(), sessionOptions_);
}

std::unordered_map<std::string, int> TranslationModel::loadVocab(const std::string &vocabPath) {
    std::unordered_map<std::string, int> vocab;
    std::ifstream file(vocabPath);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open vocab file: " + vocabPath);
    }

    std::string line;
    std::getline(file, line);

    size_t pos = 1;
    while (pos < line.length()) {
        size_t keyStart = line.find('"', pos);
        if (keyStart == std::string::npos) {
            break;
        }

        keyStart++;

        size_t keyEnd = keyStart;
        bool foundEnd = false;
        while (keyEnd < line.length()) {
            keyEnd = line.find('"', keyEnd);
            if (keyEnd == std::string::npos) {
                break;
            }

            int backslashCount = 0;
            size_t checkPos = keyEnd - 1;
            while (checkPos >= keyStart && checkPos < line.length() && line[checkPos] == '\\') {
                backslashCount++;
                if (checkPos == 0) {
                    break;
                }
                checkPos--;
            }

            if (backslashCount % 2 == 0) {
                foundEnd = true;
                break;
            }

            keyEnd++;
        }

        if (!foundEnd || keyEnd == std::string::npos) {
            break;
        }

        std::string key = line.substr(keyStart, keyEnd - keyStart);
        std::string decodedKey = decodeJsonEscapes(key);

        size_t colon = line.find(':', keyEnd);
        size_t valueStart = line.find_first_of("0123456789", colon);
        size_t valueEnd = line.find_first_of(",}", valueStart);

        if (valueStart != std::string::npos && valueEnd != std::string::npos) {
            int value = std::stoi(line.substr(valueStart, valueEnd - valueStart));
            vocab[decodedKey] = value;
        }

        pos = valueEnd + 1;
    }

    if (vocab.empty()) {
        throw std::runtime_error("Failed to load vocabulary");
    }
    std::cout << "Loaded vocab with " << vocab.size() << " tokens" << std::endl;
    return vocab;
}

MarianConfig TranslationModel::loadConfig(const std::string &configPath) {
    MarianConfig config = {};
    std::ifstream file(configPath);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + configPath);
    }

    std::string line;
    std::string tempLine;
    while (std::getline(file, tempLine)) {
        line += tempLine;
    }

    auto findValue = [&](const std::string &key) -> int {
        std::string search = "\"" + key + "\"";
        size_t pos = line.find(search);
        if (pos == std::string::npos) {
            throw std::runtime_error("Failed to find " + key + " in config");
        }

        size_t colonPos = line.find(":", pos);
        if (colonPos == std::string::npos) {
            throw std::runtime_error("Failed to find " + key + " in config");
        }

        size_t valueStart = colonPos + 1;
        while (valueStart < line.length() && std::isspace(line[valueStart])) {
            valueStart++;
        }

        size_t valueEnd = line.find_first_of(",}", valueStart);
        if (valueEnd == std::string::npos) {
            throw std::runtime_error("Failed to find " + key + " in config");
        }

        std::string valueStr = line.substr(valueStart, valueEnd - valueStart);
        while (!valueStr.empty() && std::isspace(valueStr.back())) {
            valueStr.pop_back();
        }
        return std::stoi(valueStr);
    };

    config.decoderStartTokenId = findValue("decoder_start_token_id");
    config.eosTokenId = findValue("eos_token_id");
    config.padTokenId = findValue("pad_token_id");
    config.maxLength = findValue("max_length");
    config.vocabSize = findValue("vocab_size");

    std::cout << "Loaded config:" << std::endl;
    std::cout << "  decoder_start_token_id: " << config.decoderStartTokenId << std::endl;
    std::cout << "  eos_token_id: " << config.eosTokenId << std::endl;
    std::cout << "  pad_token_id: " << config.padTokenId << std::endl;
    std::cout << "  max_length: " << config.maxLength << std::endl;
    std::cout << "  vocab_size: " << config.vocabSize << std::endl;

    return config;
}

void TranslationModel::buildReverseVocab() {
    idToToken_.clear();
    for (const auto &pair : vocab_) {
        idToToken_[pair.second] = pair.first;
    }
}

std::vector<int> TranslationModel::tokenize(const std::string &sentence) {
    std::vector<std::string> pieces;
    if (!spProcessor_.Encode(sentence, &pieces).ok()) {
        throw std::runtime_error("Failed to encode sentence");
    }

    std::vector<int> ids;
    for (const auto &piece : pieces) {
        auto it = vocab_.find(piece);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
            continue;
        }

        auto unkIt = vocab_.find("<unk>");
        if (unkIt == vocab_.end()) {
            throw std::runtime_error("Error: Cannot find token '" + piece + "' or <unk> in vocab");
        }

        ids.push_back(unkIt->second);
        std::cerr << "Warning: Unknown token '" << piece << "', using <unk>" << std::endl;
    }

    auto eosIt = vocab_.find("</s>");
    if (eosIt != vocab_.end()) {
        ids.push_back(eosIt->second);
    }

    return ids;
}

std::string TranslationModel::detokenize(const std::vector<int> &ids) {
    std::string result;
    bool firstToken = true;

    for (int id : ids) {
        auto it = idToToken_.find(id);
        if (it == idToToken_.end()) {
            result += "<UNK_ID:" + std::to_string(id) + ">";
            continue;
        }

        std::string token = it->second;

        if (token == "</s>") {
            break;
        }

        if (token == "<unk>") {
            result += "<UNK>";
            continue;
        }

        if (token == "<pad>") {
            continue;
        }

        token = decodeJsonEscapes(token);

        if (token.length() >= 3 && token.substr(0, 3) == "▁") {
            if (!firstToken) {
                result += " ";
            }
            result += token.substr(3);
        } else {
            result += token;
        }

        firstToken = false;
    }

    return result;
}

TranslationModel::BatchedInputs TranslationModel::createBatch(const std::vector<std::string> &sentences) {
    BatchedInputs batch;
    batch.sentences = sentences;
    batch.batchSize = sentences.size();

    std::vector<std::vector<int>> allTokenIds;
    size_t maxLen = 0;

    for (size_t i = 0; i < sentences.size(); ++i) {
        std::vector<int> tokenIds = tokenize(sentences[i]);
        if (tokenIds.empty()) {
            throw std::runtime_error("Failed to tokenize sentence: " + sentences[i]);
        }

        allTokenIds.push_back(tokenIds);
        maxLen = std::max(maxLen, tokenIds.size());
        batch.originalLengths.push_back(tokenIds.size());
    }

    batch.maxSequenceLength = maxLen;
    batch.inputIds.resize(batch.batchSize);
    batch.attentionMasks.resize(batch.batchSize);

    for (size_t i = 0; i < batch.batchSize; ++i) {
        const auto &tokenIds = allTokenIds[i];

        batch.inputIds[i].resize(maxLen, config_.padTokenId);
        batch.attentionMasks[i].resize(maxLen, 0);

        for (size_t j = 0; j < tokenIds.size(); ++j) {
            batch.inputIds[i][j] = static_cast<int64_t>(tokenIds[j]);
            batch.attentionMasks[i][j] = 1;
        }
    }

    return batch;
}

TranslationModel::BatchedEncoderOutput TranslationModel::runBatchedEncoder(const BatchedInputs &batch) {
    try {
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName0 = encoderSession_->GetInputNameAllocated(0, allocator);
        auto inputName1 = encoderSession_->GetInputNameAllocated(1, allocator);
        std::vector<const char *> inputNames = {inputName0.get(), inputName1.get()};

        auto outputName0 = encoderSession_->GetOutputNameAllocated(0, allocator);
        std::vector<const char *> outputNames = {outputName0.get()};

        std::vector<int64_t> flatInputIds;
        std::vector<int64_t> flatAttentionMasks;

        flatInputIds.reserve(batch.batchSize * batch.maxSequenceLength);
        flatAttentionMasks.reserve(batch.batchSize * batch.maxSequenceLength);

        for (size_t i = 0; i < batch.batchSize; ++i) {
            for (size_t j = 0; j < batch.maxSequenceLength; ++j) {
                flatInputIds.push_back(batch.inputIds[i][j]);
                flatAttentionMasks.push_back(batch.attentionMasks[i][j]);
            }
        }

        std::vector<int64_t> inputShape = {static_cast<int64_t>(batch.batchSize), static_cast<int64_t>(batch.maxSequenceLength)};

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value inputIdsTensor =
            Ort::Value::CreateTensor<int64_t>(memoryInfo, flatInputIds.data(), flatInputIds.size(), inputShape.data(), inputShape.size());

        Ort::Value attentionMaskTensor =
            Ort::Value::CreateTensor<int64_t>(memoryInfo, flatAttentionMasks.data(), flatAttentionMasks.size(), inputShape.data(), inputShape.size());

        std::vector<Ort::Value> inputTensors;
        inputTensors.push_back(std::move(inputIdsTensor));
        inputTensors.push_back(std::move(attentionMaskTensor));

        auto outputTensors = encoderSession_->Run(
            Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), inputNames.size(), outputNames.data(), outputNames.size());

        float *outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        size_t outputSize = 1;
        for (auto dim : outputShape) {
            outputSize *= dim;
        }

        BatchedEncoderOutput result;
        result.encoderOutputs = std::vector<float>(outputData, outputData + outputSize);
        result.shape = outputShape;
        result.batchSize = outputShape[0];
        result.sequenceLength = outputShape[1];
        result.hiddenSize = outputShape[2];

        return result;

    } catch (const Ort::Exception &e) {
        throw std::runtime_error("Batched Encoder ONNX Runtime error: " + std::string(e.what()));
    } catch (const std::exception &e) {
        throw std::runtime_error("Batched Encoder error: " + std::string(e.what()));
    }
}

std::vector<std::vector<int>> TranslationModel::runBatchedDecoder(const BatchedEncoderOutput &encoderOutput, const BatchedInputs &batchInputs) {
    size_t batchSize = encoderOutput.batchSize;
    std::vector<std::vector<int>> allGeneratedTokens(batchSize);
    std::vector<std::vector<int64_t>> allDecoderInputIds(batchSize);
    std::vector<bool> finished(batchSize, false);

    for (size_t i = 0; i < batchSize; ++i) {
        allGeneratedTokens[i].push_back(config_.decoderStartTokenId);
        allDecoderInputIds[i] = {static_cast<int64_t>(config_.decoderStartTokenId)};
    }

    for (int step = 0; step < config_.maxLength; ++step) {
        if (std::all_of(finished.begin(), finished.end(), [](bool f) { return f; })) {
            break;
        }

        for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
            if (finished[batchIdx]) {
                continue;
            }

            size_t encoderSeqLen = batchInputs.originalLengths[batchIdx];
            size_t encoderStartIdx = batchIdx * encoderOutput.sequenceLength * encoderOutput.hiddenSize;

            std::vector<float> singleEncoderOutput;
            singleEncoderOutput.reserve(encoderSeqLen * encoderOutput.hiddenSize);

            for (size_t seqPos = 0; seqPos < encoderSeqLen; ++seqPos) {
                size_t posStart = encoderStartIdx + seqPos * encoderOutput.hiddenSize;
                for (size_t h = 0; h < encoderOutput.hiddenSize; ++h) {
                    singleEncoderOutput.push_back(encoderOutput.encoderOutputs[posStart + h]);
                }
            }

            std::vector<int64_t> attentionMask;
            attentionMask.reserve(encoderSeqLen);
            for (size_t i = 0; i < encoderSeqLen; ++i) {
                attentionMask.push_back(1);
            }

            std::vector<int64_t> decoderShape = {1, static_cast<int64_t>(allDecoderInputIds[batchIdx].size())};
            std::vector<int64_t> encoderShape = {1, static_cast<int64_t>(encoderSeqLen), static_cast<int64_t>(encoderOutput.hiddenSize)};

            std::vector<float> decoderOutput =
                runDecoderStep(allDecoderInputIds[batchIdx], singleEncoderOutput, attentionMask, decoderShape, encoderShape);

            if (decoderOutput.empty()) {
                std::cerr << "Decoder step failed for batch " << batchIdx << "!" << std::endl;
                finished[batchIdx] = true;
                continue;
            }

            int seqLen = allDecoderInputIds[batchIdx].size();
            std::vector<float> lastHidden(encoderOutput.hiddenSize);
            for (size_t i = 0; i < encoderOutput.hiddenSize; ++i) {
                lastHidden[i] = decoderOutput[(seqLen - 1) * encoderOutput.hiddenSize + i];
            }

            std::vector<float> logits(config_.vocabSize, 0.0f);
#ifdef USE_SIMD
            computeLogitsSimd(lastHidden, *lmWeights_, *lmBias_, logits, config_.vocabSize, encoderOutput.hiddenSize);
#else
            computeLogits(lastHidden, *lmWeights_, *lmBias_, logits, config_.vocabSize, encoderOutput.hiddenSize);
#endif

            // Find argmax (greedy search)
            int nextTokenId = 0;
            float maxLogit = logits[0];
            for (int i = 1; i < config_.vocabSize; ++i) {
                if (logits[i] > maxLogit) {
                    maxLogit = logits[i];
                    nextTokenId = i;
                }
            }

            allGeneratedTokens[batchIdx].push_back(nextTokenId);
            allDecoderInputIds[batchIdx].push_back(static_cast<int64_t>(nextTokenId));

            if (nextTokenId == config_.eosTokenId) {
                finished[batchIdx] = true;
            }
        }
    }

    return allGeneratedTokens;
}

std::vector<float> TranslationModel::runEncoder(const std::vector<int> &tokenIds) {
    try {
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName0 = encoderSession_->GetInputNameAllocated(0, allocator);
        auto inputName1 = encoderSession_->GetInputNameAllocated(1, allocator);
        std::vector<const char *> inputNames = {inputName0.get(), inputName1.get()};

        auto outputName0 = encoderSession_->GetOutputNameAllocated(0, allocator);
        std::vector<const char *> outputNames = {outputName0.get()};

        std::vector<int64_t> inputIds;
        std::vector<int64_t> attentionMask;
        inputIds.reserve(tokenIds.size());
        attentionMask.reserve(tokenIds.size());

        for (int id : tokenIds) {
            inputIds.push_back(static_cast<int64_t>(id));
            attentionMask.push_back(1);
        }

        std::vector<int64_t> inputShape = {1, static_cast<int64_t>(tokenIds.size())};

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value inputIdsTensor =
            Ort::Value::CreateTensor<int64_t>(memoryInfo, inputIds.data(), inputIds.size(), inputShape.data(), inputShape.size());

        Ort::Value attentionMaskTensor =
            Ort::Value::CreateTensor<int64_t>(memoryInfo, attentionMask.data(), attentionMask.size(), inputShape.data(), inputShape.size());

        std::vector<Ort::Value> inputTensors;
        inputTensors.push_back(std::move(inputIdsTensor));
        inputTensors.push_back(std::move(attentionMaskTensor));

        auto outputTensors = encoderSession_->Run(
            Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), inputNames.size(), outputNames.data(), outputNames.size());

        float *outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        size_t outputSize = 1;
        for (auto dim : outputShape) {
            outputSize *= dim;
        }

        return std::vector<float>(outputData, outputData + outputSize);

    } catch (const Ort::Exception &e) {
        throw std::runtime_error("Encoder ONNX Runtime error: " + std::string(e.what()));
    } catch (const std::exception &e) {
        throw std::runtime_error("Encoder error: " + std::string(e.what()));
    }
}

std::vector<float> TranslationModel::runDecoderStep(const std::vector<int64_t> &decoderInputIds,
                                                    const std::vector<float> &encoderOutput,
                                                    const std::vector<int64_t> &attentionMask,
                                                    const std::vector<int64_t> &decoderShape,
                                                    const std::vector<int64_t> &encoderShape) {
    try {
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName0 = decoderSession_->GetInputNameAllocated(0, allocator);
        auto inputName1 = decoderSession_->GetInputNameAllocated(1, allocator);
        auto inputName2 = decoderSession_->GetInputNameAllocated(2, allocator);
        std::vector<const char *> inputNames = {inputName0.get(), inputName1.get(), inputName2.get()};

        auto outputName0 = decoderSession_->GetOutputNameAllocated(0, allocator);
        std::vector<const char *> outputNames = {outputName0.get()};

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value decoderIdsTensor = Ort::Value::CreateTensor<int64_t>(
            memoryInfo, const_cast<int64_t *>(decoderInputIds.data()), decoderInputIds.size(), decoderShape.data(), decoderShape.size());

        Ort::Value encoderTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, const_cast<float *>(encoderOutput.data()), encoderOutput.size(), encoderShape.data(), encoderShape.size());

        std::vector<int64_t> attentionShape = {encoderShape[0], encoderShape[1]};
        Ort::Value attentionTensor = Ort::Value::CreateTensor<int64_t>(
            memoryInfo, const_cast<int64_t *>(attentionMask.data()), attentionMask.size(), attentionShape.data(), attentionShape.size());

        std::vector<Ort::Value> inputTensors;
        inputTensors.push_back(std::move(decoderIdsTensor));
        inputTensors.push_back(std::move(encoderTensor));
        inputTensors.push_back(std::move(attentionTensor));

        auto outputTensors = decoderSession_->Run(
            Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), inputNames.size(), outputNames.data(), outputNames.size());

        float *outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        size_t outputSize = 1;
        for (auto dim : outputShape) {
            outputSize *= dim;
        }

        return std::vector<float>(outputData, outputData + outputSize);

    } catch (const Ort::Exception &e) {
        throw std::runtime_error("Decoder ONNX Runtime error: " + std::string(e.what()));
    } catch (const std::exception &e) {
        throw std::runtime_error("Decoder error: " + std::string(e.what()));
    }
}

std::vector<int> TranslationModel::runDecoder(const std::vector<float> &encoderOutput, const std::vector<int64_t> &attentionMask) {
    std::vector<int> generatedTokens;
    generatedTokens.push_back(config_.decoderStartTokenId);
    std::vector<int64_t> decoderInputIds = {static_cast<int64_t>(config_.decoderStartTokenId)};

    for (int step = 0; step < config_.maxLength; ++step) {
        std::vector<int64_t> decoderShape = {1, static_cast<int64_t>(decoderInputIds.size())};
        std::vector<int64_t> encoderShape = {1, static_cast<int64_t>(attentionMask.size()), HIDDEN_SIZE};

        std::vector<float> decoderOutput = runDecoderStep(decoderInputIds, encoderOutput, attentionMask, decoderShape, encoderShape);

        if (decoderOutput.empty()) {
            throw std::runtime_error("Decoder step failed!");
        }

        int seqLen = decoderInputIds.size();

        std::vector<float> lastHidden(HIDDEN_SIZE);
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            lastHidden[i] = decoderOutput[(seqLen - 1) * HIDDEN_SIZE + i];
        }

        std::vector<float> logits(config_.vocabSize, 0.0f);
#ifdef USE_SIMD
        computeLogitsSimd(lastHidden, *lmWeights_, *lmBias_, logits, config_.vocabSize, HIDDEN_SIZE);
#else
        computeLogits(lastHidden, *lmWeights_, *lmBias_, logits, config_.vocabSize, HIDDEN_SIZE);
#endif

        // Find argmax (greedy search)
        int nextTokenId = 0;
        float maxLogit = logits[0];

        int i = 1;
        for (; i <= config_.vocabSize - 4; i += 4) {
            if (logits[i] > maxLogit) {
                maxLogit = logits[i];
                nextTokenId = i;
            }
            if (logits[i + 1] > maxLogit) {
                maxLogit = logits[i + 1];
                nextTokenId = i + 1;
            }
            if (logits[i + 2] > maxLogit) {
                maxLogit = logits[i + 2];
                nextTokenId = i + 2;
            }
            if (logits[i + 3] > maxLogit) {
                maxLogit = logits[i + 3];
                nextTokenId = i + 3;
            }
        }

        for (; i < config_.vocabSize; ++i) {
            if (logits[i] > maxLogit) {
                maxLogit = logits[i];
                nextTokenId = i;
            }
        }

        generatedTokens.push_back(nextTokenId);
        decoderInputIds.push_back(static_cast<int64_t>(nextTokenId));

        if (nextTokenId == config_.eosTokenId) {
            break;
        }
    }

    return generatedTokens;
}

std::vector<std::string> TranslationModel::translate(const std::vector<std::string> &sentences) {
    if (sentences.empty()) {
        throw std::runtime_error("No sentences provided for translation");
    }

    auto inferenceStart = std::chrono::high_resolution_clock::now();

    BatchedInputs batch = createBatch(sentences);

    auto encoderStart = std::chrono::high_resolution_clock::now();
    BatchedEncoderOutput encoderOutput = runBatchedEncoder(batch);
    auto encoderEnd = std::chrono::high_resolution_clock::now();
    double encoderTime = std::chrono::duration<double, std::milli>(encoderEnd - encoderStart).count();

    auto decoderStart = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> allGeneratedTokens = runBatchedDecoder(encoderOutput, batch);
    auto decoderEnd = std::chrono::high_resolution_clock::now();
    auto inferenceEnd = std::chrono::high_resolution_clock::now();

    double decoderTime = std::chrono::duration<double, std::milli>(decoderEnd - decoderStart).count();
    double totalInferenceTime = std::chrono::duration<double, std::milli>(inferenceEnd - inferenceStart).count();

    std::vector<std::string> translatedSentences;
    translatedSentences.reserve(sentences.size());

    for (size_t i = 0; i < sentences.size(); ++i) {
        std::string translation = detokenize(allGeneratedTokens[i]);
        translatedSentences.push_back(translation);
    }

    size_t totalGeneratedTokens = 0;
    for (const auto &tokens : allGeneratedTokens) {
        totalGeneratedTokens += tokens.size() - 1;
    }

    double tokensPerSecond = totalGeneratedTokens / (totalInferenceTime / 1000.0);

    std::cout << "\nSentences processed: " << sentences.size() << std::endl;
    std::cout << "Encoder time: " << encoderTime << " ms" << std::endl;
    std::cout << "Decoder time: " << decoderTime << " ms" << std::endl;
    std::cout << "Total inference time: " << totalInferenceTime << " ms" << std::endl;
    std::cout << "Total generated tokens: " << totalGeneratedTokens << std::endl;
    std::cout << "Tokens per second: " << tokensPerSecond << std::endl;
    std::cout << "Sentences per second: " << sentences.size() / (totalInferenceTime / 1000.0) << std::endl;

    return translatedSentences;
}

std::string TranslationModel::translateSingle(const std::string &inputSentence) {
    auto inferenceStart = std::chrono::high_resolution_clock::now();

    std::vector<int> tokenIds = tokenize(inputSentence);
    if (tokenIds.empty()) {
        throw std::runtime_error("Failed to tokenize input sentence");
    }

    std::vector<float> encoderOutput = runEncoder(tokenIds);
    if (encoderOutput.empty()) {
        throw std::runtime_error("Encoder inference failed");
    }

    std::vector<int64_t> attentionMask;
    attentionMask.reserve(tokenIds.size());
    for (size_t i = 0; i < tokenIds.size(); ++i) {
        attentionMask.push_back(1);
    }

    std::vector<int> generatedTokens = runDecoder(encoderOutput, attentionMask);

    std::string translation = detokenize(generatedTokens);

    auto inferenceEnd = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double, std::milli>(inferenceEnd - inferenceStart).count();

    std::cout << "Inference time: " << totalTime << " ms" << std::endl;

    return translation;
}