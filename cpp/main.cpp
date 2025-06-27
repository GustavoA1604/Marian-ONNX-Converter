#include <cctype>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

// Configuration constants
const std::string OUTPUT_DIR = "../outs/";

std::string decodeJsonEscapes(const std::string &input) {
    std::string result;

    for (size_t i = 0; i < input.length(); ++i) {
        if (input[i] == '\\' && i + 1 < input.length()) {
            char next = input[i + 1];
            switch (next) {
                case '"':
                    result += '"';
                    i++;
                    break;
                case '\\':
                    result += '\\';
                    i++;
                    break;
                case '/':
                    result += '/';
                    i++;
                    break;
                case 'b':
                    result += '\b';
                    i++;
                    break;
                case 'f':
                    result += '\f';
                    i++;
                    break;
                case 'n':
                    result += '\n';
                    i++;
                    break;
                case 'r':
                    result += '\r';
                    i++;
                    break;
                case 't':
                    result += '\t';
                    i++;
                    break;
                case 'u':
                    // Handle Unicode escape sequences (\uXXXX)
                    if (i + 5 < input.length()) {
                        std::string hex_code = input.substr(i + 2, 4);
                        bool valid_hex = true;
                        for (char c : hex_code) {
                            if (!std::isxdigit(c)) {
                                valid_hex = false;
                                break;
                            }
                        }
                        if (valid_hex) {
                            try {
                                int code_point = std::stoi(hex_code, nullptr, 16);
                                // Convert to UTF-8
                                if (code_point <= 0x7F) {
                                    // 1-byte UTF-8 (ASCII)
                                    result += static_cast<char>(code_point);
                                } else if (code_point <= 0x7FF) {
                                    // 2-byte UTF-8
                                    result += static_cast<char>(0xC0 | (code_point >> 6));
                                    result += static_cast<char>(0x80 | (code_point & 0x3F));
                                } else if (code_point <= 0xFFFF) {
                                    // 3-byte UTF-8 (Basic Multilingual Plane)
                                    result += static_cast<char>(0xE0 | (code_point >> 12));
                                    result += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
                                    result += static_cast<char>(0x80 | (code_point & 0x3F));
                                } else {
                                    // 4-byte UTF-8 (for code points > 0xFFFF)
                                    result += static_cast<char>(0xF0 | (code_point >> 18));
                                    result += static_cast<char>(0x80 | ((code_point >> 12) & 0x3F));
                                    result += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
                                    result += static_cast<char>(0x80 | (code_point & 0x3F));
                                }
                                i += 5; // Skip \uXXXX
                            } catch (const std::exception &) {
                                // If conversion fails, keep the original character
                                result += input[i];
                            }
                        } else {
                            // Invalid hex characters, keep the original character
                            result += input[i];
                        }
                    } else {
                        // Not enough characters for a full unicode escape
                        result += input[i];
                    }
                    break;
                default:
                    // Unknown escape sequence, keep both characters
                    result += input[i];
                    break;
            }
        } else {
            result += input[i];
        }
    }

    return result;
}

std::unordered_map<std::string, int> loadVocab(const std::string &vocab_path) {
    std::unordered_map<std::string, int> vocab;
    std::ifstream file(vocab_path);

    if (!file.is_open()) {
        std::cerr << "Failed to open vocab file: " << vocab_path << std::endl;
        return vocab;
    }

    std::string line;
    std::getline(file, line);

    size_t pos = 1; // Skip opening brace
    while (pos < line.length()) {
        size_t key_start = line.find('"', pos);
        if (key_start == std::string::npos)
            break;
        key_start++;

        size_t key_end = line.find('"', key_start);
        if (key_end == std::string::npos)
            break;

        std::string key = line.substr(key_start, key_end - key_start);

        key = decodeJsonEscapes(key);

        size_t colon = line.find(':', key_end);
        size_t value_start = line.find_first_of("0123456789", colon);
        size_t value_end = line.find_first_of(",}", value_start);

        if (value_start != std::string::npos && value_end != std::string::npos) {
            int value = std::stoi(line.substr(value_start, value_end - value_start));
            vocab[key] = value;
        }

        pos = value_end + 1;
    }

    std::cout << "Loaded vocab with " << vocab.size() << " tokens" << std::endl;
    return vocab;
}

std::string decodeIds(const std::vector<int> &ids, const std::unordered_map<std::string, int> &vocab) {
    // Create reverse mapping (ID -> token)
    static std::unordered_map<int, std::string> id_to_token;
    if (id_to_token.empty()) {
        for (const auto &pair : vocab) {
            id_to_token[pair.second] = pair.first;
        }
    }

    std::string result;
    bool first_token = true;

    for (int id : ids) {
        auto it = id_to_token.find(id);
        if (it != id_to_token.end()) {
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
                if (!first_token) {
                    result += " ";
                }
                result += token.substr(3);
            } else {
                result += token;
            }

            first_token = false;
        } else {
            result += "<UNK_ID:" + std::to_string(id) + ">";
        }
    }

    return result;
}

struct MarianConfig {
    int decoder_start_token_id;
    int eos_token_id;
    int pad_token_id;
    int max_length;
    int vocab_size;
};

MarianConfig loadConfig(const std::string &config_path) {
    MarianConfig config = {};
    std::ifstream file(config_path);

    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << config_path << std::endl;
        return config;
    }

    std::string line;
    std::string temp_line;
    while (std::getline(file, temp_line)) {
        line += temp_line;
    }

    auto findValue = [&](const std::string &key) -> int {
        std::string search = "\"" + key + "\"";
        size_t pos = line.find(search);
        if (pos != std::string::npos) {
            size_t colon_pos = line.find(":", pos);
            if (colon_pos != std::string::npos) {
                size_t value_start = colon_pos + 1;
                while (value_start < line.length() && std::isspace(line[value_start])) {
                    value_start++;
                }

                size_t value_end = line.find_first_of(",}", value_start);
                if (value_end != std::string::npos) {
                    std::string value_str = line.substr(value_start, value_end - value_start);
                    while (!value_str.empty() && std::isspace(value_str.back())) {
                        value_str.pop_back();
                    }
                    return std::stoi(value_str);
                }
            }
        }
        std::cout << "Failed to find " << key << " in config" << std::endl;
        return -1;
    };

    config.decoder_start_token_id = findValue("decoder_start_token_id");
    config.eos_token_id = findValue("eos_token_id");
    config.pad_token_id = findValue("pad_token_id");
    config.max_length = findValue("max_length");
    config.vocab_size = findValue("vocab_size");

    std::cout << "Loaded config:" << std::endl;
    std::cout << "  decoder_start_token_id: " << config.decoder_start_token_id << std::endl;
    std::cout << "  eos_token_id: " << config.eos_token_id << std::endl;
    std::cout << "  pad_token_id: " << config.pad_token_id << std::endl;
    std::cout << "  max_length: " << config.max_length << std::endl;
    std::cout << "  vocab_size: " << config.vocab_size << std::endl;

    return config;
}

class MemoryMappedWeights {
  public:
    MemoryMappedWeights(const std::string &path) : fd_(-1), data_(nullptr), size_(0) {
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

        if (madvise(data_, size_, MADV_SEQUENTIAL | MADV_WILLNEED) != 0) {
            // Non-critical, continue anyway
        }

        volatile float prefetch_sum = 0;
        size_t prefetch_size = std::min(size_, static_cast<size_t>(64 * 1024)); // 64KB
        for (size_t i = 0; i < prefetch_size / sizeof(float); i += 16) {
            prefetch_sum += data_[i];
        }

        std::cout << "Memory mapped " << (size_ / sizeof(float)) << " weights from " << path << " (size: " << size_ / (1024 * 1024) << " MB)"
                  << std::endl;
    }

    ~MemoryMappedWeights() {
        if (data_ != nullptr && data_ != MAP_FAILED) {
            munmap(data_, size_);
        }
        if (fd_ != -1) {
            close(fd_);
        }
    }

    // Non-copyable
    MemoryMappedWeights(const MemoryMappedWeights &) = delete;
    MemoryMappedWeights &operator=(const MemoryMappedWeights &) = delete;

    // Movable
    MemoryMappedWeights(MemoryMappedWeights &&other) noexcept : fd_(other.fd_), data_(other.data_), size_(other.size_) {
        other.fd_ = -1;
        other.data_ = nullptr;
        other.size_ = 0;
    }

    const float *data() const {
        return data_;
    }
    size_t size() const {
        return size_ / sizeof(float);
    }
    bool isValid() const {
        return data_ != nullptr && fd_ != -1;
    }

    inline const float &operator[](size_t index) const {
        return data_[index];
    }

    inline void prefetch_range(size_t start_idx, size_t count) const {
        const char *start_addr = reinterpret_cast<const char *>(data_ + start_idx);
        const char *end_addr = reinterpret_cast<const char *>(data_ + start_idx + count);

        // Prefetch cache lines
        for (const char *addr = start_addr; addr < end_addr; addr += 64) {
            __builtin_prefetch(addr, 0, 3);
        }
    }

    inline const float *get_aligned_ptr(size_t offset = 0) const {
        return data_ + offset;
    }

  private:
    int fd_;
    float *data_;
    size_t size_;
};

std::vector<int> getIds(std::string sentence, const std::unordered_map<std::string, int> &vocab) {
    sentencepiece::SentencePieceProcessor processor;
    const std::string model_path = OUTPUT_DIR + "source.spm";

    if (!processor.Load(model_path).ok()) {
        std::cerr << "Failed to load SentencePiece model from: " << model_path << std::endl;
        return {};
    }

    std::vector<std::string> pieces;
    if (!processor.Encode(sentence, &pieces).ok()) {
        std::cerr << "Failed to encode sentence" << std::endl;
        return {};
    }

    std::cout << "\tPieces: ";
    for (const auto &piece : pieces) {
        std::cout << "'" << piece << "' ";
    }
    std::cout << std::endl;

    std::vector<int> ids;
    for (const auto &piece : pieces) {
        auto it = vocab.find(piece);
        if (it != vocab.end()) {
            ids.push_back(it->second);
        } else {
            auto unk_it = vocab.find("<unk>");
            if (unk_it != vocab.end()) {
                ids.push_back(unk_it->second);
                std::cerr << "Warning: Unknown token '" << piece << "', using <unk>" << std::endl;
            } else {
                std::cerr << "Error: Cannot find token '" << piece << "' or <unk> in vocab" << std::endl;
                return {};
            }
        }
    }

    auto eos_it = vocab.find("</s>");
    if (eos_it != vocab.end()) {
        ids.push_back(eos_it->second); // Should be 0
    }

    std::cout << "\tIDs: ";
    for (const auto &id : ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    std::string decoded_text = decodeIds(ids, vocab);
    std::cout << "\tDecoded text: " << decoded_text << std::endl;

    return ids;
}

std::vector<int64_t> createAttentionMask(const std::vector<int> &input_ids) {
    std::vector<int64_t> attention_mask;
    attention_mask.reserve(input_ids.size());

    for (int id : input_ids) {
        attention_mask.push_back(1);
    }

    return attention_mask;
}

std::vector<int64_t> convertToInt64(const std::vector<int> &input) {
    std::vector<int64_t> output;
    output.reserve(input.size());

    for (int val : input) {
        output.push_back(static_cast<int64_t>(val));
    }

    return output;
}

struct EncoderInputs {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::vector<int64_t> input_shape; // [batch_size, sequence_length]
};

EncoderInputs prepareEncoderInputs(const std::vector<int> &token_ids) {
    EncoderInputs inputs;

    inputs.input_ids = convertToInt64(token_ids);
    inputs.attention_mask = createAttentionMask(token_ids);
    inputs.input_shape = {1, static_cast<int64_t>(token_ids.size())};

    return inputs;
}

std::vector<float> runEncoderInference(const EncoderInputs &inputs) {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MarianEncoder");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        const std::string encoder_path = OUTPUT_DIR + "encoder.onnx";
        Ort::Session encoder_session(env, encoder_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name_0 = encoder_session.GetInputNameAllocated(0, allocator);
        auto input_name_1 = encoder_session.GetInputNameAllocated(1, allocator);
        std::vector<const char *> input_names = {input_name_0.get(), input_name_1.get()};

        auto output_name_0 = encoder_session.GetOutputNameAllocated(0, allocator);
        std::vector<const char *> output_names = {output_name_0.get()};

        std::cout << "Encoder model loaded successfully" << std::endl;
        std::cout << "  Input 0: " << input_names[0] << std::endl;
        std::cout << "  Input 1: " << input_names[1] << std::endl;
        std::cout << "  Output 0: " << output_names[0] << std::endl;

        std::vector<int64_t> input_shape = inputs.input_shape;

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, const_cast<int64_t *>(inputs.input_ids.data()), inputs.input_ids.size(), input_shape.data(), input_shape.size());

        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, const_cast<int64_t *>(inputs.attention_mask.data()), inputs.attention_mask.size(), input_shape.data(), input_shape.size());

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_ids_tensor));
        input_tensors.push_back(std::move(attention_mask_tensor));

        std::cout << "Running encoder inference..." << std::endl;

        auto output_tensors = encoder_session.Run(
            Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_names.size(), output_names.data(), output_names.size());

        std::cout << "Encoder inference completed!" << std::endl;

        float *output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        std::cout << "Encoder output shape: [";
        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::cout << output_shape[i];
            if (i < output_shape.size() - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        size_t output_size = 1;
        for (auto dim : output_shape) {
            output_size *= dim;
        }

        std::vector<float> encoder_output(output_data, output_data + output_size);

        std::cout << "Encoder output size: " << encoder_output.size() << std::endl;
        std::cout << "First few values: ";
        for (size_t i = 0; i < std::min(size_t(5), encoder_output.size()); ++i) {
            std::cout << encoder_output[i] << " ";
        }
        std::cout << std::endl;

        return encoder_output;

    } catch (const Ort::Exception &e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return {};
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return {};
    }
}

std::vector<float> runDecoderStep(const std::vector<int64_t> &decoder_input_ids,
                                  const std::vector<float> &encoder_output,
                                  const std::vector<int64_t> &attention_mask,
                                  const std::vector<int64_t> &decoder_shape,
                                  const std::vector<int64_t> &encoder_shape,
                                  Ort::Session &decoder_session) {
    try {
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_0 = decoder_session.GetInputNameAllocated(0, allocator); // input_ids
        auto input_name_1 = decoder_session.GetInputNameAllocated(1, allocator); // encoder_hidden_states
        auto input_name_2 = decoder_session.GetInputNameAllocated(2, allocator); // attention_mask
        std::vector<const char *> input_names = {input_name_0.get(), input_name_1.get(), input_name_2.get()};

        auto output_name_0 = decoder_session.GetOutputNameAllocated(0, allocator);
        std::vector<const char *> output_names = {output_name_0.get()};

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value decoder_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, const_cast<int64_t *>(decoder_input_ids.data()), decoder_input_ids.size(), decoder_shape.data(), decoder_shape.size());

        Ort::Value encoder_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float *>(encoder_output.data()), encoder_output.size(), encoder_shape.data(), encoder_shape.size());

        std::vector<int64_t> attention_shape = {encoder_shape[0], encoder_shape[1]};
        Ort::Value attention_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, const_cast<int64_t *>(attention_mask.data()), attention_mask.size(), attention_shape.data(), attention_shape.size());

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(decoder_ids_tensor));
        input_tensors.push_back(std::move(encoder_tensor));
        input_tensors.push_back(std::move(attention_tensor));

        auto output_tensors = decoder_session.Run(
            Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_names.size(), output_names.data(), output_names.size());

        float *output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        size_t output_size = 1;
        for (auto dim : output_shape) {
            output_size *= dim;
        }

        return std::vector<float>(output_data, output_data + output_size);

    } catch (const Ort::Exception &e) {
        std::cerr << "Decoder ONNX Runtime error: " << e.what() << std::endl;
        return {};
    } catch (const std::exception &e) {
        std::cerr << "Decoder error: " << e.what() << std::endl;
        return {};
    }
}

void compute_logits_optimized(const std::vector<float> &hidden_state,
                              const MemoryMappedWeights &weights,
                              const MemoryMappedWeights &bias,
                              std::vector<float> &logits,
                              int vocab_size,
                              int hidden_size) {

    const float *hidden_ptr = hidden_state.data();
    const float *weights_ptr = weights.get_aligned_ptr();
    const float *bias_ptr = bias.get_aligned_ptr();

    // Process in blocks for better cache performance
    constexpr int BLOCK_SIZE = 64;

    for (int block_start = 0; block_start < vocab_size; block_start += BLOCK_SIZE) {
        int block_end = std::min(block_start + BLOCK_SIZE, vocab_size);

        if (block_end < vocab_size) {
            weights.prefetch_range(block_end * hidden_size, BLOCK_SIZE * hidden_size);
            bias.prefetch_range(block_end, BLOCK_SIZE);
        }

        for (int i = block_start; i < block_end; ++i) {
            const float *weight_row = weights_ptr + i * hidden_size;

            float sum = bias_ptr[i];

            int j = 0;

            for (; j <= hidden_size - 8; j += 8) {
                sum += hidden_ptr[j] * weight_row[j];
                sum += hidden_ptr[j + 1] * weight_row[j + 1];
                sum += hidden_ptr[j + 2] * weight_row[j + 2];
                sum += hidden_ptr[j + 3] * weight_row[j + 3];
                sum += hidden_ptr[j + 4] * weight_row[j + 4];
                sum += hidden_ptr[j + 5] * weight_row[j + 5];
                sum += hidden_ptr[j + 6] * weight_row[j + 6];
                sum += hidden_ptr[j + 7] * weight_row[j + 7];
            }

            for (; j < hidden_size; ++j) {
                sum += hidden_ptr[j] * weight_row[j];
            }

            logits[i] = sum;
        }
    }
}

#ifdef USE_SIMD
#include <immintrin.h>

void compute_logits_simd(const std::vector<float> &hidden_state,
                         const MemoryMappedWeights &weights,
                         const MemoryMappedWeights &bias,
                         std::vector<float> &logits,
                         int vocab_size,
                         int hidden_size) {

    const float *hidden_ptr = hidden_state.data();
    const float *weights_ptr = weights.get_aligned_ptr();
    const float *bias_ptr = bias.get_aligned_ptr();

    for (int i = 0; i < vocab_size; ++i) {
        const float *weight_row = weights_ptr + i * hidden_size;

        __m256 sum_vec = _mm256_setzero_ps();

        int j = 0;
        for (; j <= hidden_size - 8; j += 8) {
            __m256 hidden_vec = _mm256_loadu_ps(&hidden_ptr[j]);
            __m256 weight_vec = _mm256_loadu_ps(&weight_row[j]);
            sum_vec = _mm256_fmadd_ps(hidden_vec, weight_vec, sum_vec);
        }

        float sum_array[8];
        _mm256_storeu_ps(sum_array, sum_vec);
        float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] + sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

        for (; j < hidden_size; ++j) {
            sum += hidden_ptr[j] * weight_row[j];
        }

        logits[i] = sum + bias_ptr[i];
    }
}
#endif

std::vector<std::string> splitIntoSentences(const std::string &text) {
    std::vector<std::string> sentences;
    std::string current_sentence;
    
    for (size_t i = 0; i < text.length(); ++i) {
        char c = text[i];
        current_sentence += c;
        
        if (c == '.' || c == '!' || c == '?') {
            bool is_sentence_end = true;
            
            size_t next_pos = i + 1;
            while (next_pos < text.length() && std::isspace(text[next_pos])) {
                current_sentence += text[next_pos];
                next_pos++;
            }
            
            if (is_sentence_end && !current_sentence.empty()) {
                std::string trimmed = current_sentence;
                while (!trimmed.empty() && std::isspace(trimmed.front())) {
                    trimmed = trimmed.substr(1);
                }
                while (!trimmed.empty() && std::isspace(trimmed.back())) {
                    trimmed.pop_back();
                }
                
                if (!trimmed.empty()) {
                    sentences.push_back(trimmed);
                }
                current_sentence.clear();
                
                i = next_pos - 1;
            }
        }
    }
    
    if (!current_sentence.empty()) {
        std::string trimmed = current_sentence;
        while (!trimmed.empty() && std::isspace(trimmed.front())) {
            trimmed = trimmed.substr(1);
        }
        while (!trimmed.empty() && std::isspace(trimmed.back())) {
            trimmed.pop_back();
        }
        
        if (!trimmed.empty()) {
            sentences.push_back(trimmed);
        }
    }
    
    return sentences;
}

class TranslationModel {
  private:
    // Configuration and data
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> id_to_token_;
    MarianConfig config_;
    std::unique_ptr<MemoryMappedWeights> lm_weights_;
    std::unique_ptr<MemoryMappedWeights> lm_bias_;

    // ONNX Runtime components
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> encoder_session_;
    std::unique_ptr<Ort::Session> decoder_session_;

    sentencepiece::SentencePieceProcessor sp_processor_;

    std::string model_dir_;

    // Performance constants
    static constexpr int HIDDEN_SIZE = 512;

    std::unordered_map<std::string, int> loadVocab(const std::string &vocab_path) {
        std::unordered_map<std::string, int> vocab;
        std::ifstream file(vocab_path);

        if (!file.is_open()) {
            std::cerr << "Failed to open vocab file: " << vocab_path << std::endl;
            return vocab;
        }

        std::string line;
        std::getline(file, line);

        size_t pos = 1; // Skip opening brace
        while (pos < line.length()) {
            size_t key_start = line.find('"', pos);
            if (key_start == std::string::npos)
                break;
            key_start++;

            size_t key_end = key_start;
            bool found_end = false;
            while (key_end < line.length()) {
                key_end = line.find('"', key_end);
                if (key_end == std::string::npos)
                    break;

                int backslash_count = 0;
                size_t check_pos = key_end - 1;
                while (check_pos >= key_start && check_pos < line.length() && line[check_pos] == '\\') {
                    backslash_count++;
                    if (check_pos == 0)
                        break;
                    check_pos--;
                }

                if (backslash_count % 2 == 0) {
                    found_end = true;
                    break;
                }

                key_end++; // Move past this escaped quote and continue searching
            }

            if (!found_end || key_end == std::string::npos)
                break;

            std::string key = line.substr(key_start, key_end - key_start);

            std::string decoded_key = decodeJsonEscapes(key);

            size_t colon = line.find(':', key_end);
            size_t value_start = line.find_first_of("0123456789", colon);
            size_t value_end = line.find_first_of(",}", value_start);

            if (value_start != std::string::npos && value_end != std::string::npos) {
                int value = std::stoi(line.substr(value_start, value_end - value_start));
                vocab[decoded_key] = value;
            }

            pos = value_end + 1;
        }

        std::cout << "Loaded vocab with " << vocab.size() << " tokens" << std::endl;
        return vocab;
    }

    MarianConfig loadConfig(const std::string &config_path) {
        MarianConfig config = {};
        std::ifstream file(config_path);

        if (!file.is_open()) {
            std::cerr << "Failed to open config file: " << config_path << std::endl;
            return config;
        }

        std::string line;
        std::string temp_line;
        while (std::getline(file, temp_line)) {
            line += temp_line;
        }

        auto findValue = [&](const std::string &key) -> int {
            std::string search = "\"" + key + "\"";
            size_t pos = line.find(search);
            if (pos != std::string::npos) {
                size_t colon_pos = line.find(":", pos);
                if (colon_pos != std::string::npos) {
                    size_t value_start = colon_pos + 1;
                    while (value_start < line.length() && std::isspace(line[value_start])) {
                        value_start++;
                    }

                    size_t value_end = line.find_first_of(",}", value_start);
                    if (value_end != std::string::npos) {
                        std::string value_str = line.substr(value_start, value_end - value_start);
                        while (!value_str.empty() && std::isspace(value_str.back())) {
                            value_str.pop_back();
                        }
                        return std::stoi(value_str);
                    }
                }
            }
            std::cout << "Failed to find " << key << " in config" << std::endl;
            return -1;
        };

        config.decoder_start_token_id = findValue("decoder_start_token_id");
        config.eos_token_id = findValue("eos_token_id");
        config.pad_token_id = findValue("pad_token_id");
        config.max_length = findValue("max_length");
        config.vocab_size = findValue("vocab_size");

        std::cout << "Loaded config:" << std::endl;
        std::cout << "  decoder_start_token_id: " << config.decoder_start_token_id << std::endl;
        std::cout << "  eos_token_id: " << config.eos_token_id << std::endl;
        std::cout << "  pad_token_id: " << config.pad_token_id << std::endl;
        std::cout << "  max_length: " << config.max_length << std::endl;
        std::cout << "  vocab_size: " << config.vocab_size << std::endl;

        return config;
    }

    void buildReverseVocab() {
        id_to_token_.clear();
        for (const auto &pair : vocab_) {
            id_to_token_[pair.second] = pair.first;
        }
    }



    // Private methods for processing
    std::vector<int> tokenize(const std::string &sentence) {
        std::vector<std::string> pieces;
        if (!sp_processor_.Encode(sentence, &pieces).ok()) {
            std::cerr << "Failed to encode sentence" << std::endl;
            return {};
        }

        std::vector<int> ids;
        for (const auto &piece : pieces) {
            auto it = vocab_.find(piece);
            if (it != vocab_.end()) {
                ids.push_back(it->second);
            } else {
                auto unk_it = vocab_.find("<unk>");
                if (unk_it != vocab_.end()) {
                    ids.push_back(unk_it->second);
                    std::cerr << "Warning: Unknown token '" << piece << "', using <unk>" << std::endl;
                } else {
                    std::cerr << "Error: Cannot find token '" << piece << "' or <unk> in vocab" << std::endl;
                    return {};
                }
            }
        }

        auto eos_it = vocab_.find("</s>");
        if (eos_it != vocab_.end()) {
            ids.push_back(eos_it->second);
        }

        return ids;
    }

    struct BatchedInputs {
        std::vector<std::vector<int64_t>> input_ids;
        std::vector<std::vector<int64_t>> attention_masks;
        std::vector<size_t> original_lengths;
        std::vector<std::string> sentences;
        size_t batch_size;
        size_t max_sequence_length;
    };

    BatchedInputs createBatch(const std::vector<std::string> &sentences) {
        BatchedInputs batch;
        batch.sentences = sentences;
        batch.batch_size = sentences.size();
        
        std::vector<std::vector<int>> all_token_ids;
        size_t max_len = 0;
        
        std::cout << "\n--- Batch Tokenization ---" << std::endl;
        for (size_t i = 0; i < sentences.size(); ++i) {
            std::cout << "Sentence " << i + 1 << ": \"" << sentences[i] << "\"" << std::endl;
            
            std::vector<int> token_ids = tokenize(sentences[i]);
            if (token_ids.empty()) {
                throw std::runtime_error("Failed to tokenize sentence: " + sentences[i]);
            }
            
            all_token_ids.push_back(token_ids);
            max_len = std::max(max_len, token_ids.size());
            batch.original_lengths.push_back(token_ids.size());
            
            std::cout << "\tTokens (" << token_ids.size() << "): ";
            for (size_t j = 0; j < std::min(token_ids.size(), size_t(10)); ++j) {
                std::cout << token_ids[j] << " ";
            }
            if (token_ids.size() > 10) std::cout << "...";
            std::cout << std::endl;
        }
        
        batch.max_sequence_length = max_len;
        std::cout << "Batch size: " << batch.batch_size << ", Max length: " << max_len << std::endl;
        
        batch.input_ids.resize(batch.batch_size);
        batch.attention_masks.resize(batch.batch_size);
        
        for (size_t i = 0; i < batch.batch_size; ++i) {
            const auto &token_ids = all_token_ids[i];
            
            batch.input_ids[i].resize(max_len, config_.pad_token_id);
            batch.attention_masks[i].resize(max_len, 0);
            
            for (size_t j = 0; j < token_ids.size(); ++j) {
                batch.input_ids[i][j] = static_cast<int64_t>(token_ids[j]);
                batch.attention_masks[i][j] = 1;
            }
        }
        
        return batch;
    }

    std::string detokenize(const std::vector<int> &ids) {
        std::string result;
        bool first_token = true;

        for (int id : ids) {
            auto it = id_to_token_.find(id);
            if (it != id_to_token_.end()) {
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
                    if (!first_token) {
                        result += " ";
                    }
                    result += token.substr(3);
                } else {
                    result += token;
                }

                first_token = false;
            } else {
                result += "<UNK_ID:" + std::to_string(id) + ">";
            }
        }

        return result;
    }

    std::vector<float> runEncoder(const std::vector<int> &token_ids) {
        try {
            Ort::AllocatorWithDefaultOptions allocator;
            auto input_name_0 = encoder_session_->GetInputNameAllocated(0, allocator);
            auto input_name_1 = encoder_session_->GetInputNameAllocated(1, allocator);
            std::vector<const char *> input_names = {input_name_0.get(), input_name_1.get()};

            auto output_name_0 = encoder_session_->GetOutputNameAllocated(0, allocator);
            std::vector<const char *> output_names = {output_name_0.get()};

            // Convert to int64 and create attention mask
            std::vector<int64_t> input_ids;
            std::vector<int64_t> attention_mask;
            input_ids.reserve(token_ids.size());
            attention_mask.reserve(token_ids.size());

            for (int id : token_ids) {
                input_ids.push_back(static_cast<int64_t>(id));
                attention_mask.push_back(1);
            }

            std::vector<int64_t> input_shape = {1, static_cast<int64_t>(token_ids.size())};

            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_ids_tensor =
                Ort::Value::CreateTensor<int64_t>(memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());

            Ort::Value attention_mask_tensor =
                Ort::Value::CreateTensor<int64_t>(memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size());

            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(input_ids_tensor));
            input_tensors.push_back(std::move(attention_mask_tensor));

            auto output_tensors = encoder_session_->Run(
                Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_names.size(), output_names.data(), output_names.size());

            float *output_data = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

            size_t output_size = 1;
            for (auto dim : output_shape) {
                output_size *= dim;
            }

            return std::vector<float>(output_data, output_data + output_size);

        } catch (const Ort::Exception &e) {
            std::cerr << "Encoder ONNX Runtime error: " << e.what() << std::endl;
            return {};
        } catch (const std::exception &e) {
            std::cerr << "Encoder error: " << e.what() << std::endl;
            return {};
        }
    }

    struct BatchedEncoderOutput {
        std::vector<float> encoder_outputs;
        std::vector<int64_t> shape;
        size_t batch_size;
        size_t sequence_length;
        size_t hidden_size;
    };

    BatchedEncoderOutput runBatchedEncoder(const BatchedInputs &batch) {
        try {
            Ort::AllocatorWithDefaultOptions allocator;
            auto input_name_0 = encoder_session_->GetInputNameAllocated(0, allocator);
            auto input_name_1 = encoder_session_->GetInputNameAllocated(1, allocator);
            std::vector<const char *> input_names = {input_name_0.get(), input_name_1.get()};

            auto output_name_0 = encoder_session_->GetOutputNameAllocated(0, allocator);
            std::vector<const char *> output_names = {output_name_0.get()};

            std::vector<int64_t> flat_input_ids;
            std::vector<int64_t> flat_attention_masks;
            
            flat_input_ids.reserve(batch.batch_size * batch.max_sequence_length);
            flat_attention_masks.reserve(batch.batch_size * batch.max_sequence_length);

            for (size_t i = 0; i < batch.batch_size; ++i) {
                for (size_t j = 0; j < batch.max_sequence_length; ++j) {
                    flat_input_ids.push_back(batch.input_ids[i][j]);
                    flat_attention_masks.push_back(batch.attention_masks[i][j]);
                }
            }

            std::vector<int64_t> input_shape = {
                static_cast<int64_t>(batch.batch_size),
                static_cast<int64_t>(batch.max_sequence_length)
            };

            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, flat_input_ids.data(), flat_input_ids.size(), input_shape.data(), input_shape.size());

            Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, flat_attention_masks.data(), flat_attention_masks.size(), input_shape.data(), input_shape.size());

            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(input_ids_tensor));
            input_tensors.push_back(std::move(attention_mask_tensor));

            std::cout << "\n--- Batched Encoder Inference ---" << std::endl;
            std::cout << "Input shape: [" << batch.batch_size << ", " << batch.max_sequence_length << "]" << std::endl;

            auto output_tensors = encoder_session_->Run(
                Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_names.size(), output_names.data(), output_names.size());

            float *output_data = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

            std::cout << "Output shape: [";
            for (size_t i = 0; i < output_shape.size(); ++i) {
                std::cout << output_shape[i];
                if (i < output_shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;

            size_t output_size = 1;
            for (auto dim : output_shape) {
                output_size *= dim;
            }

            BatchedEncoderOutput result;
            result.encoder_outputs = std::vector<float>(output_data, output_data + output_size);
            result.shape = output_shape;
            result.batch_size = output_shape[0];
            result.sequence_length = output_shape[1];
            result.hidden_size = output_shape[2];

            return result;

        } catch (const Ort::Exception &e) {
            std::cerr << "Batched Encoder ONNX Runtime error: " << e.what() << std::endl;
            throw;
        } catch (const std::exception &e) {
            std::cerr << "Batched Encoder error: " << e.what() << std::endl;
            throw;
        }
    }

    std::vector<float> runDecoderStep(const std::vector<int64_t> &decoder_input_ids,
                                      const std::vector<float> &encoder_output,
                                      const std::vector<int64_t> &attention_mask,
                                      const std::vector<int64_t> &decoder_shape,
                                      const std::vector<int64_t> &encoder_shape) {
        try {
            Ort::AllocatorWithDefaultOptions allocator;
            auto input_name_0 = decoder_session_->GetInputNameAllocated(0, allocator); // input_ids
            auto input_name_1 = decoder_session_->GetInputNameAllocated(1, allocator); // encoder_hidden_states
            auto input_name_2 = decoder_session_->GetInputNameAllocated(2, allocator); // attention_mask
            std::vector<const char *> input_names = {input_name_0.get(), input_name_1.get(), input_name_2.get()};

            auto output_name_0 = decoder_session_->GetOutputNameAllocated(0, allocator);
            std::vector<const char *> output_names = {output_name_0.get()};

            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value decoder_ids_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, const_cast<int64_t *>(decoder_input_ids.data()), decoder_input_ids.size(), decoder_shape.data(), decoder_shape.size());

            Ort::Value encoder_tensor = Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float *>(encoder_output.data()), encoder_output.size(), encoder_shape.data(), encoder_shape.size());

            std::vector<int64_t> attention_shape = {encoder_shape[0], encoder_shape[1]};
            Ort::Value attention_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, const_cast<int64_t *>(attention_mask.data()), attention_mask.size(), attention_shape.data(), attention_shape.size());

            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(decoder_ids_tensor));
            input_tensors.push_back(std::move(encoder_tensor));
            input_tensors.push_back(std::move(attention_tensor));

            auto output_tensors = decoder_session_->Run(
                Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_names.size(), output_names.data(), output_names.size());

            float *output_data = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

            size_t output_size = 1;
            for (auto dim : output_shape) {
                output_size *= dim;
            }

            return std::vector<float>(output_data, output_data + output_size);

        } catch (const Ort::Exception &e) {
            std::cerr << "Decoder ONNX Runtime error: " << e.what() << std::endl;
            return {};
        } catch (const std::exception &e) {
            std::cerr << "Decoder error: " << e.what() << std::endl;
            return {};
        }
    }

    std::vector<int> runDecoder(const std::vector<float> &encoder_output, const std::vector<int64_t> &attention_mask) {
        std::vector<int> generated_tokens;
        generated_tokens.push_back(config_.decoder_start_token_id);
        std::vector<int64_t> decoder_input_ids = {static_cast<int64_t>(config_.decoder_start_token_id)};

        std::cout << "Generating tokens..." << std::endl;

        // Generation loop
        for (int step = 0; step < config_.max_length; ++step) {
            std::vector<int64_t> decoder_shape = {1, static_cast<int64_t>(decoder_input_ids.size())};
            std::vector<int64_t> encoder_shape = {1, static_cast<int64_t>(attention_mask.size()), HIDDEN_SIZE};

            std::vector<float> decoder_output = runDecoderStep(decoder_input_ids, encoder_output, attention_mask, decoder_shape, encoder_shape);

            if (decoder_output.empty()) {
                std::cerr << "Decoder step failed!" << std::endl;
                break;
            }

            int seq_len = decoder_input_ids.size();

            std::vector<float> last_hidden(HIDDEN_SIZE);
            for (int i = 0; i < HIDDEN_SIZE; ++i) {
                last_hidden[i] = decoder_output[(seq_len - 1) * HIDDEN_SIZE + i];
            }

            std::vector<float> logits(config_.vocab_size, 0.0f);
#ifdef USE_SIMD
            compute_logits_simd(last_hidden, *lm_weights_, *lm_bias_, logits, config_.vocab_size, HIDDEN_SIZE);
#else
            compute_logits_optimized(last_hidden, *lm_weights_, *lm_bias_, logits, config_.vocab_size, HIDDEN_SIZE);
#endif

            // Find argmax (greedy search)
            int next_token_id = 0;
            float max_logit = logits[0];

            int i = 1;
            for (; i <= config_.vocab_size - 4; i += 4) {
                if (logits[i] > max_logit) {
                    max_logit = logits[i];
                    next_token_id = i;
                }
                if (logits[i + 1] > max_logit) {
                    max_logit = logits[i + 1];
                    next_token_id = i + 1;
                }
                if (logits[i + 2] > max_logit) {
                    max_logit = logits[i + 2];
                    next_token_id = i + 2;
                }
                if (logits[i + 3] > max_logit) {
                    max_logit = logits[i + 3];
                    next_token_id = i + 3;
                }
            }

            for (; i < config_.vocab_size; ++i) {
                if (logits[i] > max_logit) {
                    max_logit = logits[i];
                    next_token_id = i;
                }
            }

            generated_tokens.push_back(next_token_id);
            decoder_input_ids.push_back(static_cast<int64_t>(next_token_id));

            if (next_token_id == config_.eos_token_id) {
                break;
            }
        }

        return generated_tokens;
    }

    std::vector<std::vector<int>> runBatchedDecoder(const BatchedEncoderOutput &encoder_output, const BatchedInputs &batch_inputs) {
        std::cout << "\n--- Batched Decoder Generation ---" << std::endl;
        
        size_t batch_size = encoder_output.batch_size;
        std::vector<std::vector<int>> all_generated_tokens(batch_size);
        std::vector<std::vector<int64_t>> all_decoder_input_ids(batch_size);
        std::vector<bool> finished(batch_size, false);
        
        for (size_t i = 0; i < batch_size; ++i) {
            all_generated_tokens[i].push_back(config_.decoder_start_token_id);
            all_decoder_input_ids[i] = {static_cast<int64_t>(config_.decoder_start_token_id)};
        }

        for (int step = 0; step < config_.max_length; ++step) {
            bool all_finished = true;
            for (size_t i = 0; i < batch_size; ++i) {
                if (!finished[i]) {
                    all_finished = false;
                    break;
                }
            }
            if (all_finished) break;

            for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                if (finished[batch_idx]) continue;

                size_t encoder_seq_len = batch_inputs.original_lengths[batch_idx];
                size_t encoder_start_idx = batch_idx * encoder_output.sequence_length * encoder_output.hidden_size;
                
                std::vector<float> single_encoder_output;
                single_encoder_output.reserve(encoder_seq_len * encoder_output.hidden_size);
                
                for (size_t seq_pos = 0; seq_pos < encoder_seq_len; ++seq_pos) {
                    size_t pos_start = encoder_start_idx + seq_pos * encoder_output.hidden_size;
                    for (size_t h = 0; h < encoder_output.hidden_size; ++h) {
                        single_encoder_output.push_back(encoder_output.encoder_outputs[pos_start + h]);
                    }
                }

                std::vector<int64_t> attention_mask;
                attention_mask.reserve(encoder_seq_len);
                for (size_t i = 0; i < encoder_seq_len; ++i) {
                    attention_mask.push_back(1);
                }

                std::vector<int64_t> decoder_shape = {1, static_cast<int64_t>(all_decoder_input_ids[batch_idx].size())};
                std::vector<int64_t> encoder_shape = {1, static_cast<int64_t>(encoder_seq_len), static_cast<int64_t>(encoder_output.hidden_size)};

                std::vector<float> decoder_output = runDecoderStep(
                    all_decoder_input_ids[batch_idx], 
                    single_encoder_output, 
                    attention_mask, 
                    decoder_shape, 
                    encoder_shape
                );

                if (decoder_output.empty()) {
                    std::cerr << "Decoder step failed for batch " << batch_idx << "!" << std::endl;
                    finished[batch_idx] = true;
                    continue;
                }

                int seq_len = all_decoder_input_ids[batch_idx].size();
                std::vector<float> last_hidden(encoder_output.hidden_size);
                for (size_t i = 0; i < encoder_output.hidden_size; ++i) {
                    last_hidden[i] = decoder_output[(seq_len - 1) * encoder_output.hidden_size + i];
                }

                std::vector<float> logits(config_.vocab_size, 0.0f);
#ifdef USE_SIMD
                compute_logits_simd(last_hidden, *lm_weights_, *lm_bias_, logits, config_.vocab_size, encoder_output.hidden_size);
#else
                compute_logits_optimized(last_hidden, *lm_weights_, *lm_bias_, logits, config_.vocab_size, encoder_output.hidden_size);
#endif

                // Find argmax (greedy search)
                int next_token_id = 0;
                float max_logit = logits[0];
                for (int i = 1; i < config_.vocab_size; ++i) {
                    if (logits[i] > max_logit) {
                        max_logit = logits[i];
                        next_token_id = i;
                    }
                }

                all_generated_tokens[batch_idx].push_back(next_token_id);
                all_decoder_input_ids[batch_idx].push_back(static_cast<int64_t>(next_token_id));

                if (next_token_id == config_.eos_token_id) {
                    finished[batch_idx] = true;
                }
            }
        }

        std::cout << "Batched generation completed!" << std::endl;
        return all_generated_tokens;
    }

  public:
    TranslationModel(const std::string &model_dir) : model_dir_(model_dir), env_(ORT_LOGGING_LEVEL_WARNING, "MarianTranslationModel") {

        std::cout << "\n=== LOADING TRANSLATION MODEL ===" << std::endl;

        std::cout << "Loading vocabulary..." << std::endl;
        vocab_ = loadVocab(model_dir_ + "/vocab.json");
        if (vocab_.empty()) {
            throw std::runtime_error("Failed to load vocabulary");
        }
        buildReverseVocab();

        std::cout << "Loading configuration..." << std::endl;
        config_ = loadConfig(model_dir_ + "/config.json");
        if (config_.decoder_start_token_id == -1) {
            throw std::runtime_error("Failed to load configuration");
        }

        std::cout << "Loading weights using memory mapping..." << std::endl;
        lm_weights_ = std::make_unique<MemoryMappedWeights>(model_dir_ + "/lm_weight_raw.bin");
        lm_bias_ = std::make_unique<MemoryMappedWeights>(model_dir_ + "/lm_bias_raw.bin");
        if (!lm_weights_->isValid() || !lm_bias_->isValid()) {
            throw std::runtime_error("Failed to memory map weight files");
        }

        std::cout << "Loading SentencePiece model..." << std::endl;
        if (!sp_processor_.Load(model_dir_ + "/source.spm").ok()) {
            throw std::runtime_error("Failed to load SentencePiece model");
        }

        std::cout << "Initializing ONNX Runtime..." << std::endl;
        session_options_.SetIntraOpNumThreads(1);

        encoder_session_ = std::make_unique<Ort::Session>(env_, (model_dir_ + "/encoder.onnx").c_str(), session_options_);
        decoder_session_ = std::make_unique<Ort::Session>(env_, (model_dir_ + "/decoder.onnx").c_str(), session_options_);

        std::cout << "Translation model loaded successfully!" << std::endl;
    }

    std::vector<std::string> translate(const std::vector<std::string> &sentences) {
        std::cout << "\n=== BATCH TRANSLATION INFERENCE ===" << std::endl;
        
        if (sentences.empty()) {
            throw std::runtime_error("No sentences provided for translation");
        }
        
        std::cout << "Processing " << sentences.size() << " sentence(s):" << std::endl;
        for (size_t i = 0; i < sentences.size(); ++i) {
            std::cout << "  " << i + 1 << ": \"" << sentences[i] << "\"" << std::endl;
        }

        auto inference_start = std::chrono::high_resolution_clock::now();

        // Step 1: Create batch with tokenization and padding
        BatchedInputs batch = createBatch(sentences);

        // Step 2: Batched encoder inference
        auto encoder_start = std::chrono::high_resolution_clock::now();
        BatchedEncoderOutput encoder_output = runBatchedEncoder(batch);
        auto encoder_end = std::chrono::high_resolution_clock::now();
        double encoder_time = std::chrono::duration<double, std::milli>(encoder_end - encoder_start).count();
        std::cout << "Batched encoder completed in " << encoder_time << " ms" << std::endl;

        // Step 3: Batched decoder generation
        auto decoder_start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<int>> all_generated_tokens = runBatchedDecoder(encoder_output, batch);
        auto decoder_end = std::chrono::high_resolution_clock::now();
        auto inference_end = std::chrono::high_resolution_clock::now();

        double decoder_time = std::chrono::duration<double, std::milli>(decoder_end - decoder_start).count();
        double total_inference_time = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();

        // Step 4: Detokenize all results
        std::cout << "\n--- Final Results ---" << std::endl;
        std::vector<std::string> translated_sentences;
        translated_sentences.reserve(sentences.size());

        for (size_t i = 0; i < sentences.size(); ++i) {
            std::cout << "\nSentence " << i + 1 << ":" << std::endl;
            std::cout << "  Original: \"" << sentences[i] << "\"" << std::endl;
            
            std::cout << "  Generated tokens: ";
            for (int id : all_generated_tokens[i]) {
                std::cout << id << " ";
            }
            std::cout << std::endl;

            std::string translation = detokenize(all_generated_tokens[i]);
            translated_sentences.push_back(translation);
            std::cout << "  Translation: \"" << translation << "\"" << std::endl;
        }

        // Calculate performance metrics
        size_t total_generated_tokens = 0;
        for (const auto &tokens : all_generated_tokens) {
            total_generated_tokens += tokens.size() - 1; // Subtract start token
        }

        double tokens_per_second = total_generated_tokens / (total_inference_time / 1000.0);

        std::cout << "\n=== PERFORMANCE METRICS ===" << std::endl;
        std::cout << "Sentences processed: " << sentences.size() << std::endl;
        std::cout << "Encoder time: " << encoder_time << " ms" << std::endl;
        std::cout << "Decoder time: " << decoder_time << " ms" << std::endl;
        std::cout << "Total inference time: " << total_inference_time << " ms" << std::endl;
        std::cout << "Total generated tokens: " << total_generated_tokens << std::endl;
        std::cout << "Tokens per second: " << tokens_per_second << std::endl;
        std::cout << "Sentences per second: " << sentences.size() / (total_inference_time / 1000.0) << std::endl;

        return translated_sentences;
    }
};

int main(int argc, char *argv[]) {
    try {
        std::string input_text = "This is a test. How are you today? I hope everything is going well!";
        if (argc > 1 && strlen(argv[1]) > 0) {
            input_text = argv[1];
        }

        const std::string model_dir = "../outs/";
        TranslationModel model(model_dir);
        
        std::cout << "\n=== TRANSLATION MODEL LOADED ===" << std::endl;
        std::cout << "Usage: " << argv[0] << " \"<text with multiple sentences>\"" << std::endl;
        std::cout << "Note: The model will automatically split text on '.', '!', and '?' and process each sentence in a batch." << std::endl;
        
        // Step 1: Split input text into sentences
        std::cout << "\n=== SENTENCE SPLITTING ===" << std::endl;
        std::cout << "Input text: \"" << input_text << "\"" << std::endl;
        
        std::vector<std::string> sentences = splitIntoSentences(input_text);
        
        if (sentences.empty()) {
            throw std::runtime_error("No sentences found in input text");
        }
        
        std::cout << "Found " << sentences.size() << " sentence(s):" << std::endl;
        for (size_t i = 0; i < sentences.size(); ++i) {
            std::cout << "  " << i + 1 << ": \"" << sentences[i] << "\"" << std::endl;
        }
        
        // Step 2: Translate sentences
        std::vector<std::string> translations = model.translate(sentences);
        
        // Step 3: Combine translated sentences
        std::cout << "\n=== COMBINING TRANSLATIONS ===" << std::endl;
        std::string final_result;
        for (size_t i = 0; i < translations.size(); ++i) {
            if (i > 0) final_result += " ";
            final_result += translations[i];
        }
        
        std::cout << "Final combined translation: \"" << final_result << "\"" << std::endl;
        
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}