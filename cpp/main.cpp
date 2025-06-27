#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <cctype>
#include <memory>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sentencepiece_processor.h>
#include <onnxruntime_cxx_api.h>
#include <chrono>

// Configuration constants
const std::string OUTPUT_DIR = "../outs/";

std::string decodeUnicodeEscapes(const std::string& input) {
    std::string result;
    size_t pos = 0;
    
    while (pos < input.length()) {
        size_t unicode_pos = input.find("\\u", pos);
        if (unicode_pos == std::string::npos) {
            result += input.substr(pos);
            break;
        }
        
        result += input.substr(pos, unicode_pos - pos);
        
        if (unicode_pos + 6 <= input.length()) {
            std::string hex_code = input.substr(unicode_pos + 2, 4);
            
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
                    pos = unicode_pos + 6;
                } catch (const std::exception&) {
                    // If conversion fails, keep the original escape sequence
                    result += input.substr(unicode_pos, 6);
                    pos = unicode_pos + 6;
                }
            } else {
                // Invalid hex characters, keep the original escape sequence
                result += input.substr(unicode_pos, 6);
                pos = unicode_pos + 6;
            }
        } else {
            // Not enough characters for a full escape, keep as-is
            result += input.substr(unicode_pos);
            break;
        }
    }
    
    return result;
}

std::unordered_map<std::string, int> loadVocab(const std::string& vocab_path) {
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
        if (key_start == std::string::npos) break;
        key_start++;
        
        size_t key_end = line.find('"', key_start);
        if (key_end == std::string::npos) break;
        
        std::string key = line.substr(key_start, key_end - key_start);
        
        key = decodeUnicodeEscapes(key);
        
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

std::string decodeIds(const std::vector<int>& ids, const std::unordered_map<std::string, int>& vocab) {
    // Create reverse mapping (ID -> token)
    static std::unordered_map<int, std::string> id_to_token;
    if (id_to_token.empty()) {
        for (const auto& pair : vocab) {
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
            
            token = decodeUnicodeEscapes(token);
            
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

MarianConfig loadConfig(const std::string& config_path) {
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
    
    auto findValue = [&](const std::string& key) -> int {
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
    MemoryMappedWeights(const std::string& path) : fd_(-1), data_(nullptr), size_(0) {
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
        
        // Use MAP_POPULATE to prefault pages and MAP_HUGETLB for large pages if available
        int flags = MAP_PRIVATE | MAP_POPULATE;
        
        data_ = static_cast<float*>(mmap(nullptr, size_, PROT_READ, flags, fd_, 0));
        if (data_ == MAP_FAILED) {
            // Fallback without MAP_POPULATE if it fails
            data_ = static_cast<float*>(mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
            if (data_ == MAP_FAILED) {
                std::cerr << "Failed to memory map file: " << path << std::endl;
                close(fd_);
                fd_ = -1;
                data_ = nullptr;
                return;
            }
        }
        
        // Advise the kernel about our access pattern
        if (madvise(data_, size_, MADV_SEQUENTIAL | MADV_WILLNEED) != 0) {
            // Non-critical, continue anyway
        }
        
        // Pre-fault the first few pages to reduce initial latency
        volatile float prefetch_sum = 0;
        size_t prefetch_size = std::min(size_, static_cast<size_t>(64 * 1024)); // 64KB
        for (size_t i = 0; i < prefetch_size / sizeof(float); i += 16) {
            prefetch_sum += data_[i];
        }
        
        std::cout << "Memory mapped " << (size_ / sizeof(float)) << " weights from " << path 
                  << " (size: " << size_ / (1024*1024) << " MB)" << std::endl;
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
    MemoryMappedWeights(const MemoryMappedWeights&) = delete;
    MemoryMappedWeights& operator=(const MemoryMappedWeights&) = delete;
    
    // Movable
    MemoryMappedWeights(MemoryMappedWeights&& other) noexcept 
        : fd_(other.fd_), data_(other.data_), size_(other.size_) {
        other.fd_ = -1;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    const float* data() const { return data_; }
    size_t size() const { return size_ / sizeof(float); }
    bool isValid() const { return data_ != nullptr && fd_ != -1; }
    
    // Optimized access with prefetching
    inline const float& operator[](size_t index) const {
        return data_[index];
    }
    
    // Batch access for better cache performance
    inline void prefetch_range(size_t start_idx, size_t count) const {
        const char* start_addr = reinterpret_cast<const char*>(data_ + start_idx);
        const char* end_addr = reinterpret_cast<const char*>(data_ + start_idx + count);
        
        // Prefetch cache lines
        for (const char* addr = start_addr; addr < end_addr; addr += 64) {
            __builtin_prefetch(addr, 0, 3); // Read, high temporal locality
        }
    }
    
    // Get aligned pointer for SIMD operations
    inline const float* get_aligned_ptr(size_t offset = 0) const {
        return data_ + offset;
    }
    
private:
    int fd_;
    float* data_;
    size_t size_;
};

std::vector<int> getIds(std::string sentence, const std::unordered_map<std::string, int>& vocab) {
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
    for (const auto& piece : pieces) {
        std::cout << "'" << piece << "' ";
    }
    std::cout << std::endl;
    
    std::vector<int> ids;
    for (const auto& piece : pieces) {
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
        ids.push_back(eos_it->second);  // Should be 0
    }
    
    std::cout << "\tIDs: ";
    for (const auto& id : ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    
    std::string decoded_text = decodeIds(ids, vocab);
    std::cout << "\tDecoded text: " << decoded_text << std::endl;

    return ids;
}

std::vector<int64_t> createAttentionMask(const std::vector<int>& input_ids) {
    std::vector<int64_t> attention_mask;
    attention_mask.reserve(input_ids.size());
    
    for (int id : input_ids) {
        attention_mask.push_back(1);
    }
    
    return attention_mask;
}

std::vector<int64_t> convertToInt64(const std::vector<int>& input) {
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
    std::vector<int64_t> input_shape;  // [batch_size, sequence_length]
};

EncoderInputs prepareEncoderInputs(const std::vector<int>& token_ids) {
    EncoderInputs inputs;
    
    inputs.input_ids = convertToInt64(token_ids);
    inputs.attention_mask = createAttentionMask(token_ids);
    inputs.input_shape = {1, static_cast<int64_t>(token_ids.size())};
    
    return inputs;
}

std::vector<float> runEncoderInference(const EncoderInputs& inputs) {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MarianEncoder");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        
        const std::string encoder_path = OUTPUT_DIR + "encoder.onnx";
        Ort::Session encoder_session(env, encoder_path.c_str(), session_options);
        
        Ort::AllocatorWithDefaultOptions allocator;
        
        auto input_name_0 = encoder_session.GetInputNameAllocated(0, allocator);
        auto input_name_1 = encoder_session.GetInputNameAllocated(1, allocator);
        std::vector<const char*> input_names = {input_name_0.get(), input_name_1.get()};
        
        auto output_name_0 = encoder_session.GetOutputNameAllocated(0, allocator);
        std::vector<const char*> output_names = {output_name_0.get()};
        
        std::cout << "Encoder model loaded successfully" << std::endl;
        std::cout << "  Input 0: " << input_names[0] << std::endl;
        std::cout << "  Input 1: " << input_names[1] << std::endl;
        std::cout << "  Output 0: " << output_names[0] << std::endl;
        
        std::vector<int64_t> input_shape = inputs.input_shape;
        
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, 
            const_cast<int64_t*>(inputs.input_ids.data()), 
            inputs.input_ids.size(),
            input_shape.data(), 
            input_shape.size()
        );
        
        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, 
            const_cast<int64_t*>(inputs.attention_mask.data()), 
            inputs.attention_mask.size(),
            input_shape.data(), 
            input_shape.size()
        );
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_ids_tensor));
        input_tensors.push_back(std::move(attention_mask_tensor));
        
        std::cout << "Running encoder inference..." << std::endl;
        
        auto output_tensors = encoder_session.Run(
            Ort::RunOptions{nullptr}, 
            input_names.data(), 
            input_tensors.data(), 
            input_names.size(), 
            output_names.data(), 
            output_names.size()
        );
        
        std::cout << "Encoder inference completed!" << std::endl;
        
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        std::cout << "Encoder output shape: [";
        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::cout << output_shape[i];
            if (i < output_shape.size() - 1) std::cout << ", ";
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
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return {};
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return {};
    }
}

std::vector<float> runDecoderStep(
    const std::vector<int64_t>& decoder_input_ids,
    const std::vector<float>& encoder_output,
    const std::vector<int64_t>& attention_mask,
    const std::vector<int64_t>& decoder_shape,
    const std::vector<int64_t>& encoder_shape,
    Ort::Session& decoder_session
) {
    try {
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_0 = decoder_session.GetInputNameAllocated(0, allocator);  // input_ids
        auto input_name_1 = decoder_session.GetInputNameAllocated(1, allocator);  // encoder_hidden_states  
        auto input_name_2 = decoder_session.GetInputNameAllocated(2, allocator);  // attention_mask
        std::vector<const char*> input_names = {input_name_0.get(), input_name_1.get(), input_name_2.get()};
        
        auto output_name_0 = decoder_session.GetOutputNameAllocated(0, allocator);
        std::vector<const char*> output_names = {output_name_0.get()};
        
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value decoder_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            const_cast<int64_t*>(decoder_input_ids.data()),
            decoder_input_ids.size(),
            decoder_shape.data(),
            decoder_shape.size()
        );
        
        Ort::Value encoder_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(encoder_output.data()),
            encoder_output.size(),
            encoder_shape.data(),
            encoder_shape.size()
        );
        
        std::vector<int64_t> attention_shape = {encoder_shape[0], encoder_shape[1]};
        Ort::Value attention_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            const_cast<int64_t*>(attention_mask.data()),
            attention_mask.size(),
            attention_shape.data(),
            attention_shape.size()
        );
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(decoder_ids_tensor));
        input_tensors.push_back(std::move(encoder_tensor));
        input_tensors.push_back(std::move(attention_tensor));
        
        auto output_tensors = decoder_session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_names.size(),
            output_names.data(),
            output_names.size()
        );
        
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        size_t output_size = 1;
        for (auto dim : output_shape) {
            output_size *= dim;
        }
        
        return std::vector<float>(output_data, output_data + output_size);
        
    } catch (const Ort::Exception& e) {
        std::cerr << "Decoder ONNX Runtime error: " << e.what() << std::endl;
        return {};
    } catch (const std::exception& e) {
        std::cerr << "Decoder error: " << e.what() << std::endl;
        return {};
    }
}

void compute_logits_optimized(const std::vector<float>& hidden_state,
                             const MemoryMappedWeights& weights,
                             const MemoryMappedWeights& bias,
                             std::vector<float>& logits,
                             int vocab_size,
                             int hidden_size) {
    
    const float* hidden_ptr = hidden_state.data();
    const float* weights_ptr = weights.get_aligned_ptr();
    const float* bias_ptr = bias.get_aligned_ptr();
    
    // Process in blocks for better cache performance
    constexpr int BLOCK_SIZE = 64;
    
    for (int block_start = 0; block_start < vocab_size; block_start += BLOCK_SIZE) {
        int block_end = std::min(block_start + BLOCK_SIZE, vocab_size);
        
        if (block_end < vocab_size) {
            weights.prefetch_range(block_end * hidden_size, BLOCK_SIZE * hidden_size);
            bias.prefetch_range(block_end, BLOCK_SIZE);
        }
        
        for (int i = block_start; i < block_end; ++i) {
            const float* weight_row = weights_ptr + i * hidden_size;
            
            float sum = bias_ptr[i];
            
            int j = 0;
            
            for (; j <= hidden_size - 8; j += 8) {
                sum += hidden_ptr[j] * weight_row[j];
                sum += hidden_ptr[j+1] * weight_row[j+1];
                sum += hidden_ptr[j+2] * weight_row[j+2];
                sum += hidden_ptr[j+3] * weight_row[j+3];
                sum += hidden_ptr[j+4] * weight_row[j+4];
                sum += hidden_ptr[j+5] * weight_row[j+5];
                sum += hidden_ptr[j+6] * weight_row[j+6];
                sum += hidden_ptr[j+7] * weight_row[j+7];
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

void compute_logits_simd(const std::vector<float>& hidden_state,
                        const MemoryMappedWeights& weights,
                        const MemoryMappedWeights& bias,
                        std::vector<float>& logits,
                        int vocab_size,
                        int hidden_size) {
    
    const float* hidden_ptr = hidden_state.data();
    const float* weights_ptr = weights.get_aligned_ptr();
    const float* bias_ptr = bias.get_aligned_ptr();
    
    for (int i = 0; i < vocab_size; ++i) {
        const float* weight_row = weights_ptr + i * hidden_size;
        
        __m256 sum_vec = _mm256_setzero_ps();
        
        int j = 0;
        for (; j <= hidden_size - 8; j += 8) {
            __m256 hidden_vec = _mm256_loadu_ps(&hidden_ptr[j]);
            __m256 weight_vec = _mm256_loadu_ps(&weight_row[j]);
            sum_vec = _mm256_fmadd_ps(hidden_vec, weight_vec, sum_vec);
        }
        
        float sum_array[8];
        _mm256_storeu_ps(sum_array, sum_vec);
        float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                   sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
        
        for (; j < hidden_size; ++j) {
            sum += hidden_ptr[j] * weight_row[j];
        }
        
        logits[i] = sum + bias_ptr[i];
    }
}
#endif

int main(int argc, char* argv[]) {    
    std::string sentence = "This is a test";
    if (argc > 1 && strlen(argv[1]) > 0) {
        sentence = argv[1];
    }    
    std::cout << "Input sentence: \"" << sentence << "\"" << std::endl;

    std::cout << "\n=== LOADING PHASE ===" << std::endl;
    
    std::cout << "Loading vocabulary..." << std::endl;
    std::unordered_map<std::string, int> vocab = loadVocab(OUTPUT_DIR + "vocab.json");
    if (vocab.empty()) {
        std::cerr << "Failed to load vocabulary" << std::endl;
        return -1;
    }
    
    std::cout << "Loading configuration..." << std::endl;
    MarianConfig config = loadConfig(OUTPUT_DIR + "config.json");
    if (config.decoder_start_token_id == -1) {
        std::cerr << "Failed to load configuration!" << std::endl;
        return -1;
    }
    
    std::cout << "Loading weights using memory mapping..." << std::endl;
    MemoryMappedWeights lm_weights(OUTPUT_DIR + "lm_weight_raw.bin");
    MemoryMappedWeights lm_bias(OUTPUT_DIR + "lm_bias_raw.bin");
    if (!lm_weights.isValid() || !lm_bias.isValid()) {
        std::cerr << "Failed to memory map weight files!" << std::endl;
        return -1;
    }
    
    std::cout << "Initializing ONNX Runtime..." << std::endl;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MarianInference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    
    const std::string encoder_path = OUTPUT_DIR + "encoder.onnx";
    Ort::Session encoder_session(env, encoder_path.c_str(), session_options);
    
    const std::string decoder_path = OUTPUT_DIR + "decoder.onnx";
    std::ifstream decoder_file(decoder_path);
    if (!decoder_file.is_open()) {
        std::cerr << "Cannot open decoder model file: " << decoder_path << std::endl;
        return -1;
    }
    decoder_file.close();
    Ort::Session decoder_session(env, decoder_path.c_str(), session_options);
    
    std::cout << "All models and data loaded successfully!" << std::endl;
    
    std::cout << "\n=== INFERENCE PHASE ===" << std::endl;
    auto inference_start = std::chrono::high_resolution_clock::now();
    
    // Step 1: Tokenize the sentence
    std::cout << "\n--- Tokenization ---" << std::endl;
    std::vector<int> ids = getIds(sentence, vocab);
    if (ids.empty()) {
        std::cerr << "Failed to tokenize sentence" << std::endl;
        return -1;
    }
    
    // Step 2: Prepare inputs for encoder
    EncoderInputs encoder_inputs = prepareEncoderInputs(ids);
    std::cout << "Prepared encoder inputs: [" << encoder_inputs.input_shape[0] 
              << ", " << encoder_inputs.input_shape[1] << "]" << std::endl;
    
    // Step 3: Run encoder inference
    std::cout << "\n--- Encoder Inference ---" << std::endl;
    auto encoder_start = std::chrono::high_resolution_clock::now();
    
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_0 = encoder_session.GetInputNameAllocated(0, allocator);
    auto input_name_1 = encoder_session.GetInputNameAllocated(1, allocator);
    std::vector<const char*> encoder_input_names = {input_name_0.get(), input_name_1.get()};
    auto output_name_0 = encoder_session.GetOutputNameAllocated(0, allocator);
    std::vector<const char*> encoder_output_names = {output_name_0.get()};
    
    std::vector<int64_t> input_shape = encoder_inputs.input_shape;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, 
        const_cast<int64_t*>(encoder_inputs.input_ids.data()), 
        encoder_inputs.input_ids.size(),
        input_shape.data(), 
        input_shape.size()
    );
    
    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, 
        const_cast<int64_t*>(encoder_inputs.attention_mask.data()), 
        encoder_inputs.attention_mask.size(),
        input_shape.data(), 
        input_shape.size()
    );
    
    std::vector<Ort::Value> encoder_input_tensors;
    encoder_input_tensors.push_back(std::move(input_ids_tensor));
    encoder_input_tensors.push_back(std::move(attention_mask_tensor));
    
    auto encoder_output_tensors = encoder_session.Run(
        Ort::RunOptions{nullptr}, 
        encoder_input_names.data(), 
        encoder_input_tensors.data(), 
        encoder_input_names.size(), 
        encoder_output_names.data(), 
        encoder_output_names.size()
    );
    
    float* encoder_output_data = encoder_output_tensors[0].GetTensorMutableData<float>();
    auto encoder_output_shape = encoder_output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t encoder_output_size = 1;
    for (auto dim : encoder_output_shape) {
        encoder_output_size *= dim;
    }
    std::vector<float> encoder_output(encoder_output_data, encoder_output_data + encoder_output_size);
    
    auto encoder_end = std::chrono::high_resolution_clock::now();
    double encoder_time = std::chrono::duration<double, std::milli>(encoder_end - encoder_start).count();
    std::cout << "Encoder completed in " << encoder_time << " ms" << std::endl;
    
    // Step 4: Run decoder inference and generation
    std::cout << "\n--- Decoder Generation ---" << std::endl;
    auto decoder_start = std::chrono::high_resolution_clock::now();
    
    std::vector<int> generated_tokens;
    generated_tokens.push_back(config.decoder_start_token_id);
    std::vector<int64_t> decoder_input_ids = {static_cast<int64_t>(config.decoder_start_token_id)};
    
    std::cout << "Generating tokens..." << std::endl;
    
    // Generation loop
    for (int step = 0; step < config.max_length; ++step) {
        std::vector<int64_t> decoder_shape = {1, static_cast<int64_t>(decoder_input_ids.size())};
        std::vector<int64_t> encoder_shape = {1, static_cast<int64_t>(encoder_inputs.input_ids.size()), 512};
        
        std::vector<float> decoder_output = runDecoderStep(
            decoder_input_ids,
            encoder_output,
            encoder_inputs.attention_mask,
            decoder_shape,
            encoder_shape,
            decoder_session
        );
        
        if (decoder_output.empty()) {
            std::cerr << "Decoder step failed!" << std::endl;
            break;
        }
        
        int hidden_size = 512;
        int seq_len = decoder_input_ids.size();
        
        std::vector<float> last_hidden(hidden_size);
        for (int i = 0; i < hidden_size; ++i) {
            last_hidden[i] = decoder_output[(seq_len - 1) * hidden_size + i];
        }
        
        std::vector<float> logits(config.vocab_size, 0.0f);
#ifdef USE_SIMD
        compute_logits_simd(last_hidden, lm_weights, lm_bias, logits, config.vocab_size, hidden_size);
#else
        compute_logits_optimized(last_hidden, lm_weights, lm_bias, logits, config.vocab_size, hidden_size);
#endif
        /* for (int i = 0; i < config.vocab_size; ++i) {
            float sum = lm_bias[i];  // Start with bias
            for (int j = 0; j < hidden_size; ++j) {
                sum += last_hidden[j] * lm_weights[i * hidden_size + j];
            }
            logits[i] = sum;
        } */
        
        // Find argmax (greedy search)
        int next_token_id = 0;
        float max_logit = logits[0];
        
        int i = 1;
        for (; i <= config.vocab_size - 4; i += 4) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                next_token_id = i;
            }
            if (logits[i+1] > max_logit) {
                max_logit = logits[i+1];
                next_token_id = i+1;
            }
            if (logits[i+2] > max_logit) {
                max_logit = logits[i+2];
                next_token_id = i+2;
            }
            if (logits[i+3] > max_logit) {
                max_logit = logits[i+3];
                next_token_id = i+3;
            }
        }
        
        for (; i < config.vocab_size; ++i) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                next_token_id = i;
            }
        }
        
        generated_tokens.push_back(next_token_id);
        decoder_input_ids.push_back(static_cast<int64_t>(next_token_id));
        
        if (next_token_id == config.eos_token_id) {
            break;
        }
    }
    
    auto decoder_end = std::chrono::high_resolution_clock::now();
    auto inference_end = std::chrono::high_resolution_clock::now();
    
    double decoder_time = std::chrono::duration<double, std::milli>(decoder_end - decoder_start).count();
    double total_inference_time = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
    double tokens_per_second = (generated_tokens.size() - 1) / (total_inference_time / 1000.0); // Exclude start token
    
    std::cout << "\n=== PERFORMANCE METRICS ===" << std::endl;
    std::cout << "Encoder time: " << encoder_time << " ms" << std::endl;
    std::cout << "Decoder time: " << decoder_time << " ms" << std::endl;
    std::cout << "Total inference time: " << total_inference_time << " ms" << std::endl;
    std::cout << "Generated tokens: " << (generated_tokens.size() - 1) << std::endl;
    std::cout << "Tokens per second: " << tokens_per_second << std::endl;
    
    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "Generated token IDs: ";
    for (int id : generated_tokens) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    
    std::vector<int> generated_int_tokens(generated_tokens.begin(), generated_tokens.end());
    std::string final_translation = decodeIds(generated_int_tokens, vocab);
    std::cout << "Final translation: " << final_translation << std::endl;
    
    return 0;
}