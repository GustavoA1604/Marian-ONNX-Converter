#include "TranslationModel.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cctype>

std::vector<std::string> splitIntoSentences(const std::string &text) {
    std::vector<std::string> sentences;
    std::string currentSentence;
    
    for (size_t i = 0; i < text.length(); ++i) {
        char c = text[i];
        currentSentence += c;
        
        if (c == '.' || c == '!' || c == '?') {
            bool isSentenceEnd = true;
            
            size_t nextPos = i + 1;
            while (nextPos < text.length() && std::isspace(text[nextPos])) {
                currentSentence += text[nextPos];
                nextPos++;
            }
            
            if (isSentenceEnd && !currentSentence.empty()) {
                std::string trimmed = currentSentence;
                while (!trimmed.empty() && std::isspace(trimmed.front())) {
                    trimmed = trimmed.substr(1);
                }
                while (!trimmed.empty() && std::isspace(trimmed.back())) {
                    trimmed.pop_back();
                }
                
                if (!trimmed.empty()) {
                    sentences.push_back(trimmed);
                }
                currentSentence.clear();
                
                i = nextPos - 1;
            }
        }
    }
    
    if (!currentSentence.empty()) {
        std::string trimmed = currentSentence;
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

std::string joinSentences(const std::vector<std::string> &sentences) {
    std::string result;
    for (size_t i = 0; i < sentences.size(); ++i) {
        if (i > 0) {
            result += " ";
        }
        result += sentences[i];
    }
    return result;
}

void printUsage(const char* programName) {
    std::cout << "\n=== Marian ONNX Translation Model ===" << std::endl;
    std::cout << "Usage: " << programName << " [OPTIONS] \"<Text to translate>\"" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --single    Use single-sentence translation mode" << std::endl;
    std::cout << "  --batch     Use batch translation mode (default)" << std::endl;
    std::cout << "  --help      Show this help message" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << programName << " \"Hello world!\"" << std::endl;
    std::cout << "  " << programName << " \"This is a test. How are you? I hope everything is well!\"" << std::endl;
    std::cout << "  " << programName << " --single \"Single sentence translation.\"" << std::endl;
    std::cout << "\nNote: " << std::endl;
    std::cout << "  - This model translates according to the language specified in modelDir" << std::endl;
    std::cout << "  - In batch mode, text is automatically split on '.', '!', and '?' and processed efficiently" << std::endl;
}

int main(int argc, char *argv[]) {
    try {
        std::string inputText = "This is a test";
        bool useSingleMode = false;
        bool showHelp = false;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            
            if (arg == "--help" || arg == "-h") {
                showHelp = true;
                break;
            } else if (arg == "--single") {
                useSingleMode = true;
            } else if (arg == "--batch") {
                useSingleMode = false;
            } else if (arg.empty() || arg[0] == '-') {
                std::cerr << "Unknown option: " << arg << std::endl;
                printUsage(argv[0]);
                return -1;
            } else {
                inputText = arg;
            }
        }

        if (showHelp) {
            printUsage(argv[0]);
            return 0;
        }

        if (inputText.empty()) {
            std::cerr << "Error: No input text provided" << std::endl;
            printUsage(argv[0]);
            return -1;
        }

        const std::string modelDir = "../outs/";
        std::cout << "\n=== INITIALIZING TRANSLATION MODEL ===" << std::endl;
        std::cout << "Model directory: " << modelDir << std::endl;
        std::cout << "Model type: English → Italian" << std::endl;
        std::cout << "Input text: \"" << inputText << "\"" << std::endl;
        std::cout << "Translation mode: " << (useSingleMode ? "Single" : "Batch") << std::endl;

        TranslationModel model(modelDir);
        
        std::string finalResult;

        if (useSingleMode) {
            std::cout << "\n=== SINGLE SENTENCE MODE ===" << std::endl;
            finalResult = model.translateSingle(inputText);
        } else {
            std::cout << "\n=== BATCH TRANSLATION MODE ===" << std::endl;
            
            std::cout << "\n--- Sentence Splitting ---" << std::endl;
            std::cout << "Input text: \"" << inputText << "\"" << std::endl;
            
            std::vector<std::string> sentences = splitIntoSentences(inputText);
            
            if (sentences.empty()) {
                throw std::runtime_error("No sentences found in input text");
            }
            
            std::cout << "Found " << sentences.size() << " sentence(s):" << std::endl;
            for (size_t i = 0; i < sentences.size(); ++i) {
                std::cout << "  " << i + 1 << ": \"" << sentences[i] << "\"" << std::endl;
            }
            
            std::vector<std::string> translations = model.translate(sentences);
            
            std::cout << "\n--- Combining Translations ---" << std::endl;
            finalResult = joinSentences(translations);
        }
        
        std::cout << "\n=== FINAL RESULT ===" << std::endl;
        std::cout << "Original: \"" << inputText << "\"" << std::endl;
        std::cout << "Translation: \"" << finalResult << "\"" << std::endl;
        
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        std::cerr << "Please check that:" << std::endl;
        std::cerr << "  - Model files exist in ../outs/ directory" << std::endl;
        std::cerr << "  - ONNX Runtime and SentencePiece libraries are properly installed" << std::endl;
        std::cerr << "  - Input text is valid and not empty" << std::endl;
        return -1;
    }
}