import argparse
from transformers import MarianTokenizer
from core.marian import MarianOnnx

def main():
    parser = argparse.ArgumentParser(description='Marian ONNX Translation')
    parser.add_argument('sentences', nargs='*', default=['This is a test'], 
                       help='Input sentence(s) to translate (default: "This is a test")')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], 
                       help='Device to use for inference (default: cpu)')
    parser.add_argument('--model-path', default='./outs-en-it', 
                       help='Path to the model directory (default: ./outs-en-it)')
    
    args = parser.parse_args()

    sentences = args.sentences if args.sentences else ['This is a test']
    
    if len(sentences) == 1:
        print(f'Input sentence: "{sentences[0]}"')
    else:
        print(f'Input sentences ({len(sentences)}):')
        for i, sentence in enumerate(sentences, 1):
            print(f'  {i}: "{sentence}"')
    
    tokenizer = MarianTokenizer.from_pretrained(args.model_path)
    input_ids = tokenizer(sentences, return_tensors='pt', padding=True).to(args.device)
    
    model = MarianOnnx(args.model_path, device=args.device)
    tokens = model.generate(**input_ids)
    results = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
    if len(results) == 1:
        print(f"Final translation: {results[0]}")
    else:
        print(f"Final translations:")
        for i, result in enumerate(results, 1):
            print(f"  {i}: {result}")

if __name__ == "__main__":
    main()