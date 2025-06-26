import os
import shutil

import torch
import numpy as np

from transformers import MarianMTModel, MarianTokenizer

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from core.layers import MarianDecoder, MarianEncoder
from core.quantize import quantize


def create_model_for_provider(path: str, provider: str):
    """Create an ONNX Runtime session

    Args:
        path (str): Path to an onnx model
        provider (str): An ONNX Runtime provider ['CPUExecutionProvider', 'CUDAExecutionProvider]

    Returns:
        session: an ONNX Runtime session
    """

    # Few properties that might have an impact on performances
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(path, options, providers=[provider])
    session.disable_fallback()

    return session


def save_tensor_as_raw_binary(tensor, filepath):
    """Save a PyTorch tensor as raw binary file for C++ memory mapping
    
    Args:
        tensor: PyTorch tensor to save
        filepath: Path where to save the raw binary file
    """
    tensor_np = tensor.detach().cpu().numpy().astype(np.float32)
    if len(tensor_np.shape) > 1 and tensor_np.shape[0] == 1:
        tensor_np = tensor_np.flatten()
    tensor_np.tofile(filepath)


def create_marian_encoder_decoder(model_path: str, outdir: str):
    """Create a MarianMTModel encoder & decoder

    Args:
        model_path (str): Path to a Marian model
        outdir (str): Output directory

    Returns the MarianMTModel encoder & decoder
    """
    model = MarianMTModel.from_pretrained(model_path)

    encoder = model.get_encoder()
    decoder = model.get_decoder()

    marian_encoder = MarianEncoder(encoder).eval()
    marian_decoder = MarianDecoder(decoder).eval()

    torch.save(model.model.shared.weight, os.path.join(outdir, 'lm_weight.bin'))
    torch.save(model.final_logits_bias, os.path.join(outdir, 'lm_bias.bin'))
    
    print("Saving raw binary weights for C++ compatibility...")
    save_tensor_as_raw_binary(model.model.shared.weight, os.path.join(outdir, 'lm_weight_raw.bin'))
    save_tensor_as_raw_binary(model.final_logits_bias, os.path.join(outdir, 'lm_bias_raw.bin'))

    for file in ['config.json', 'source.spm', 'target.spm', 'tokenizer_config.json', 'vocab.json']:
        shutil.copyfile(os.path.join(model_path, file), os.path.join(outdir, file))

    return marian_encoder, marian_decoder


def generate_onnx_graph(model_path, encoder_path, decoder_path, outdir, quant=True):
    encoder, decoder = create_marian_encoder_decoder(model_path, outdir)

    # Exemple sequence
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    inputs = tokenizer("Hello World !", return_tensors="pt")
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

    print("Exporting encoder to ONNX...")
    # Exports to ONNX
    torch.onnx.export(
        encoder,
        (input_ids, attention_mask),
        encoder_path,
        export_params=True,
        opset_version=12,
        input_names=["input_ids", "attention_mask"],
        output_names=["encoder_hidden_states"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
        }
    )
    if quant:
        quantize(encoder_path)

    print("Exporting decoder to ONNX...")
    encoder_hidden_states = encoder(input_ids, attention_mask)[0]
    torch.onnx.export(
        decoder,
        (input_ids, encoder_hidden_states, attention_mask),
        decoder_path,
        export_params=True,
        opset_version=14,
        input_names=["input_ids", "encoder_hidden_states", "attention_mask"],
        output_names=["decoder_output"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
            "decoder_output": {0: "batch", 1: "sequence"},
        }
    )
    if quant:
        quantize(decoder_path)
