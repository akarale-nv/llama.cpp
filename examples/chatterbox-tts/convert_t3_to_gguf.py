#!/usr/bin/env python3
"""
Unified converter for T3 models to GGUF format.
Outputs two files:
1. T3 Preprocessing GGUF (Embeddings + Tokenizer)
2. Llama Backbone GGUF (Quantized Model)
"""

#pip install gguf
#pip install safetensors
#pip install transformers
#pip install torch

import sys
import argparse
import os
import json
from pathlib import Path
import numpy as np
import torch

try:
    import gguf
except ImportError:
    print("Error: gguf-py is not installed. Install it with: pip install gguf")
    sys.exit(1)

try:
    from safetensors.torch import load_file
except ImportError:
    print("Error: safetensors is not installed. Install it with: pip install safetensors")
    sys.exit(1)

try:
    from transformers import AutoConfig
except ImportError:
    print("Error: transformers is not installed. Install it with: pip install transformers")
    sys.exit(1)


# =============================================================================
# Helper Functions
# =============================================================================

def load_model_file(model_path):
    """Load model from .pth or .safetensors file."""
    print(f"Loading model from {model_path}...")
    
    if str(model_path).endswith('.safetensors'):
        state_dict = load_file(str(model_path))
    elif str(model_path).endswith('.pth') or str(model_path).endswith('.pt'):
        state_dict = torch.load(str(model_path), map_location='cpu')
    else:
        print(f"Error: Unsupported file format. Use .pth, .pt, or .safetensors")
        sys.exit(1)
    
    # If it's a full model (not just state dict), extract the state dict
    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    
    # Remove any 'module.' prefix from DataParallel
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
    
    print(f"Loaded {len(clean_state_dict)} tensors from model")
    return clean_state_dict

def save_tensor(gguf_writer, name, tensor, quantization_type=None, all_f32=False, force_f16=False):
    """Save a single tensor to GGUF with optional quantization"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    if all_f32:
        # Force float32
        gguf_writer.add_tensor(name, tensor.astype(np.float32))
    elif force_f16:
        # Force float16
        if tensor.dtype != np.float16:
            tensor = tensor.astype(np.float16)
        gguf_writer.add_tensor(name, tensor)
    else:
        data_type = gguf.GGMLQuantizationType.F32
        if tensor.dtype != np.float16 and not "bias" in name and not "emb" in name and not "norm" in name:
            tensor = tensor.astype(np.float16)
            data_type = gguf.GGMLQuantizationType.F16
        
         # Apply quantization if specified and tensor is suitable for quantization
        if quantization_type and quantization_type != "f16" and tensor.size > 1024:
            # Only quantize larger tensors (weights), not small ones like biases
            # Skip output layer to avoid dequantization overhead during logits extraction
            if not any(skip_name in name.lower() for skip_name in ["bias", "emb", "norm", "pos_emb", "output"]):
                try:
                    print(f"Quantizing {name} with {quantization_type}")
                    if quantization_type == "q4_0":
                        data_type = gguf.GGMLQuantizationType.Q4_0
                    elif quantization_type == "q4_1":
                        data_type = gguf.GGMLQuantizationType.Q4_1
                    elif quantization_type == "q5_0":
                        data_type = gguf.GGMLQuantizationType.Q5_0
                    elif quantization_type == "q5_1":
                        data_type = gguf.GGMLQuantizationType.Q5_1
                    elif quantization_type == "q8_0":
                        data_type = gguf.GGMLQuantizationType.Q8_0
                    else:
                        raise ValueError(f"Unsupported quantization type: {quantization_type}")

                    tensor = gguf.quants.quantize(tensor, data_type)

                except Exception as e:
                    data_type = gguf.GGMLQuantizationType.F16
                    print(f"Warning: Failed to quantize {name}: {e}")
                    # Fall back to original tensor if quantization fails

        print(f"Saving: {name} {tensor.shape}")
        gguf_writer.add_tensor(name, tensor, raw_dtype=data_type)

def load_tokenizer_data(model_dir):
    """Load tokenizer data from tokenizer.json (fast format) or vocab.json+merges.txt (GPT-2 format).

    Returns (tokenizer_data, tokenizer_config) where tokenizer_data['added_token_ids']
    is a set of token IDs that were added (e.g. emotion/style tags) and should be
    written as USER_DEFINED type in the GGUF file.
    """
    model_dir = Path(model_dir)
    tokenizer_config_path = model_dir / "tokenizer_config.json"

    if not tokenizer_config_path.exists():
        print(f"Warning: tokenizer_config.json not found in {model_dir}")
        return None, None

    try:
        with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)

        # Prefer the unified fast-tokenizer format
        tokenizer_path = model_dir / "tokenizer.json"
        if tokenizer_path.exists():
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
            # Collect added token IDs from the top-level "added_tokens" list
            added_token_ids = set()
            for entry in tokenizer_data.get('added_tokens', []):
                added_token_ids.add(entry['id'])
            tokenizer_data['added_token_ids'] = added_token_ids
            if added_token_ids:
                print(f"Found {len(added_token_ids)} added/special tokens in tokenizer.json")
            return tokenizer_data, tokenizer_config

        # Fall back to GPT-2 split format: vocab.json + merges.txt
        vocab_path  = model_dir / "vocab.json"
        merges_path = model_dir / "merges.txt"
        added_tokens_path = model_dir / "added_tokens.json"
        if vocab_path.exists() and merges_path.exists():
            print("tokenizer.json not found; loading vocab.json + merges.txt")
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            with open(merges_path, 'r', encoding='utf-8') as f:
                # skip the "#version" header line
                merges = [line.rstrip('\n') for line in f if not line.startswith('#') and line.strip()]
            # Merge added_tokens.json if present (e.g. emotion/style tags)
            added_token_ids = set()
            if added_tokens_path.exists():
                with open(added_tokens_path, 'r', encoding='utf-8') as f:
                    added_tokens = json.load(f)
                print(f"Merging {len(added_tokens)} added tokens from added_tokens.json")
                added_token_ids = set(added_tokens.values())
                vocab.update(added_tokens)
            tokenizer_data = {"model": {"vocab": vocab, "merges": merges}, "added_token_ids": added_token_ids}
            return tokenizer_data, tokenizer_config

        print(f"Warning: no tokenizer files found in {model_dir}")
        return None, None
    except Exception as e:
        print(f"Error loading tokenizer data: {e}")
        return None, None

def write_tokenizer_to_gguf(gguf_writer, tokenizer_data, tokenizer_config, tokenizer_model="gpt2"):
    """Write BPE tokenizer vocab, merges, and special token IDs to a GGUFWriter."""
    vocab = tokenizer_data['model']['vocab']
    added_token_ids = tokenizer_data.get('added_token_ids', set())
    tokens = []
    scores = []
    toktypes = []

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    token_id_map = {v: k for k, v in vocab.items()}
    for i in range(vocab_size):
        if i in token_id_map:
            token = token_id_map[i]
            tokens.append(token.encode('utf-8'))
            scores.append(0.0)
            if i in added_token_ids:
                toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                toktypes.append(gguf.TokenType.NORMAL)
        else:
            tokens.append(f"[UNUSED{i}]".encode('utf-8'))
            scores.append(0.0)
            toktypes.append(gguf.TokenType.UNUSED)

    n_added = sum(1 for t in toktypes if t == gguf.TokenType.USER_DEFINED)
    if n_added:
        print(f"Marked {n_added} tokens as USER_DEFINED (added/special)")

    gguf_writer.add_tokenizer_model(tokenizer_model)
    gguf_writer.add_tokenizer_pre("default")
    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)
    gguf_writer.add_token_types(toktypes)

    if 'merges' in tokenizer_data['model']:
        merges = tokenizer_data['model']['merges']
        print(f"Adding {len(merges)} BPE merges")
        gguf_writer.add_token_merges(merges)

    bos_token_id = tokenizer_config.get('bos_token_id', 255)
    eos_token_id = tokenizer_config.get('eos_token_id', 0)
    unk_token_id = tokenizer_config.get('unk_token_id', 1)
    pad_token_id = tokenizer_config.get('pad_token_id')

    gguf_writer.add_bos_token_id(bos_token_id)
    gguf_writer.add_eos_token_id(eos_token_id)
    gguf_writer.add_unk_token_id(unk_token_id)
    if pad_token_id is not None:
        gguf_writer.add_pad_token_id(pad_token_id)

    gguf_writer.add_add_bos_token(tokenizer_config.get('add_bos_token', False))
    gguf_writer.add_add_eos_token(tokenizer_config.get('add_eos_token', False))

def map_transformer_tensor_name(tensor_name):
    """Map GPT-2 style tfmr.* tensor names to llama.cpp GGUF gpt2 naming convention.

    GPT-2 Conv1D weights are stored transposed vs PyTorch Linear:
      c_attn / c_proj / c_fc  shape = [in, out]  → must transpose to [out, in].
    Returns (gguf_name, needs_transpose).
    """
    if not tensor_name.startswith('tfmr.'):
        return None, False

    name = tensor_name[5:]

    # Top-level tensors
    if name == 'wte.weight':
        return 'token_embd.weight', False
    if name == 'wpe.weight':
        return 'position_embd.weight', False
    if name in ('ln_f.weight', 'ln_f.bias'):
        suffix = name.split('.')[1]
        return f'output_norm.{suffix}', False

    # Per-block tensors: tfmr.h.{N}.*
    if name.startswith('h.'):
        parts = name.split('.')
        block_idx = int(parts[1])
        remaining = '.'.join(parts[2:])

        mapping = {
            'ln_1.weight':        (f'blk.{block_idx}.attn_norm.weight',  False),
            'ln_1.bias':          (f'blk.{block_idx}.attn_norm.bias',    False),
            'ln_2.weight':        (f'blk.{block_idx}.ffn_norm.weight',   False),
            'ln_2.bias':          (f'blk.{block_idx}.ffn_norm.bias',     False),
            # c_attn is Conv1D → fused QKV, shape [n_embd, 3*n_embd] → transpose
            'attn.c_attn.weight': (f'blk.{block_idx}.attn_qkv.weight',  True),
            'attn.c_attn.bias':   (f'blk.{block_idx}.attn_qkv.bias',    False),
            # c_proj output projection, shape [n_embd, n_embd] → transpose
            'attn.c_proj.weight': (f'blk.{block_idx}.attn_output.weight',  True),
            'attn.c_proj.bias':   (f'blk.{block_idx}.attn_output.bias',    False),
            # MLP: c_fc shape [n_embd, 4*n_embd] → transpose
            'mlp.c_fc.weight':    (f'blk.{block_idx}.ffn_up.weight',    True),
            'mlp.c_fc.bias':      (f'blk.{block_idx}.ffn_up.bias',      False),
            # c_proj shape [4*n_embd, n_embd] → transpose
            'mlp.c_proj.weight':  (f'blk.{block_idx}.ffn_down.weight',  True),
            'mlp.c_proj.bias':    (f'blk.{block_idx}.ffn_down.bias',    False),
        }
        return mapping.get(remaining, (None, False))

    return None, False

def is_backbone_tensor(tensor_name):
    """Check if a tensor belongs to the GPT-2 backbone (tfmr)."""
    return tensor_name.startswith('tfmr.')

def compute_llama3_rope_freqs(head_dim, rope_theta, scaling_factor,
                              max_position_embeddings, low_freq_factor=1.0,
                              high_freq_factor=4.0, old_context_len=8192):
    """Compute Llama-3 style rope_factors vector."""
    n_elem = head_dim // 2
    if n_elem <= 0:
        raise ValueError("head_dim too small for rotary computation")

    inds = np.arange(0, n_elem, dtype=np.float64)
    inv_freq = 1.0 / (rope_theta ** ((2.0 * inds) / float(head_dim)))
    freqs = inv_freq.astype(np.float64)
    wavelengths = (2.0 * np.pi) / freqs

    low_wavelen = float(old_context_len) / float(low_freq_factor)
    high_wavelen = float(old_context_len) / float(high_freq_factor)

    factors = np.empty_like(wavelengths, dtype=np.float32)
    if scaling_factor is None:
        scaling_factor = 1.0

    for i, wavelen in enumerate(wavelengths):
        if wavelen < high_wavelen:
            factors[i] = 1.0
        elif wavelen > low_wavelen:
            factors[i] = float(scaling_factor)
        else:
            smooth = (float(old_context_len) / wavelen - float(low_freq_factor)) / (float(high_freq_factor) - float(low_freq_factor))
            factors[i] = 1.0 / ((1.0 - smooth) / float(scaling_factor) + smooth)

    return factors.astype(np.float32)


# =============================================================================
# Logic Phase 1: T3 Embeddings & Tokenizer
# =============================================================================

def convert_t3_embeddings_gguf(state_dict, model_dir, output_path):
    """Convert embeddings and tokenizer to GGUF."""
    print(f"\n{'='*60}")
    print(f"Phase 1: Converting T3 Embeddings & Tokenizer")
    print(f"{'='*60}")

    gguf_writer = gguf.GGUFWriter(output_path, 't3-embeddings')
    
    gguf_writer.add_string('general.architecture', 't3-embeddings')
    gguf_writer.add_string('general.name', 'T3 Embeddings with Tokenizer')
    gguf_writer.add_string('general.description', 'T3 embeddings and BPE tokenizer for t3_integrated')
    
    # 1. Add Tokenizer Data
    print("Loading tokenizer data...")
    tokenizer_data, tokenizer_config = load_tokenizer_data(model_dir)
    
    if tokenizer_data and tokenizer_config:
        write_tokenizer_to_gguf(gguf_writer, tokenizer_data, tokenizer_config)
    else:
        print("Warning: Tokenizer data not added (files not found)")

    # 2. Add Embedding Tensors
    print("\nSearching for embedding tensors...")
    required_weights = {
        'text_emb.weight': 'text_emb_weight',
        'speech_emb.weight': 'speech_emb_weight',
        'text_pos_emb.emb.weight': 'text_pos_emb_emb_weight',
        'speech_pos_emb.emb.weight': 'speech_pos_emb_emb_weight'
    }
    
    found_weights = {}
    for pytorch_key, gguf_key in required_weights.items():
        if pytorch_key in state_dict:
            found_weights[gguf_key] = state_dict[pytorch_key]
            print(f"Found: {pytorch_key} -> {gguf_key}")
        else:
            print(f"Warning: {pytorch_key} not found in state dict")
    
    if len(found_weights) == 0:
        print("Error: No required weights found for embeddings!")
        return False

    if 'text_emb_weight' in found_weights:
        text_vocab_size, text_emb_dim = found_weights['text_emb_weight'].shape
        gguf_writer.add_uint32('t3.text_vocab_size', text_vocab_size)
        gguf_writer.add_uint32('t3.text_emb_dim', text_emb_dim)
    
    if 'speech_emb_weight' in found_weights:
        speech_vocab_size, speech_emb_dim = found_weights['speech_emb_weight'].shape
        gguf_writer.add_uint32('t3.speech_vocab_size', speech_vocab_size)
        gguf_writer.add_uint32('t3.speech_emb_dim', speech_emb_dim)
    
    print(f"Saving {len(found_weights)} tensors to GGUF...")
    for name, tensor in found_weights.items():
        # Force F16 for embeddings as in original script
        save_tensor(gguf_writer, name, tensor, force_f16=True)
    
    print(f"Writing to {output_path}...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print("Phase 1 Complete!")
    return True


# =============================================================================
# Logic Phase 2: GPT-2 Backbone
# =============================================================================

def convert_llama_backbone_gguf(state_dict, model_dir, output_path, config_override_path=None,
                                quantization_type=None, all_f32=False, rope_mode="auto",
                                rope_factor=None, rope_theta_override=None):
    """Convert the GPT-2 backbone from T3 to GGUF format for TTS inference."""
    print(f"\n{'='*60}")
    print(f"Phase 2: Converting GPT-2 Backbone (TTS Mode)")
    print(f"{'='*60}")

    gguf_writer = gguf.GGUFWriter(output_path, 'gpt2')
    
    gguf_writer.add_name("T3 GPT-2 Backbone (TTS - Speech Head)")
    gguf_writer.add_description("Extracted GPT-2 backbone from T3 with speech_head for TTS inference")
    gguf_writer.add_architecture()

    if all_f32:
        gguf_writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    elif quantization_type == "q4_0":
        gguf_writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_0)
    elif quantization_type == "q4_1":
        gguf_writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_1)
    elif quantization_type == "q5_0":
        gguf_writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q5_0)
    elif quantization_type == "q5_1":
        gguf_writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q5_1)
    elif quantization_type == "q8_0":
        gguf_writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)
    else:
        gguf_writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    # Derive architecture from actual weights (GPT-2 medium: 24 layers, hidden=1024, ff=4096, heads=16)
    backbone_layers = sorted(set(
        int(k.split('.')[2]) for k in state_dict if k.startswith('tfmr.h.')
    ))
    n_layer = len(backbone_layers)
    # Infer n_embd from the wte or speech_emb embedding width
    n_embd = int(state_dict['speech_emb.weight'].shape[1])
    # Infer n_head and n_ff from block 0
    n_ff   = int(state_dict['tfmr.h.0.mlp.c_fc.bias'].shape[0])
    n_head = 16  # GPT-2 medium; c_attn weight is [n_embd, 3*n_embd], head_dim = n_embd // n_head
    n_ctx  = int(state_dict['tfmr.wpe.weight'].shape[0])  # max context from positional embedding table

    print(f"Detected GPT-2 backbone: {n_layer} layers, n_embd={n_embd}, n_ff={n_ff}, n_head={n_head}, n_ctx={n_ctx}")

    # GPT-2 GGUF metadata keys
    gguf_writer.add_context_length(n_ctx)
    gguf_writer.add_embedding_length(n_embd)
    gguf_writer.add_block_count(n_layer)
    gguf_writer.add_feed_forward_length(n_ff)
    gguf_writer.add_head_count(n_head)
    gguf_writer.add_layer_norm_eps(1e-5)  # GPT-2 default
    # Use the speech head output dimension as vocab_size so llama.cpp sizes
    # the output logit tensor correctly (6563 speech tokens, not 50257 text tokens).
    speech_vocab_size = int(state_dict['speech_head.weight'].shape[0])
    gguf_writer.add_vocab_size(speech_vocab_size)
    print(f"Setting gpt2.vocab_size = {speech_vocab_size} (speech token count)")

    # Tokenizer — backbone uses a 6563-token speech vocabulary, not the GPT-2 text
    # vocabulary.  Set tokenizer.ggml.model = "no_vocab" so llama.cpp does not
    # instantiate a 50257-token BPE tokenizer that conflicts with the speech head.
    print("Loading tokenizer data...")
    tokenizer_data, tokenizer_config = load_tokenizer_data(model_dir)
    if tokenizer_data and tokenizer_config:
        write_tokenizer_to_gguf(gguf_writer, tokenizer_data, tokenizer_config, tokenizer_model="no_vocab")
    else:
        print("Warning: Tokenizer data not added (files not found)")

    # Convert backbone tensors
    tensor_count = 0

    if 'speech_emb.weight' not in state_dict:
        print("Error: speech_emb.weight not found in model")
        return False
    if 'speech_head.weight' not in state_dict:
        print("Error: speech_head.weight not found in model")
        return False

    speech_emb_tensor   = state_dict['speech_emb.weight']
    speech_head_tensor  = state_dict['speech_head.weight']
    speech_head_bias    = state_dict.get('speech_head.bias')

    for name, tensor in state_dict.items():
        if not is_backbone_tensor(name):
            continue

        # wte is replaced by speech_emb; skip it
        if name == 'tfmr.wte.weight':
            print(f"Skipping {name} (replaced by speech_emb.weight)")
            continue

        gguf_key, needs_transpose = map_transformer_tensor_name(name)
        if gguf_key is None:
            print(f"Warning: Skipping unmapped tensor: {name}")
            continue

        t = tensor
        if needs_transpose:
            # GPT-2 Conv1D stores weights as [in, out]; transpose to [out, in] for GGML matmul
            if isinstance(t, torch.Tensor):
                t = t.t().contiguous()
            else:
                t = np.ascontiguousarray(t.T)

        save_tensor(gguf_writer, gguf_key, t, quantization_type=quantization_type, all_f32=all_f32)
        tensor_count += 1

    print(f"Using speech_head.weight as output.weight")
    save_tensor(gguf_writer, 'output.weight', speech_head_tensor, quantization_type=quantization_type, all_f32=all_f32)
    tensor_count += 1

    if speech_head_bias is not None:
        print(f"Using speech_head.bias as output.bias")
        save_tensor(gguf_writer, 'output.bias', speech_head_bias, all_f32=True)
        tensor_count += 1

    print(f"Using speech_emb.weight as token_embd.weight")
    save_tensor(gguf_writer, 'token_embd.weight', speech_emb_tensor, quantization_type=quantization_type, all_f32=all_f32)
    tensor_count += 1

    print(f"Total backbone tensors saved: {tensor_count}")
    
    print(f"Writing to {output_path}...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print("Phase 2 Complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Unified converter for T3 models to GGUF format')
    parser.add_argument('model_path', type=str, help='Path to T3 safetensors/pth file')
    parser.add_argument('--model-dir', type=str, default=None, 
                        help='Directory containing config.json and tokenizer files tokenizer.json/tokenizer_config.json (default: model_path parent)')
    
    # Output options
    parser.add_argument('--output-embeddings', type=str, default=None,
                        help='Output path for embeddings/tokenizer GGUF')
    parser.add_argument('--output-backbone', type=str, default=None,
                        help='Output path for backbone GGUF')
    
    # Backbone options
    parser.add_argument('--quantize', '-q', type=str, default=None,
                        choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"],
                        help="Quantization type for backbone (default: f16)")
    parser.add_argument('--config-json', type=str, help='Override path to config.json')
    parser.add_argument('--all-f32', action='store_true', help='Use F32 for all backbone tensors')
    parser.add_argument('--rope-mode', type=str, choices=['auto', 'classic', 'compute', 'none'], 
                       default='auto', help='RoPE handling mode')
    parser.add_argument('--rope-factor', type=float, help='Override rope scaling factor')
    parser.add_argument('--rope-theta', type=float, help='Override rope theta value')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
        
    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = model_path.parent
    
    # Determine output paths
    if args.output_embeddings:
        out_emb = Path(args.output_embeddings)
    else:
        out_emb = Path(os.getcwd()) / 't3_model_with_tokenizer.gguf'
        
    if args.output_backbone:
        out_back = Path(args.output_backbone)
    else:
        quant_str = args.quantize.lower() if args.quantize else ("f32" if args.all_f32 else "f16")
        out_back = Path(os.getcwd()) / f"chatterbox_llama_backbone_tts_{quant_str}.gguf"

    try:
        # Load model once
        state_dict = load_model_file(model_path)
        
        # Phase 1: Embeddings
        success_emb = convert_t3_embeddings_gguf(state_dict, model_dir, str(out_emb))
        
        # Phase 2: Backbone
        success_back = convert_llama_backbone_gguf(
            state_dict, 
            model_dir, 
            str(out_back),
            config_override_path=args.config_json,
            quantization_type=args.quantize,
            all_f32=args.all_f32,
            rope_mode=args.rope_mode,
            rope_factor=args.rope_factor,
            rope_theta_override=args.rope_theta
        )
        
        if success_emb and success_back:
            print(f"\n{'='*60}")
            print(f"ALL SUCCESSFUL")
            print(f"Generated:")
            print(f"1. {out_emb} (Embeddings + Tokenizer)")
            print(f"2. {out_back} (Backbone)")
            print(f"{'='*60}")
        else:
            print("\nSome conversions failed.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

