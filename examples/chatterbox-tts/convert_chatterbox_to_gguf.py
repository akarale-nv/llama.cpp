#!/usr/bin/env python3
"""
Convert Chatterbox model checkpoints to GGUF format.

Subcommands:
  t3      Convert a T3 checkpoint to two GGUF files:
            - Embeddings GGUF   (default: t3_embeddings.gguf)
            - GPT-2 Backbone GGUF + tokenizer (default: chatterbox_llama_backbone_tts_<quant>.gguf)

  s3gen   Convert an S3Gen vocoder checkpoint to a single GGUF file:
            - S3Gen GGUF        (default: S3Gen-<params>-<quant>.gguf)

Usage:
  python convert_chatterbox_to_gguf.py all chatterbox-turbo/ --quantize f16

  python convert_chatterbox_to_gguf.py t3 path/to/t3_model.safetensors \\
      --model-dir path/to/tokenizer_dir/ --quantize f16

  python convert_chatterbox_to_gguf.py s3gen path/to/s3gen_model.safetensors \\
      --quantize f16 --output s3gen_f16.gguf

Dependencies: pip install gguf safetensors transformers torch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

try:
    import gguf
    import gguf.quants
except ImportError:
    print("Error: gguf-py is not installed. Install it with: pip install gguf")
    sys.exit(1)

try:
    from safetensors.torch import load_file
except ImportError:
    print("Error: safetensors is not installed. Install it with: pip install safetensors")
    sys.exit(1)

TensorDict = Dict[str, Any]


# =============================================================================
# Common helpers
# =============================================================================


def load_state_dict(path: str) -> TensorDict:
    """Load a state dict from .safetensors, .pth, or .pt file."""
    print(f"Loading model from {path}...")

    if path.endswith(".safetensors"):
        data = load_file(path)
    elif path.endswith((".pth", ".pt")):
        data = torch.load(path, map_location="cpu")
    else:
        print("Error: Unsupported file format. Use .pth, .pt, or .safetensors")
        sys.exit(1)

    def unwrap(obj: Any) -> Any:
        if isinstance(obj, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                if key in obj:
                    return unwrap(obj[key])
            return obj
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    try:
                        return unwrap(item)
                    except ValueError:
                        continue
            raise ValueError("Checkpoint list does not contain a usable state dict")
        if hasattr(obj, "state_dict"):
            return unwrap(obj.state_dict())
        raise ValueError(f"Unsupported checkpoint component: {type(obj)}")

    state_dict = unwrap(data)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(state_dict)}")

    # Remove any 'module.' prefix from DataParallel
    clean: TensorDict = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        clean[new_key] = value

    print(f"Loaded {len(clean)} tensors")
    return clean


def count_parameters(tensors: TensorDict) -> int:
    total = 0
    for value in tensors.values():
        if isinstance(value, torch.Tensor):
            total += value.numel()
        else:
            try:
                total += np.asarray(value).size
            except Exception:
                continue
    return total


def format_param_count(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B".replace(".0B", "B")
    if count >= 1_000_000:
        return f"{count / 1_000_000:.0f}M"
    if count >= 1_000:
        return f"{count / 1_000:.0f}K"
    return str(count)


# =============================================================================
# T3-specific helpers
# =============================================================================


def save_tensor(gguf_writer, name, tensor, quantization_type=None, all_f32=False, force_f16=False):
    """Save a single tensor to GGUF with optional quantization."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    if all_f32:
        gguf_writer.add_tensor(name, tensor.astype(np.float32))
    elif force_f16:
        if tensor.dtype != np.float16:
            tensor = tensor.astype(np.float16)
        gguf_writer.add_tensor(name, tensor)
    else:
        data_type = gguf.GGMLQuantizationType.F32
        if tensor.dtype != np.float16 and "bias" not in name and "emb" not in name and "norm" not in name:
            tensor = tensor.astype(np.float16)
            data_type = gguf.GGMLQuantizationType.F16

        # Apply quantization if specified and tensor is suitable
        if quantization_type and quantization_type != "f16" and tensor.size > 1024:
            if not any(skip in name.lower() for skip in ["bias", "emb", "norm", "pos_emb", "output"]):
                try:
                    print(f"Quantizing {name} with {quantization_type}")
                    qmap = {
                        "q4_0": gguf.GGMLQuantizationType.Q4_0,
                        "q4_1": gguf.GGMLQuantizationType.Q4_1,
                        "q5_0": gguf.GGMLQuantizationType.Q5_0,
                        "q5_1": gguf.GGMLQuantizationType.Q5_1,
                        "q8_0": gguf.GGMLQuantizationType.Q8_0,
                    }
                    data_type = qmap[quantization_type]
                    tensor = gguf.quants.quantize(tensor, data_type)
                except Exception as e:
                    data_type = gguf.GGMLQuantizationType.F16
                    print(f"Warning: Failed to quantize {name}: {e}")

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
            added_token_ids = set()
            for entry in tokenizer_data.get('added_tokens', []):
                added_token_ids.add(entry['id'])
            tokenizer_data['added_token_ids'] = added_token_ids
            if added_token_ids:
                print(f"Found {len(added_token_ids)} added/special tokens in tokenizer.json")
            return tokenizer_data, tokenizer_config

        # Fall back to GPT-2 split format: vocab.json + merges.txt
        vocab_path = model_dir / "vocab.json"
        merges_path = model_dir / "merges.txt"
        added_tokens_path = model_dir / "added_tokens.json"
        if vocab_path.exists() and merges_path.exists():
            print("tokenizer.json not found; loading vocab.json + merges.txt")
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            with open(merges_path, 'r', encoding='utf-8') as f:
                merges = [line.rstrip('\n') for line in f if not line.startswith('#') and line.strip()]
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
      c_attn / c_proj / c_fc  shape = [in, out] -> must transpose to [out, in].
    Returns (gguf_name, needs_transpose).
    """
    if not tensor_name.startswith('tfmr.'):
        return None, False

    name = tensor_name[5:]

    if name == 'wte.weight':
        return 'token_embd.weight', False
    if name == 'wpe.weight':
        return 'position_embd.weight', False
    if name in ('ln_f.weight', 'ln_f.bias'):
        suffix = name.split('.')[1]
        return f'output_norm.{suffix}', False

    if name.startswith('h.'):
        parts = name.split('.')
        block_idx = int(parts[1])
        remaining = '.'.join(parts[2:])

        mapping = {
            'ln_1.weight':        (f'blk.{block_idx}.attn_norm.weight',    False),
            'ln_1.bias':          (f'blk.{block_idx}.attn_norm.bias',      False),
            'ln_2.weight':        (f'blk.{block_idx}.ffn_norm.weight',     False),
            'ln_2.bias':          (f'blk.{block_idx}.ffn_norm.bias',       False),
            'attn.c_attn.weight': (f'blk.{block_idx}.attn_qkv.weight',    True),
            'attn.c_attn.bias':   (f'blk.{block_idx}.attn_qkv.bias',      False),
            'attn.c_proj.weight': (f'blk.{block_idx}.attn_output.weight',  True),
            'attn.c_proj.bias':   (f'blk.{block_idx}.attn_output.bias',    False),
            'mlp.c_fc.weight':    (f'blk.{block_idx}.ffn_up.weight',       True),
            'mlp.c_fc.bias':      (f'blk.{block_idx}.ffn_up.bias',         False),
            'mlp.c_proj.weight':  (f'blk.{block_idx}.ffn_down.weight',     True),
            'mlp.c_proj.bias':    (f'blk.{block_idx}.ffn_down.bias',       False),
        }
        return mapping.get(remaining, (None, False))

    return None, False


# =============================================================================
# T3 conversion: Embeddings (Phase 1) + Backbone (Phase 2)
# =============================================================================


def convert_t3_embeddings_gguf(state_dict, output_path):
    """Convert text/speech/positional embeddings to GGUF."""
    print(f"\n{'='*60}")
    print(f"Phase 1: Converting T3 Embeddings")
    print(f"{'='*60}")

    gguf_writer = gguf.GGUFWriter(output_path, 't3-embeddings')

    gguf_writer.add_string('general.architecture', 't3-embeddings')
    gguf_writer.add_string('general.name', 'T3 Embeddings')
    gguf_writer.add_string('general.description', 'T3 text/speech embeddings for Chatterbox TTS')

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
        save_tensor(gguf_writer, name, tensor, force_f16=True)

    print(f"Writing to {output_path}...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print("Phase 1 Complete!")
    return True


def convert_t3_backbone_gguf(state_dict, model_dir, output_path,
                             quantization_type=None, all_f32=False):
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

    # Derive architecture from actual weights
    backbone_layers = sorted(set(
        int(k.split('.')[2]) for k in state_dict if k.startswith('tfmr.h.')
    ))
    n_layer = len(backbone_layers)
    n_embd = int(state_dict['speech_emb.weight'].shape[1])
    n_ff   = int(state_dict['tfmr.h.0.mlp.c_fc.bias'].shape[0])
    n_head = 16  # GPT-2 medium
    n_ctx  = int(state_dict['tfmr.wpe.weight'].shape[0])

    print(f"Detected GPT-2 backbone: {n_layer} layers, n_embd={n_embd}, n_ff={n_ff}, n_head={n_head}, n_ctx={n_ctx}")

    gguf_writer.add_context_length(n_ctx)
    gguf_writer.add_embedding_length(n_embd)
    gguf_writer.add_block_count(n_layer)
    gguf_writer.add_feed_forward_length(n_ff)
    gguf_writer.add_head_count(n_head)
    gguf_writer.add_layer_norm_eps(1e-5)

    speech_vocab_size = int(state_dict['speech_head.weight'].shape[0])
    gguf_writer.add_vocab_size(speech_vocab_size)
    print(f"Setting gpt2.vocab_size = {speech_vocab_size} (speech token count)")

    # Tokenizer -- written to the backbone GGUF (not the embeddings GGUF).
    # Set tokenizer.ggml.model = "no_vocab" so llama.cpp does not instantiate
    # a 50257-token BPE tokenizer that conflicts with the 6563-token speech head.
    print("Loading tokenizer data...")
    tokenizer_data, tokenizer_config = load_tokenizer_data(model_dir)
    if tokenizer_data and tokenizer_config:
        write_tokenizer_to_gguf(gguf_writer, tokenizer_data, tokenizer_config, tokenizer_model="no_vocab")
    else:
        print("Warning: Tokenizer data not added (files not found)")

    tensor_count = 0

    if 'speech_emb.weight' not in state_dict:
        print("Error: speech_emb.weight not found in model")
        return False
    if 'speech_head.weight' not in state_dict:
        print("Error: speech_head.weight not found in model")
        return False

    speech_emb_tensor  = state_dict['speech_emb.weight']
    speech_head_tensor = state_dict['speech_head.weight']
    speech_head_bias   = state_dict.get('speech_head.bias')

    for name, tensor in state_dict.items():
        if not name.startswith('tfmr.'):
            continue
        if name == 'tfmr.wte.weight':
            print(f"Skipping {name} (replaced by speech_emb.weight)")
            continue

        gguf_key, needs_transpose = map_transformer_tensor_name(name)
        if gguf_key is None:
            print(f"Warning: Skipping unmapped tensor: {name}")
            continue

        t = tensor
        if needs_transpose:
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


# =============================================================================
# S3Gen-specific helpers
# =============================================================================


def convert_s3gen_tensor_name(pytorch_name: str) -> str:
    """Return a GGUF-friendly tensor name for the S3Gen architecture."""
    replacements = (
        ("mel2wav.f0_predictor.", "m2wf0p_"),
        ("mel2wav.m_source.", "m2ws_"),
        ("mel2wav.source_resblocks.", "m2wsrb_"),
        ("mel2wav.source_downs.", "m2wsdown_"),
        ("mel2wav.source_upblocks.", "m2wsupblk_"),
        ("mel2wav.resblocks.", "m2wsrb_"),
        ("mel2wav.ups.", "m2wsups_"),
        ("mel2wav.", "m2w_"),
        ("flow.decoder.estimator.", "fdes_"),
        ("flow.decoder.", "fdec_"),
        ("flow.encoder.embed.", "fenc_embed_"),
        ("flow.encoder.up_embed.", "fenc_up_embed_"),
        ("flow.encoder.up_encoders.", "fenc_upblk_"),
        ("flow.encoder.encoders.", "fenc_blk_"),
        ("flow.encoder.pre_lookahead_layer.", "fenc_prelook_"),
        ("flow.encoder.up_layer.", "fenc_up_layer_"),
        ("flow.encoder.", "fenc_"),
        ("flow.input_embedding.", "finput_emb_"),
        ("flow.spk_embed_affine_layer.", "fspk_affine_"),
        ("flow.encoder_proj.", "fenc_proj_"),
        ("flow.", "f_"),
        ("speaker_encoder.xvector.", "s_en_xvec_"),
        ("tokenizer.", "tkn_"),
    )

    for prefix, repl in replacements:
        if pytorch_name.startswith(prefix):
            pytorch_name = pytorch_name.replace(prefix, repl, 1)
            break
        pytorch_name = pytorch_name.replace("block", "blk")

    return pytorch_name.replace(".", "_")


# =============================================================================
# S3Gen conversion
# =============================================================================


def convert_s3gen_to_gguf(
    state_dict: TensorDict,
    output: str,
    quant: str | None,
    total_params: int,
) -> None:
    """Convert an S3Gen vocoder state dict to GGUF."""
    fout = gguf.GGUFWriter(path=output, arch="s3gen")

    fout.add_string("general.name", "S3Gen Speech-to-Speech Model")
    fout.add_string(
        "general.description",
        "Chatterbox S3Gen token-to-waveform model converted to GGUF",
    )
    fout.add_uint64("general.parameter_count", total_params)

    ftype = 0  # f32 by default
    if quant == "f16":
        ftype = 1
    fout.add_file_type(ftype)

    fout.add_uint32("s3gen.sample_rate", 24_000)
    fout.add_uint32("s3gen.mel_channels", 80)
    fout.add_uint32("s3gen.token_mel_ratio", 2)
    fout.add_uint32("s3gen.diffusion_steps", 10)
    fout.add_bool("s3gen.uses_causal_decoder", True)
    fout.add_bool("s3gen.uses_hift_generator", True)
    fout.add_bool("s3gen.uses_speaker_encoder", True)

    flow_vocab = state_dict.get("flow.input_embedding.weight")
    if isinstance(flow_vocab, torch.Tensor):
        vocab_size, embed_dim = flow_vocab.shape
        fout.add_uint32("s3gen.token_vocab_size", vocab_size)
        fout.add_uint32("s3gen.embedding_dim", embed_dim)

    prompt_proj = state_dict.get("flow.spk_embed_affine_layer.weight")
    if isinstance(prompt_proj, torch.Tensor):
        out_dim, in_dim = prompt_proj.shape
        fout.add_uint32("s3gen.speaker_embedding_dim", in_dim)
        fout.add_uint32("s3gen.condition_dim", out_dim)

    count = 0
    for name in sorted(state_dict):
        value = state_dict[name]
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu()
            if t.dtype == torch.bfloat16:
                t = t.to(torch.float32)
            elif str(t.dtype).startswith('torch.float8'):
                t = t.to(torch.float16)
            array = t.squeeze().numpy()
        else:
            try:
                array = np.asarray(value).squeeze()
            except Exception:
                print(f"Skipping non-tensor entry {name}")
                continue

        if array.dtype == object:
            print(f"Skipping non-numeric entry {name}")
            continue

        gguf_name = convert_s3gen_tensor_name(name)
        ftype_cur = 0

        if quant and quant.lower() in ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"]:
            name_l = name.lower()
            is_matrix = array.ndim == 2
            skip_tokens = ("bias", "emb", "norm", "layernorm", "pos_emb", "alpha", "beta",
                           "conv", "convt", "convtranspose", "parametrizations")
            skip_by_name = any(tok in name_l for tok in skip_tokens)
            allow_prefixes = ("flow.encoder.", "flow.encoder_proj.")
            in_allow_scope = name_l.startswith(allow_prefixes)

            should_quantise = (is_matrix and array.size > 1024 and not skip_by_name and in_allow_scope)

            if should_quantise:
                qmap = {
                    "q4_0": gguf.GGMLQuantizationType.Q4_0,
                    "q4_1": gguf.GGMLQuantizationType.Q4_1,
                    "q5_0": gguf.GGMLQuantizationType.Q5_0,
                    "q5_1": gguf.GGMLQuantizationType.Q5_1,
                    "q8_0": gguf.GGMLQuantizationType.Q8_0,
                }
                qtype = qmap.get(quant.lower())
                if qtype is not None:
                    try:
                        print(f"Quantizing {gguf_name} -> {quant.upper()}")
                        array = gguf.quants.quantize(array, qtype)
                        fout.add_tensor(gguf_name, array, raw_dtype=qtype)
                        count += 1
                        continue
                    except Exception as exc:
                        print(f"Warning: failed to quantize {gguf_name}: {exc}")

        if quant == "f16" or (not quant and array.ndim > 1):
            if array.dtype != np.float16:
                print(f"Converting {gguf_name} to float16")
                array = array.astype(np.float16)
            ftype_cur = 1
        elif quant == "f32" or not quant:
            if array.dtype != np.float32:
                print(f"Converting {gguf_name} to float32")
                array = array.astype(np.float32)
            ftype_cur = 0

        ftype_str = ["f32", "f16"]
        print(f"{gguf_name} - {ftype_str[ftype_cur]} - shape = {array.shape}")
        fout.add_tensor(gguf_name, array)
        count += 1

    fout.write_header_to_file()
    fout.write_kv_data_to_file()
    fout.write_tensors_to_file()
    fout.close()


# =============================================================================
# CLI
# =============================================================================


def cmd_t3(args):
    """Run the T3 conversion."""
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    model_dir = args.model_dir if args.model_dir else model_path.parent

    if args.output_embeddings:
        out_emb = Path(args.output_embeddings)
    else:
        out_emb = Path(os.getcwd()) / 't3_embeddings.gguf'

    if args.output_backbone:
        out_back = Path(args.output_backbone)
    else:
        quant_str = args.quantize.lower() if args.quantize else ("f32" if args.all_f32 else "f16")
        out_back = Path(os.getcwd()) / f"chatterbox_llama_backbone_tts_{quant_str}.gguf"

    try:
        state_dict = load_state_dict(str(model_path))

        success_emb = convert_t3_embeddings_gguf(state_dict, str(out_emb))

        success_back = convert_t3_backbone_gguf(
            state_dict,
            model_dir,
            str(out_back),
            quantization_type=args.quantize,
            all_f32=args.all_f32,
        )

        if success_emb and success_back:
            print(f"\n{'='*60}")
            print(f"ALL SUCCESSFUL")
            print(f"Generated:")
            print(f"1. {out_emb} (Embeddings)")
            print(f"2. {out_back} (Backbone + Tokenizer)")
            print(f"{'='*60}")
        else:
            print("\nSome conversions failed.")
            sys.exit(1)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_s3gen(args):
    """Run the S3Gen conversion."""
    if not os.path.exists(args.model_path):
        print(f"Error: checkpoint '{args.model_path}' not found")
        sys.exit(1)

    try:
        print(f"Loading checkpoint from {args.model_path} ...")
        state_dict = load_state_dict(args.model_path)

        total_params = count_parameters(state_dict)
        param_label = format_param_count(total_params)
        type_label = args.quantize.upper() if args.quantize else "F32"
        print(f"Total parameters: {total_params:,} ({param_label})")
        print(f"Data type: {type_label}")

        if "flow.input_embedding.weight" in state_dict:
            vocab, dim = state_dict["flow.input_embedding.weight"].shape
            print(f"Token vocab size: {vocab}, embedding dim: {dim}")
        if "mel2wav.conv_pre.weight" in state_dict:
            out_channels, in_channels, width = state_dict["mel2wav.conv_pre.weight"].shape
            print(f"Mel2Wav conv_pre: out={out_channels}, in={in_channels}, kernel={width}")

        output = args.output or f"S3Gen-{param_label}-{type_label}.gguf"
        print(f"Writing GGUF -> {output}")
        convert_s3gen_to_gguf(state_dict, output, args.quantize, total_params)

        size = os.path.getsize(output)
        print(f"Output size: {size:,} bytes ({size / (1024**2):.1f} MB)")
        print("Conversion completed successfully.")
    except Exception as exc:
        print(f"Error converting model: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def find_checkpoint(model_dir: Path, prefixes: list[str], label: str) -> Path:
    """Find a .safetensors or .pth file matching one of the given prefixes."""
    for prefix in prefixes:
        for ext in (".safetensors", ".pth", ".pt"):
            candidate = model_dir / f"{prefix}{ext}"
            if candidate.exists():
                return candidate
    # Fallback: glob for partial match
    for prefix in prefixes:
        for ext in (".safetensors", ".pth", ".pt"):
            matches = sorted(model_dir.glob(f"{prefix}*{ext}"))
            if matches:
                return matches[0]
    print(f"Error: could not find {label} checkpoint in {model_dir}")
    print(f"  Looked for files starting with: {', '.join(prefixes)}")
    sys.exit(1)


def cmd_all(args):
    """Run both T3 and S3Gen conversions from a model directory."""
    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"Error: {model_dir} is not a directory")
        sys.exit(1)

    t3_path = args.t3_model or str(find_checkpoint(model_dir, ["t3_turbo", "t3"], "T3"))
    s3gen_path = args.s3gen_model or str(find_checkpoint(model_dir, ["s3gen_meanflow"], "S3Gen"))

    print(f"T3 checkpoint:    {t3_path}")
    print(f"S3Gen checkpoint: {s3gen_path}")

    # Reuse cmd_t3 and cmd_s3gen by building compatible args namespaces
    class T3Args:
        model_path = t3_path
        model_dir = str(args.model_dir)
        output_embeddings = args.output_embeddings
        output_backbone = args.output_backbone
        quantize = args.quantize
        all_f32 = args.all_f32
    cmd_t3(T3Args())

    class S3Args:
        model_path = s3gen_path
        quantize = args.s3gen_quantize or args.quantize
        output = args.output_s3gen
    cmd_s3gen(S3Args())

    print(f"\n{'='*60}")
    print("All conversions completed.")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Chatterbox model checkpoints to GGUF format")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- all subcommand ---
    p_all = subparsers.add_parser("all",
        help="Convert both T3 and S3Gen from a model directory")
    p_all.add_argument("model_dir",
        help="Path to model directory (e.g. chatterbox-turbo/). "
             "Auto-discovers t3_turbo*.safetensors and s3gen*.safetensors")
    p_all.add_argument("--t3-model", default=None,
        help="Override T3 checkpoint path (default: auto-detect in model_dir)")
    p_all.add_argument("--s3gen-model", default=None,
        help="Override S3Gen checkpoint path (default: auto-detect in model_dir)")
    p_all.add_argument("--quantize", "-q", default=None,
        choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"],
        help="Quantization type for both models (default: f16 for T3, f32 for S3Gen)")
    p_all.add_argument("--s3gen-quantize", default=None,
        choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"],
        help="Override quantization for S3Gen only")
    p_all.add_argument("--output-embeddings", default=None,
        help="Output embeddings GGUF (default: t3_embeddings.gguf)")
    p_all.add_argument("--output-backbone", default=None,
        help="Output backbone + tokenizer GGUF")
    p_all.add_argument("--output-s3gen", default=None,
        help="Output S3Gen GGUF")
    p_all.add_argument("--all-f32", action="store_true",
        help="Use F32 for all T3 backbone tensors")

    # --- t3 subcommand ---
    p_t3 = subparsers.add_parser("t3",
        help="Convert T3 checkpoint to embeddings + backbone GGUF files")
    p_t3.add_argument("model_path",
        help="Path to T3 .safetensors or .pth checkpoint "
             "(must contain tfmr.*, text_emb, speech_emb, speech_head tensors)")
    p_t3.add_argument("--model-dir", default=None,
        help="Directory with tokenizer files (default: parent of model_path)")
    p_t3.add_argument("--output-embeddings", default=None,
        help="Output embeddings GGUF (default: t3_embeddings.gguf)")
    p_t3.add_argument("--output-backbone", default=None,
        help="Output backbone + tokenizer GGUF")
    p_t3.add_argument("--quantize", "-q", default=None,
        choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"],
        help="Quantization type for backbone (default: f16)")
    p_t3.add_argument("--all-f32", action="store_true",
        help="Use F32 for all backbone tensors")

    # --- s3gen subcommand ---
    p_s3 = subparsers.add_parser("s3gen",
        help="Convert S3Gen vocoder checkpoint to GGUF")
    p_s3.add_argument("model_path",
        help="Path to S3Gen .safetensors or .pth checkpoint")
    p_s3.add_argument("--quantize", "-q", default=None,
        choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"],
        help="Quantization type (default: f32)")
    p_s3.add_argument("--output", "-o", default=None,
        help="Output GGUF path (default: S3Gen-<params>-<quant>.gguf)")

    args = parser.parse_args()

    if args.command == "all":
        cmd_all(args)
    elif args.command == "t3":
        cmd_t3(args)
    elif args.command == "s3gen":
        cmd_s3gen(args)


if __name__ == "__main__":
    main()
