#!/usr/bin/env python3
"""Convert the Chatterbox S3Gen speech-to-speech checkpoint to GGUF."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

import gguf
import gguf.quants
import numpy as np
import torch

TensorDict = Dict[str, Any]


# -----------------------------------------------------------------------------
# Tensor helpers
# -----------------------------------------------------------------------------


def convert_tensor_name(pytorch_name: str) -> str:
    """Return a GGUF-friendly tensor name for the S3Gen architecture."""

    # replacements = (
    #     ("mel2wav.f0_predictor.", "mel2wav_f0_"),
    #     ("mel2wav.m_source.", "mel2wav_source_"),
    #     ("mel2wav.source_resblocks.", "mel2wav_source_resblk_"),
    #     ("mel2wav.source_downs.", "mel2wav_source_down_"),
    #     ("mel2wav.source_upblocks.", "mel2wav_source_upblk_"),
    #     ("mel2wav.resblocks.", "mel2wav_resblk_"),
    #     ("mel2wav.ups.", "mel2wav_ups_"),
    #     ("mel2wav.", "mel2wav_"),
    #     ("flow.decoder.estimator.", "flow_decoder_estimator_"),
    #     ("flow.decoder.", "flow_decoder_"),
    #     ("flow.encoder.embed.", "flow_encoder_embed_"),
    #     ("flow.encoder.up_embed.", "flow_encoder_up_embed_"),
    #     ("flow.encoder.up_encoders.", "flow_encoder_upblk_"),
    #     ("flow.encoder.encoders.", "flow_encoder_blk_"),
    #     ("flow.encoder.pre_lookahead_layer.", "flow_encoder_prelook_"),
    #     ("flow.encoder.up_layer.", "flow_encoder_up_layer_"),
    #     ("flow.encoder.", "flow_encoder_"),
    #     ("flow.input_embedding.", "flow_input_emb_"),
    #     ("flow.spk_embed_affine_layer.", "flow_spk_affine_"),
    #     ("flow.encoder_proj.", "flow_encoder_proj_"),
    #     ("flow.", "flow_"),
    #     ("speaker_encoder.", "speaker_encoder_"),
    #     ("tokenizer.", "tokenizer_"),
    # )

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


# -----------------------------------------------------------------------------
# State dict handling
# -----------------------------------------------------------------------------


def load_state_dict(path: str) -> TensorDict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file  # lazy import

        data = load_file(path)
    else:
        data = torch.load(path, map_location="cpu")

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

    clean: TensorDict = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        clean[new_key] = value
    return clean


# -----------------------------------------------------------------------------
# Conversion logic
# -----------------------------------------------------------------------------


def convert_state_dict_to_gguf(
    state_dict: TensorDict,
    output: str,
    quant: str | None,
    total_params: int,
) -> None:
    fout = gguf.GGUFWriter(path=output, arch="s3gen")

    # fout.add_string("general.architecture", "s3gen")
    fout.add_string("general.name", "S3Gen Speech-to-Speech Model")
    fout.add_string(
        "general.description",
        "Chatterbox S3Gen token-to-waveform model converted to GGUF",
    )
    fout.add_uint64("general.parameter_count", total_params)

    # Set file type based on data type
    ftype = 0  # f32 by default
    if quant == "f16":
        ftype = 1
    fout.add_file_type(ftype)

    # mmproj-compatible metadata so clip_model_load() can identify this as an s3gen projector
    fout.add_string("clip.projector_type", "s3gen")
    fout.add_bool("clip.has_audio_encoder", False)
    fout.add_bool("clip.has_vision_encoder", False)

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
    # Add tensors directly (like image encoder approach)
    for name in sorted(state_dict):
        # if count > 100: break
        value = state_dict[name]
        if isinstance(value, torch.Tensor):
            # Safely handle Torch dtypes that NumPy doesn't support directly
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

        if array.dtype == object:  # guard against nested dict leftovers
            print(f"Skipping non-numeric entry {name}")
            continue

        gguf_name = convert_tensor_name(name)
        
        # Handle data type conversion (similar to image encoder)
        n_dims = len(array.shape)
        ftype_cur = 0
        
        if quant and quant.lower() in ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"]:
            # Apply quantization for large 2D matrices only
            name_l = name.lower()
            is_matrix = array.ndim == 2
            skip_tokens = ("bias", "emb", "norm", "layernorm", "pos_emb", "alpha", "beta", "conv", "convt", "convtranspose", "parametrizations")
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
                        # For quantized tensors, we still need raw_dtype
                        fout.add_tensor(gguf_name, array, raw_dtype=qtype)
                        count += 1
                        continue
                    except Exception as exc:
                        print(f"Warning: failed to quantize {gguf_name}: {exc}")
        
        # Default data type handling (like image encoder - no raw_dtype parameter)
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
        # Use add_tensor WITHOUT raw_dtype parameter (like image encoder)
        fout.add_tensor(gguf_name, array)
        count += 1

    fout.write_header_to_file()
    fout.write_kv_data_to_file()
    fout.write_tensors_to_file()
    fout.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the S3Gen checkpoint to GGUF")
    parser.add_argument("model_path", help="Path to the PyTorch or safetensors checkpoint")
    parser.add_argument(
        "--quantize",
        "-q",
        choices=[
            "q4_0",
            "q4_1",
            "q5_0",
            "q5_1",
            "q8_0",
            "q2_k",
            "q3_k",
            "q4_k",
            "q5_k",
            "q6_k",
            "q8_k",
            "f16",
            "f32",
        ],
        default=None,
        help="Data type/quantization (default: f32)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output GGUF path (default: S3Gen-<params>-<quant>.gguf)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: checkpoint '{args.model_path}' not found")
        sys.exit(1)

    try:
        print(f"Loading checkpoint from {args.model_path} ...")
        state_dict = load_state_dict(args.model_path)
        print(f"Loaded {len(state_dict)} tensors")

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
        convert_state_dict_to_gguf(state_dict, output, args.quantize, total_params)

        size = os.path.getsize(output)
        print(f"Output size: {size:,} bytes ({size / (1024**2):.1f} MB)")
        print("Conversion completed successfully.")
    except Exception as exc:  # pragma: no cover - expose details then exit
        print(f"Error converting model: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

