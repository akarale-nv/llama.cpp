# Chatterbox TTS

Text-to-speech inference using the [Chatterbox](https://github.com/resemble-ai/chatterbox) model by Resemble AI, running entirely in llama.cpp.

The pipeline has two stages:

1. **T3** -- a GPT-2 backbone (24 layers, 1024 hidden, 16 heads) that converts text into speech tokens
2. **S3Gen** -- a flow-matching vocoder that converts speech tokens into a 24 kHz waveform

Both stages run on CPU, CUDA, Vulkan, or Metal via ggml.

## Quick start (turbo model)

### 1. Install Python dependencies

```bash
pip install gguf safetensors transformers torch
```

### 2. Download the model

Download from [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo):

```bash
git clone https://huggingface.co/ResembleAI/chatterbox-turbo
```

Key files in the repo:

| File | Description |
|------|-------------|
| `t3_turbo_v1.safetensors` | T3 checkpoint (GPT-2 backbone + embeddings + speech head) |
| `s3gen.safetensors` | S3Gen vocoder (flow matching + HiFT decoder) |
| `vocab.json` + `merges.txt` | GPT-2 BPE tokenizer |
| `tokenizer_config.json` | Tokenizer configuration |
| `conds.pt` | Speaker conditioning data |
| `ve.safetensors` | Voice encoder (for future runtime speaker embedding) |

### 3. Convert to GGUF

Point the converter at the model folder -- it auto-discovers the T3 and S3Gen checkpoints:

```bash
python examples/chatterbox-tts/convert_chatterbox_to_gguf.py all \
    chatterbox-turbo/
```

Output (three GGUF files):
- `t3_embeddings.gguf` -- text/speech embeddings and positional embeddings
- `chatterbox_llama_backbone_tts_f16.gguf` -- the 24-layer GPT-2 transformer backbone + tokenizer
- `S3Gen-<params>-F16.gguf` -- the flow-matching vocoder (token to mel to waveform)

You can also convert each model separately using the `t3` and `s3gen` subcommands (see [Conversion script reference](#conversion-script-reference)).

### 4. Build

```bash
cmake -B build
cmake --build build --target llama-chatterbox -j
```

### 5. Prepare speaker conditioning

The model requires a speaker conditioning JSON file that provides the voice characteristics. This file is located in the model directory alongside the checkpoints (e.g. `chatterbox-turbo/cond.json`). Pass it via `--cond-emb`.

The JSON file contains base64-encoded fields (each with `bin_data` and `shape` keys):

- `cond_emb` -- prefill conditioning embeddings (shape `[1, 376, 1024]`)
- `spk_emb` -- 192-dim ECAPA speaker embedding for S3Gen voice control
- `prompt_token` -- reference speech token IDs for voice cloning
- `prompt_feat` -- reference mel spectrogram frames (80 channels)

> **Note:** A reference voice encoder is planned for a future release, which will generate the speaker embedding at runtime from a reference audio file instead of requiring pre-exported embeddings.

### 6. Run

**Full pipeline (text to WAV):**

```bash
./build/bin/llama-chatterbox \
    -m t3_embeddings.gguf \
    --model-llama chatterbox_llama_backbone_tts_f16.gguf \
    --prompt "So I was thinking [chuckle] maybe we should just go for it." \
    --cond-emb speaker.json \
    --s3gen S3Gen-266M-F32.gguf \
    --output output.wav
```

**Token generation only (no vocoder):**

```bash
./build/bin/llama-chatterbox \
    -m t3_embeddings.gguf \
    --model-llama chatterbox_llama_backbone_tts_f16.gguf \
    --prompt "So I was thinking [chuckle] maybe we should just go for it." \
    --cond-emb speaker.json \
    --tokens-out tokens.txt
```

**Vocoder only (from pre-generated tokens):**

```bash
./build/bin/llama-chatterbox \
    --cond-emb speaker.json \
    --s3gen S3Gen-266M-F32.gguf \
    --tokens-in 1234,5678,9012,3456 \
    --output output.wav
```

## CLI reference

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-m PATH` | yes* | -- | T3 embeddings GGUF |
| `--model-llama PATH` | yes* | -- | T3 backbone GGUF |
| `--prompt TEXT` | yes* | -- | Text to synthesize |
| `--cond-emb PATH` | yes | -- | Speaker conditioning JSON |
| `--s3gen PATH` | no | -- | S3Gen vocoder GGUF (enables WAV output) |
| `--output PATH` | no | `output.wav` | Output WAV file path |
| `--tokens-out PATH` | no | -- | Write generated speech token IDs to file |
| `--tokens-in T1,T2,..` | no | -- | Feed token list directly to S3Gen (skips T3) |
| `--max-tokens N` | no | 2000 | Maximum speech tokens to generate |
| `--temperature N` | no | 1.0 | Sampling temperature |
| `--top-p N` | no | 0.95 | Top-p nucleus sampling |
| `--min-p N` | no | 0.05 | Min-p sampling threshold |
| `--rep-penalty N` | no | 1.2 | Repetition penalty |
| `--seed N` | no | -1 | Random seed (-1 = random) |
| `--ngl N` | no | 99 | GPU layers to offload |

\* Not required when using `--tokens-in` mode.

## Conversion script reference

All conversions are handled by a single script with subcommands.

### convert_chatterbox_to_gguf.py all

Auto-discovers T3 and S3Gen checkpoints in a model directory and converts both, producing three GGUF files. Looks for `t3_turbo*.safetensors` and `s3gen*.safetensors` by default.

```
python convert_chatterbox_to_gguf.py all <model_dir> [options]

  model_dir                   Path to model directory (e.g. chatterbox-turbo/)

  --t3-model PATH             Override T3 checkpoint path (default: auto-detect)
  --s3gen-model PATH          Override S3Gen checkpoint path (default: auto-detect)
  --quantize TYPE             Quantization for both models (default: f16 for T3, f32 for S3Gen)
  --s3gen-quantize TYPE       Override quantization for S3Gen only
  --output-embeddings PATH    Output embeddings GGUF (default: t3_embeddings.gguf)
  --output-backbone PATH      Output backbone + tokenizer GGUF
  --output-s3gen PATH         Output S3Gen GGUF
  --all-f32                   Force F32 for all T3 backbone tensors
```

### convert_chatterbox_to_gguf.py t3

Converts a T3 checkpoint to two GGUF files (embeddings, and GPT-2 backbone + tokenizer).

The input checkpoint must contain:
- `tfmr.*` -- GPT-2 backbone weights (24 layers)
- `text_emb.weight`, `speech_emb.weight` -- token embeddings
- `text_pos_emb.emb.weight`, `speech_pos_emb.emb.weight` -- positional embeddings
- `speech_head.weight`, `speech_head.bias` -- output projection (6563 speech tokens)

Tokenizer files (`tokenizer.json` or `vocab.json` + `merges.txt`, and `tokenizer_config.json`) must be in the checkpoint directory or supplied via `--model-dir`.

```
python convert_chatterbox_to_gguf.py t3 <model_path> [options]

  model_path                  Path to T3 .safetensors or .pth checkpoint

  --model-dir DIR             Directory with tokenizer files (default: parent of model_path)
  --output-embeddings PATH    Output embeddings GGUF (default: t3_embeddings.gguf)
  --output-backbone PATH      Output backbone + tokenizer GGUF (default: chatterbox_llama_backbone_tts_<quant>.gguf)
  --quantize TYPE             q4_0, q4_1, q5_0, q5_1, q8_0, f16, f32 (default: f16)
  --all-f32                   Force F32 for all backbone tensors
```

### convert_chatterbox_to_gguf.py s3gen

Converts an S3Gen vocoder checkpoint to a single GGUF file.

The input checkpoint must contain:
- `flow.*` -- flow-matching encoder and decoder
- `mel2wav.*` -- HiFT vocoder (mel to waveform)
- `speaker_encoder.*` -- ECAPA speaker encoder (optional)

```
python convert_chatterbox_to_gguf.py s3gen <model_path> [options]

  model_path                  Path to S3Gen .safetensors or .pth checkpoint

  --quantize TYPE             q4_0, q4_1, q5_0, q5_1, q8_0, f16, f32 (default: f32)
  --output PATH               Output GGUF (default: S3Gen-<params>-<quant>.gguf)
```

S3Gen quantization only applies to large 2D matrices in the encoder; biases, norms, embeddings, and convolution weights are kept at full precision.

## Architecture overview

```
Input Text
    |
    v
GPT-2 BPE Tokenizer
    |
    v
T3 Backbone (GPT-2, 24 layers)
  Input: [speaker_cond(376) | text_embs(N) | SOS(1)]
  Output: speech token IDs (vocab 6563)
    |
    v
S3Gen Vocoder
  +-- UpsampleConformer Encoder (6 blocks + 4 upsample stages)
  +-- Flow Matching Decoder (causal UNet, 12 mid-blocks)
  +-- HiFT Vocoder (mel to waveform, F0 predictor + source-filter synthesis)
    |
    v
24 kHz mono WAV output
```

The turbo model uses meanflow mode (2 ODE solver steps) for faster S3Gen inference versus the standard model (5 steps).
