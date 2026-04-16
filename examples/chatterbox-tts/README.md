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

### 2. Obtain model files

The Chatterbox Turbo pipeline uses two separate model checkpoints:

- **T3 checkpoint** -- contains the GPT-2 backbone (`tfmr.*`), text/speech embeddings (`text_emb.weight`, `speech_emb.weight`), positional embeddings, and speech head (`speech_head.weight`/`bias`)
- **S3Gen checkpoint** -- contains the flow-matching vocoder (`flow.*`), HiFT decoder (`mel2wav.*`), and speaker encoder (`speaker_encoder.*`)

You also need the tokenizer files alongside the T3 checkpoint:
- `tokenizer.json` (or `vocab.json` + `merges.txt`)
- `tokenizer_config.json`

### 3. Convert to GGUF

Convert both models in a single command:

```bash
python examples/chatterbox-tts/convert_chatterbox_to_gguf.py all \
    path/to/t3_model.safetensors \
    path/to/s3gen_model.safetensors \
    --model-dir path/to/tokenizer_dir/ \
    --quantize f16
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
    --prompt "Hello, how are you doing today?" \
    --cond-emb speaker.json \
    --s3gen s3gen_f16.gguf \
    --output output.wav
```

**Token generation only (no vocoder):**

```bash
./build/bin/llama-chatterbox \
    -m t3_embeddings.gguf \
    --model-llama chatterbox_llama_backbone_tts_f16.gguf \
    --prompt "Hello, how are you doing today?" \
    --cond-emb speaker.json \
    --tokens-out tokens.txt
```

**Vocoder only (from pre-generated tokens):**

```bash
./build/bin/llama-chatterbox \
    --cond-emb speaker.json \
    --s3gen s3gen_f16.gguf \
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

Converts both T3 and S3Gen checkpoints in one command, producing three GGUF files.

```
python convert_chatterbox_to_gguf.py all <t3_model> <s3gen_model> [options]

  t3_model                    Path to T3 .safetensors or .pth checkpoint
  s3gen_model                 Path to S3Gen .safetensors or .pth checkpoint

  --model-dir DIR             Directory with tokenizer files (default: parent of t3_model)
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
