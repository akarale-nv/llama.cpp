/**
 * llama-t3: Chatterbox-Turbo T3 speech token generation + S3Gen vocoder
 *
 * Required:
 *   -m PATH             t3_embeddings.gguf  (text + speech embeddings + GPT-2 tokenizer)
 *   --model-llama PATH  t3_backbone_f16.gguf  (GPT-2 24-layer transformer backbone)
 *   --prompt TEXT       text to synthesise
 *   --cond-emb PATH     Speaker conditioning JSON
 *
 * Optional:
 *   --s3gen PATH     S3Gen vocoder model (enables audio output)
 *   --output PATH    output WAV file (default: output.wav)
 */


#include "llama.h"
#include "gguf.h"
#include "ggml.h"
#include "s3gen.h"

#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static const int SPEECH_VOCAB   = 6563;
static const int TEXT_VOCAB     = 50276;
static const int N_EMBD         = 1024;
static const int SPEECH_SOS     = 6561;
static const int SPEECH_EOS     = 6562;

// ---------------------------------------------------------------------------
// Section 1: Tokenizer (via llama.cpp vocab, loaded from t3_embeddings.gguf)
// ---------------------------------------------------------------------------

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> ids(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                           ids.data(), (int32_t)ids.size(),
                           /*add_special=*/true, /*parse_special=*/true);
    if (n < 0) {
        // buffer too small, resize and retry
        ids.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                           ids.data(), (int32_t)ids.size(),
                           /*add_special=*/true, /*parse_special=*/true);
    }
    ids.resize(n < 0 ? 0 : n);
    return ids;
}

// ---------------------------------------------------------------------------
// Section 2: GGUF Embedding Loader
// ---------------------------------------------------------------------------

static const int COND_LEN     = 376;  // 1 (spkr_enc) + 375 (speech prompt tokens)

struct T3Embeddings {
    std::vector<float> text_emb;      // [TEXT_VOCAB * N_EMBD] float32
    std::vector<float> speech_emb;    // [SPEECH_VOCAB * N_EMBD] float32
};

static bool load_embeddings(const char * gguf_path, T3Embeddings & emb) {
    struct ggml_context * ggml_ctx = nullptr;
    struct gguf_init_params params = { /*no_alloc=*/false, /*ctx=*/&ggml_ctx };
    struct gguf_context * ctx = gguf_init_from_file(gguf_path, params);
    if (!ctx || !ggml_ctx) {
        fprintf(stderr, "Error: failed to load embeddings from %s\n", gguf_path);
        return false;
    }

    struct ggml_tensor * text_t   = ggml_get_tensor(ggml_ctx, "text_emb_weight");
    struct ggml_tensor * speech_t = ggml_get_tensor(ggml_ctx, "speech_emb_weight");
    if (!text_t || !speech_t) {
        fprintf(stderr, "Error: text_emb_weight or speech_emb_weight not found in %s\n", gguf_path);
        gguf_free(ctx);
        ggml_free(ggml_ctx);
        return false;
    }

    // text_emb_weight: shape [N_EMBD, TEXT_VOCAB] in ggml (ne[0]=N_EMBD, ne[1]=TEXT_VOCAB)
    // speech_emb_weight: shape [N_EMBD, SPEECH_VOCAB]
    int64_t text_n   = text_t->ne[1];    // number of rows (vocab size)
    int64_t speech_n = speech_t->ne[1];
    int64_t n_embd   = text_t->ne[0];    // embedding dim

    emb.text_emb.resize(text_n * n_embd);
    emb.speech_emb.resize(speech_n * n_embd);

    auto convert_f16 = [](const ggml_fp16_t * src, float * dst, size_t n) {
        for (size_t i = 0; i < n; ++i) dst[i] = ggml_fp16_to_fp32(src[i]);
    };

    if (text_t->type == GGML_TYPE_F16) {
        convert_f16((const ggml_fp16_t *)text_t->data, emb.text_emb.data(), text_n * n_embd);
    } else if (text_t->type == GGML_TYPE_F32) {
        memcpy(emb.text_emb.data(), text_t->data, text_n * n_embd * sizeof(float));
    } else {
        fprintf(stderr, "Error: unsupported type for text_emb_weight\n");
        gguf_free(ctx); ggml_free(ggml_ctx); return false;
    }

    if (speech_t->type == GGML_TYPE_F16) {
        convert_f16((const ggml_fp16_t *)speech_t->data, emb.speech_emb.data(), speech_n * n_embd);
    } else if (speech_t->type == GGML_TYPE_F32) {
        memcpy(emb.speech_emb.data(), speech_t->data, speech_n * n_embd * sizeof(float));
    } else {
        fprintf(stderr, "Error: unsupported type for speech_emb_weight\n");
        gguf_free(ctx); ggml_free(ggml_ctx); return false;
    }

    gguf_free(ctx);
    ggml_free(ggml_ctx);
    return true;
}

// ---------------------------------------------------------------------------
// Section 3: Speaker Conditioning JSON Loader (minimal, no external deps)
// ---------------------------------------------------------------------------

// Base64 decode (standard alphabet)
static const std::string B64_CHARS =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::vector<uint8_t> base64_decode(const std::string & in) {
    std::vector<uint8_t> out;
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; ++i) T[(unsigned char)B64_CHARS[i]] = i;

    int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (T[c] == -1) break;
        val  = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back((uint8_t)((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

struct SpeakerEmb {
    std::vector<float> data;     // [len_cond * N_EMBD] — T3 prefill conditioning
    int len_cond = 0;
    std::vector<float> spk_emb;  // [192] — raw ECAPA speaker embedding for S3Gen
    std::vector<int32_t> prompt_tokens;   // prompt token IDs for S3Gen voice cloning
    std::vector<float>   prompt_features; // [mel_ch * prompt_feat_frames] flat, channels-first
    int prompt_feat_frames = 0;           // number of mel frames in prompt_features
};

// Minimal JSON parser: finds "cond_emb" object with "bin_data" and "shape"
static bool load_speaker_emb(const char * path, SpeakerEmb & spk) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "Error: cannot open %s\n", path); return false; }
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // Extract shape[1] (len_cond)
    // Look for "shape": [X, Y, Z] or "shape":[X,Y,Z]
    auto find_shape = [&]() -> int {
        size_t pos = json.find("\"shape\"");
        if (pos == std::string::npos) return -1;
        pos = json.find('[', pos);
        if (pos == std::string::npos) return -1;
        // skip first number, read second
        pos = json.find(',', pos);
        if (pos == std::string::npos) return -1;
        ++pos;
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r')) ++pos;
        return std::atoi(json.c_str() + pos);
    };

    // Extract bin_data string value
    auto find_bin_data = [&]() -> std::string {
        size_t pos = json.find("\"bin_data\"");
        if (pos == std::string::npos) return "";
        pos = json.find('"', pos + 10);
        if (pos == std::string::npos) return "";
        size_t start = pos + 1;
        size_t end   = json.find('"', start);
        if (end == std::string::npos) return "";
        return json.substr(start, end - start);
    };

    int len_cond = find_shape();
    if (len_cond <= 0) {
        fprintf(stderr, "Error: could not parse shape from %s\n", path);
        return false;
    }

    std::string b64 = find_bin_data();
    if (b64.empty()) {
        // Try "data" key as fallback
        size_t pos = json.find("\"data\"");
        if (pos != std::string::npos) {
            pos = json.find('"', pos + 6);
            if (pos != std::string::npos) {
                size_t start = pos + 1;
                size_t end   = json.find('"', start);
                if (end != std::string::npos) b64 = json.substr(start, end - start);
            }
        }
    }
    if (b64.empty()) {
        // cond_emb might be stored as a flat JSON float array
        size_t pos = json.find("\"cond_emb\"");
        if (pos == std::string::npos) { fprintf(stderr, "Error: cond_emb not found in %s\n", path); return false; }
        pos = json.find('[', pos);
        if (pos == std::string::npos) { fprintf(stderr, "Error: cond_emb array not found\n"); return false; }
        // Parse float array
        ++pos;
        while (pos < json.size() && json[pos] != ']') {
            while (pos < json.size() && (json[pos] == ' ' || json[pos] == ',' || json[pos] == '\n' || json[pos] == '\r')) ++pos;
            if (pos < json.size() && json[pos] != ']') {
                spk.data.push_back((float)std::atof(json.c_str() + pos));
                while (pos < json.size() && json[pos] != ',' && json[pos] != ']') ++pos;
            }
        }
        spk.len_cond = len_cond;
        return !spk.data.empty();
    }

    auto bytes = base64_decode(b64);
    if (bytes.size() % sizeof(float) != 0) {
        fprintf(stderr, "Error: base64 data size %zu not aligned to float\n", bytes.size());
        return false;
    }
    size_t n_floats = bytes.size() / sizeof(float);
    spk.data.resize(n_floats);
    memcpy(spk.data.data(), bytes.data(), bytes.size());
    spk.len_cond = len_cond;

    fprintf(stderr, "Loaded speaker emb: len_cond=%d, total=%zu floats\n", len_cond, n_floats);

    // Helper: extract base64-encoded bin_data for a named JSON key
    auto extract_bin_data = [&](const char * key) -> std::vector<uint8_t> {
        size_t pos = json.find(std::string("\"") + key + "\"");
        if (pos == std::string::npos) return {};
        size_t bd_pos = json.find("\"bin_data\"", pos);
        if (bd_pos == std::string::npos) return {};
        size_t q1 = json.find('"', bd_pos + 10);
        if (q1 == std::string::npos) return {};
        size_t s2 = q1 + 1;
        size_t e2 = json.find('"', s2);
        if (e2 == std::string::npos) return {};
        return base64_decode(json.substr(s2, e2 - s2));
    };

    // Extract speaker embedding for S3Gen (key "spk_emb" or "embedding")
    {
        auto sbytes = extract_bin_data("spk_emb");
        if (sbytes.empty()) sbytes = extract_bin_data("embedding");
        if (!sbytes.empty() && sbytes.size() % sizeof(float) == 0) {
            spk.spk_emb.resize(sbytes.size() / sizeof(float));
            memcpy(spk.spk_emb.data(), sbytes.data(), sbytes.size());
            fprintf(stderr, "  spk_emb: %zu floats\n", spk.spk_emb.size());
        }
    }

    // Extract prompt_token for S3Gen (int64 in JSON, convert to int32)
    {
        auto tbytes = extract_bin_data("prompt_token");
        if (!tbytes.empty() && tbytes.size() % sizeof(int64_t) == 0) {
            size_t n = tbytes.size() / sizeof(int64_t);
            spk.prompt_tokens.resize(n);
            const int64_t * src = reinterpret_cast<const int64_t *>(tbytes.data());
            for (size_t i = 0; i < n; i++) {
                spk.prompt_tokens[i] = (int32_t)src[i];
            }
            fprintf(stderr, "  prompt_token: %zu tokens\n", n);
        }
    }

    // Extract prompt_feat for S3Gen (float32, shape [1, frames, 80])
    // Python layout [frames, 80] maps directly to ggml tensor ne[0]=80, ne[1]=frames
    // because Python's innermost dimension (80 channels) matches ggml's ne[0]. No transpose needed.
    {
        auto fbytes = extract_bin_data("prompt_feat");
        if (!fbytes.empty() && fbytes.size() % sizeof(float) == 0) {
            size_t n_floats = fbytes.size() / sizeof(float);
            int n_frames = (int)(n_floats / 80);
            if (n_frames > 0 && n_frames * 80 == (int)n_floats) {
                spk.prompt_feat_frames = n_frames;
                spk.prompt_features.resize(n_floats);
                memcpy(spk.prompt_features.data(), fbytes.data(), fbytes.size());
                fprintf(stderr, "  prompt_feat: %d frames x 80 channels\n", n_frames);
            }
        }
    }

    return true;
}



// ---------------------------------------------------------------------------
// Section 4: Prefill Embedding Construction
// ---------------------------------------------------------------------------
// Layout: [cond_emb (len_cond tokens) || text_embs (n_text tokens) || speech_SOS (1 token)]
// Each token = N_EMBD contiguous floats (row-major, matches llama_batch.embd)

static std::vector<float> build_prefill(
        const SpeakerEmb    & spk,
        const std::vector<int32_t> & text_ids,
        const T3Embeddings  & emb) {
    int n_cond  = spk.len_cond;
    int n_text  = (int)text_ids.size();
    int total   = n_cond + n_text + 1;
    std::vector<float> buf(total * N_EMBD);
    float * dst = buf.data();

    // Cond emb: spk.data is [n_cond * N_EMBD] row-major
    memcpy(dst, spk.data.data(), n_cond * N_EMBD * sizeof(float));
    dst += n_cond * N_EMBD;

    // Text embeddings
    for (int id : text_ids) {
        if (id < 0 || id >= TEXT_VOCAB) {
            memset(dst, 0, N_EMBD * sizeof(float));
        } else {
            memcpy(dst, emb.text_emb.data() + id * N_EMBD, N_EMBD * sizeof(float));
        }
        dst += N_EMBD;
    }

    // Speech SOS embedding
    memcpy(dst, emb.speech_emb.data() + SPEECH_SOS * N_EMBD, N_EMBD * sizeof(float));

    return buf;
}

// ---------------------------------------------------------------------------
// Section 5: WAV writer
// ---------------------------------------------------------------------------

static void write_wav(const char * path, const std::vector<float> & samples, int sample_rate) {
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "write_wav: cannot open %s\n", path); return; }

    int32_t data_size   = (int32_t)(samples.size() * sizeof(int16_t));
    int32_t chunk_size  = 36 + data_size;
    int16_t num_ch      = 1;
    int32_t byte_rate   = sample_rate * 2;
    int16_t block_align = 2;
    int16_t bits        = 16;
    int16_t audio_fmt   = 1; // PCM
    int32_t subchunk1   = 16;

    fwrite("RIFF", 1, 4, f);
    fwrite(&chunk_size,  4, 1, f);
    fwrite("WAVEfmt ", 1, 8, f);
    fwrite(&subchunk1,   4, 1, f);
    fwrite(&audio_fmt,   2, 1, f);
    fwrite(&num_ch,      2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate,   4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits,        2, 1, f);
    fwrite("data",       1, 4, f);
    fwrite(&data_size,   4, 1, f);

    for (float s : samples) {
        float c = s < -1.0f ? -1.0f : (s > 1.0f ? 1.0f : s);
        int16_t v = (int16_t)(c * 32767.0f);
        fwrite(&v, 2, 1, f);
    }
    fclose(f);
    fprintf(stderr, "Wrote %zu samples (%.2f s) to %s\n",
            samples.size(), (double)samples.size() / sample_rate, path);
}

// ---------------------------------------------------------------------------
// Section 6: llama.cpp Inference Loop
// ---------------------------------------------------------------------------

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "Usage: %s -m <t3_embeddings.gguf> --model-llama <t3_backbone.gguf>"
        " --prompt TEXT --cond-emb PATH [options]\n"
        "\nRequired:\n"
        "  -m PATH              T3 embedding model (t3_embeddings.gguf)\n"
        "  --model-llama PATH   T3 backbone model  (t3_backbone_f16.gguf)\n"
        "  --prompt TEXT        Text to synthesise\n"
        "  --cond-emb PATH      Speaker conditioning JSON\n"
        "\nVocoder (optional — enables audio output):\n"
        "  --s3gen PATH         S3Gen vocoder model (s3gen_f16.gguf)\n"
        "  --output PATH        Output WAV file (default: output.wav)\n"
        "  --tokens-out PATH    Write speech tokens to file (one per line)\n"
        "  --tokens-in T1,T2,.. Comma-separated token list to feed directly to S3Gen (skips T3)\n"
        "\nOptions:\n"
        "  --max-tokens N       Max speech tokens to generate (default: 2000)\n"
        "  --temperature N      Sampling temperature (default: 1.0)\n"
        "  --top-p N            Top-p (default: 0.95)\n"
        "  --min-p N            Min-p (default: 0.05)\n"
        "  --rep-penalty N      Repetition penalty (default: 1.2)\n"
        "  --seed N             Random seed (-1 = random, default: -1)\n"
        "  --ngl N              GPU layers to offload (default: 99)\n"
        "  -h, --help           Show this help\n",
        argv0);
}

struct Args {
    std::string emb_model_path;
    std::string backbone_path;
    std::string prompt;
    std::string cond_emb_path;
    std::string s3gen_path;
    std::string output_path   = "output.wav";
    std::string tokens_path;
    std::string tokens_in_path;
    int   max_tokens      = 2000;
    float temperature     = 1.0f;
    float top_p           = 0.95f;
    float min_p           = 0.05f;
    float rep_penalty     = 1.2f;
    int   seed            = -1;
    int   n_gpu_layers    = 99;
};

static bool parse_args(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) return false;
#define STR_ARG(flag, field) if (!strcmp(argv[i], flag) && i+1<argc) { a.field = argv[++i]; continue; }
#define INT_ARG(flag, field) if (!strcmp(argv[i], flag) && i+1<argc) { a.field = std::atoi(argv[++i]); continue; }
#define FLT_ARG(flag, field) if (!strcmp(argv[i], flag) && i+1<argc) { a.field = (float)std::atof(argv[++i]); continue; }
        STR_ARG("-m",            emb_model_path)
        STR_ARG("--model-llama", backbone_path)
        STR_ARG("--prompt",      prompt)
        STR_ARG("--cond-emb",    cond_emb_path)
        STR_ARG("--s3gen",       s3gen_path)
        STR_ARG("--output",      output_path)
        STR_ARG("--tokens-out",  tokens_path)
        STR_ARG("--tokens-in",   tokens_in_path)
        INT_ARG("--max-tokens",  max_tokens)
        FLT_ARG("--temperature", temperature)
        FLT_ARG("--top-p",       top_p)
        FLT_ARG("--min-p",       min_p)
        FLT_ARG("--rep-penalty", rep_penalty)
        INT_ARG("--seed",        seed)
        INT_ARG("--ngl",         n_gpu_layers)
#undef STR_ARG
#undef INT_ARG
#undef FLT_ARG
        fprintf(stderr, "Unknown argument: %s\n", argv[i]);
        return false;
    }
    if (a.tokens_in_path.empty()) {
        if (a.emb_model_path.empty() || a.backbone_path.empty() ||
            a.prompt.empty()         || a.cond_emb_path.empty()) {
            fprintf(stderr, "Error: -m, --model-llama, --prompt, and --cond-emb are required\n");
            return false;
        }
    } else {
        if (a.cond_emb_path.empty()) {
            fprintf(stderr, "Error: --cond-emb is required with --tokens-in\n");
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    // ------------------------------------------------------------------
    // tokens-in shortcut: skip T3, feed tokens directly to S3Gen
    // ------------------------------------------------------------------
    if (!args.tokens_in_path.empty()) {
        std::vector<int> speech_tokens;
        const std::string & s = args.tokens_in_path;
        for (size_t i = 0; i < s.size(); ) {
            size_t end = s.find(',', i);
            if (end == std::string::npos) end = s.size();
            if (end > i) speech_tokens.push_back(std::stoi(s.substr(i, end - i)));
            i = end + 1;
        }
        fprintf(stderr, "Loaded %zu tokens\n", speech_tokens.size());

        SpeakerEmb spk;
        if (!load_speaker_emb(args.cond_emb_path.c_str(), spk)) return 1;

        if (args.s3gen_path.empty()) {
            fprintf(stderr, "Error: --s3gen is required with --tokens-in\n");
            return 1;
        }
        if (spk.spk_emb.empty()) {
            fprintf(stderr, "Error: no spk_emb found in '%s'\n", args.cond_emb_path.c_str());
            return 1;
        }

        fprintf(stderr, "Loading S3Gen: %s\n", args.s3gen_path.c_str());
        S3Token2Wav s3gen;
        if (!s3gen.init_backend() || !s3gen.load_model(args.s3gen_path, true)) {
            fprintf(stderr, "Error: failed to load S3Gen\n"); return 1;
        }

        std::vector<int> toks;
        for (int t : speech_tokens)
            if (t >= 0 && t < SPEECH_SOS) toks.push_back(t);
        fprintf(stderr, "Running S3Gen on %zu tokens...\n", toks.size());

        // Prepare GenerateParams
        std::vector<int64_t> pt64(spk.prompt_tokens.begin(), spk.prompt_tokens.end());
        int32_t pt_shape[] = {1, (int32_t)pt64.size()};
        int32_t pf_shape[] = {80, (int32_t)spk.prompt_feat_frames};
        int32_t emb_shape[] = {(int32_t)spk.spk_emb.size()};

        S3Token2Wav::GenerateParams gp;
        gp.prompt_tokens = pt64.data();
        gp.prompt_tokens_shape = pt_shape;
        gp.prompt_tokens_n_dims = 2;
        gp.prompt_feat = spk.prompt_features.data();
        gp.prompt_feat_shape = pf_shape;
        gp.prompt_feat_n_dims = 2;
        gp.speaker_emb = spk.spk_emb.data();
        gp.speaker_emb_shape = emb_shape;
        gp.speaker_emb_n_dims = 1;

        float* audio_data = nullptr;
        size_t audio_len = 0;
        if (!s3gen.generate(toks, gp, &audio_data, &audio_len) || !audio_data) {
            fprintf(stderr, "Error: S3Gen produced no audio\n"); return 1;
        }

        std::vector<float> audio(audio_data, audio_data + audio_len);
        free(audio_data);

        write_wav(args.output_path.c_str(), audio, 24000);
        return 0;
    }

    // ------------------------------------------------------------------
    // 1. Load tokenizer + embeddings
    // ------------------------------------------------------------------
    // Load tokenizer from the backbone (gpt2 architecture, known to llama.cpp).
    // The backbone sets tokenizer.ggml.model=no_vocab for inference, but the full
    // 50257-token BPE data is still present. Override to "gpt2" for vocab-only load.
    fprintf(stderr, "Loading tokenizer from: %s\n", args.backbone_path.c_str());
    llama_model_kv_override tok_override[2] = {};
    tok_override[0].tag = LLAMA_KV_OVERRIDE_TYPE_STR;
    snprintf(tok_override[0].key, sizeof(tok_override[0].key), "tokenizer.ggml.model");
    snprintf(tok_override[0].val_str, sizeof(tok_override[0].val_str), "gpt2");
    tok_override[1].key[0] = '\0';  // sentinel

    llama_model_params vocab_mparams = llama_model_default_params();
    vocab_mparams.vocab_only   = true;
    vocab_mparams.kv_overrides = tok_override;
    llama_model * vocab_model = llama_model_load_from_file(args.backbone_path.c_str(), vocab_mparams);
    if (!vocab_model) {
        fprintf(stderr, "Error: failed to load tokenizer from %s\n", args.backbone_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(vocab_model);
    fprintf(stderr, "  vocab size: %d\n", llama_vocab_n_tokens(vocab));

    fprintf(stderr, "Loading embeddings...\n");
    T3Embeddings emb;
    if (!load_embeddings(args.emb_model_path.c_str(), emb)) {
        llama_model_free(vocab_model);
        return 1;
    }
    fprintf(stderr, "  text_emb: %zu floats, speech_emb: %zu floats\n",
            emb.text_emb.size(), emb.speech_emb.size());

    // ------------------------------------------------------------------
    // 2. Tokenize prompt
    // ------------------------------------------------------------------
    auto text_ids = tokenize(vocab, args.prompt);
    fprintf(stderr, "Text tokens (%zu):\n", text_ids.size());
    for (size_t i = 0; i < text_ids.size(); i++) {
        char buf[128] = {0};
        int len = llama_token_to_piece(vocab, text_ids[i], buf, sizeof(buf) - 1, 0, /*special=*/true);
        if (len < 0) len = 0;
        buf[len] = '\0';
        fprintf(stderr, "  [%3zu] id=%5d  '%s'\n", i, text_ids[i], buf);
    }
    fprintf(stderr, "\n");

    // ------------------------------------------------------------------
    // 3. Load speaker conditioning
    // ------------------------------------------------------------------
    SpeakerEmb spk;
    fprintf(stderr, "Loading speaker emb: %s\n", args.cond_emb_path.c_str());
    if (!load_speaker_emb(args.cond_emb_path.c_str(), spk)) {
        llama_model_free(vocab_model);
        return 1;
    }

    // ------------------------------------------------------------------
    // 4. Build prefill embeddings
    // ------------------------------------------------------------------
    auto prefill = build_prefill(spk, text_ids, emb);
    int total_prefill = spk.len_cond + (int)text_ids.size() + 1;
    fprintf(stderr, "Prefill: %d tokens (cond=%d, text=%d, sos=1)\n",
            total_prefill, spk.len_cond, (int)text_ids.size());

    // ------------------------------------------------------------------
    // 5. Load backbone model
    // ------------------------------------------------------------------
    fprintf(stderr, "Loading backbone: %s\n", args.backbone_path.c_str());
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(args.backbone_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "Error: failed to load backbone model\n");
        return 1;
    }

    int n_ctx_needed = total_prefill + args.max_tokens + 64;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = (uint32_t)n_ctx_needed;
    cparams.n_batch   = (uint32_t)std::max(total_prefill, 512);
    cparams.n_seq_max = 1;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Error: failed to create llama context\n");
        llama_model_free(model);
        return 1;
    }

    // ------------------------------------------------------------------
    // 6. Sampler chain
    // ------------------------------------------------------------------
    uint32_t seed = (args.seed < 0) ? (uint32_t)time(nullptr) : (uint32_t)args.seed;
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, args.rep_penalty, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(args.temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(1));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));

    // ------------------------------------------------------------------
    // 7. Prefill pass: inject embedding sequence
    // ------------------------------------------------------------------
    // llama_batch_init with embd!=0 allocates batch.embd (and does NOT allocate batch.token)
    llama_batch batch = llama_batch_init(total_prefill, N_EMBD, 1);
    batch.n_tokens = total_prefill;
    // Copy prefill embeddings into batch.embd
    memcpy(batch.embd, prefill.data(), total_prefill * N_EMBD * sizeof(float));
    // Clear batch.token — we are providing embeddings directly
    batch.token = nullptr;
    for (int i = 0; i < total_prefill; ++i) {
        batch.pos[i]        = i;
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = 0;
        batch.logits[i]     = (i == total_prefill - 1) ? 1 : 0;
    }

    fprintf(stderr, "Running prefill (%d tokens)...\n", total_prefill);
    auto t0 = std::chrono::high_resolution_clock::now();

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Error: prefill llama_decode() failed\n");
        llama_batch_free(batch);
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    llama_batch_free(batch);

    // ------------------------------------------------------------------
    // 8. Sample first speech token from prefill logits
    // ------------------------------------------------------------------
    llama_token next_token = llama_sampler_sample(smpl, ctx, total_prefill - 1);

    // ------------------------------------------------------------------
    // 9. Autoregressive decode loop
    // ------------------------------------------------------------------
    std::vector<int> speech_tokens;

    for (int step = 0; step < args.max_tokens; ++step) {
        if (next_token == SPEECH_EOS) break;
        speech_tokens.push_back(next_token);

        llama_batch step_batch = llama_batch_get_one(&next_token, 1);
        if (llama_decode(ctx, step_batch) != 0) {
            fprintf(stderr, "Error: decode step %d failed\n", step);
            break;
        }

        next_token = llama_sampler_sample(smpl, ctx, 0);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_model_free(vocab_model);

    // ------------------------------------------------------------------
    // 10. Output speech tokens
    // ------------------------------------------------------------------
    fprintf(stderr, "\nGenerated %zu speech tokens in %.2f s (%.1f tok/s)\n",
            speech_tokens.size(), elapsed,
            speech_tokens.empty() ? 0.0 : speech_tokens.size() / elapsed);

    printf("[");
    for (size_t i = 0; i < speech_tokens.size(); ++i) {
        if (i) printf(", ");
        printf("%d", speech_tokens[i]);
    }
    printf("]\n");

    if (!args.tokens_path.empty()) {
        FILE * tf = fopen(args.tokens_path.c_str(), "w");
        if (!tf) {
            fprintf(stderr, "Warning: cannot open tokens file '%s'\n", args.tokens_path.c_str());
        } else {
            for (size_t i = 0; i < speech_tokens.size(); ++i) {
                if (i) fprintf(tf, "\n");
                fprintf(tf, "%d", speech_tokens[i]);
            }
            fprintf(tf, "\n");
            fclose(tf);
            fprintf(stderr, "Wrote %zu tokens to %s\n", speech_tokens.size(), args.tokens_path.c_str());
        }
    }

    // ------------------------------------------------------------------
    // 11. S3Gen vocoder (optional)
    // ------------------------------------------------------------------
    if (!args.s3gen_path.empty() && !speech_tokens.empty()) {
        if (spk.spk_emb.empty()) {
            fprintf(stderr, "Warning: --s3gen specified but no spk_emb found in JSON; skipping vocoder\n");
        } else {
            fprintf(stderr, "Loading S3Gen: %s\n", args.s3gen_path.c_str());
            S3Token2Wav s3gen;
            if (!s3gen.init_backend() || !s3gen.load_model(args.s3gen_path, true)) {
                fprintf(stderr, "Error: failed to load S3Gen model\n");
                return 1;
            }

            // Filter speech tokens: keep only valid range [0, 6561)
            std::vector<int> toks;
            toks.reserve(speech_tokens.size());
            for (int t : speech_tokens) {
                if (t >= 0 && t < SPEECH_SOS) toks.push_back(t);
            }
            fprintf(stderr, "Running S3Gen on %zu tokens...\n", toks.size());

            // Prepare GenerateParams
            std::vector<int64_t> pt64(spk.prompt_tokens.begin(), spk.prompt_tokens.end());
            int32_t pt_shape[] = {1, (int32_t)pt64.size()};
            int32_t pf_shape[] = {80, (int32_t)spk.prompt_feat_frames};
            int32_t emb_shape[] = {(int32_t)spk.spk_emb.size()};

            S3Token2Wav::GenerateParams gp;
            gp.prompt_tokens = pt64.data();
            gp.prompt_tokens_shape = pt_shape;
            gp.prompt_tokens_n_dims = 2;
            gp.prompt_feat = spk.prompt_features.data();
            gp.prompt_feat_shape = pf_shape;
            gp.prompt_feat_n_dims = 2;
            gp.speaker_emb = spk.spk_emb.data();
            gp.speaker_emb_shape = emb_shape;
            gp.speaker_emb_n_dims = 1;

            float* audio_data = nullptr;
            size_t audio_len = 0;
            if (!s3gen.generate(toks, gp, &audio_data, &audio_len) || !audio_data) {
                fprintf(stderr, "Error: S3Gen produced no audio\n");
                return 1;
            }

            std::vector<float> audio(audio_data, audio_data + audio_len);
            free(audio_data);

            write_wav(args.output_path.c_str(), audio, 24000);
        }
    }

    return 0;
}
