#include "voice_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <utility>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

// ============================================================================
// Tap dumping (enabled by env var VE_DEBUG_TAPS pointing to output file path)
// ============================================================================
// Binary layout (little-endian):
//   char magic[4] = "CBTP"; uint32_t version=1; uint32_t n_taps;
//   Per tap:
//     uint32_t name_len; char name[name_len];
//     uint32_t dtype;              // 0=f32, 1=i32
//     uint32_t n_dims;             // 1..4
//     int64_t  shape[4];           // trailing dims = 1
//     int64_t  n_bytes;
//     uint8_t  data[n_bytes];
struct TapDumper {
    FILE * fp = nullptr;
    uint32_t n_taps = 0;
    long n_taps_offset = 0;   // where to patch n_taps after we finish

    // Filled by mark_tap during graph construction. encode_reference then
    // walks this list to (a) forward-expand each tap into the graph, and
    // (b) read them back into the file after compute.
    struct Pending { std::string name; ggml_tensor * tensor; };
    std::vector<Pending> pending;

    bool enabled() const { return fp != nullptr; }

    void open_from_env() {
        const char * path = std::getenv("VE_DEBUG_TAPS");
        if (!path || !*path) return;
        fp = std::fopen(path, "wb");
        if (!fp) { fprintf(stderr, "TapDumper: cannot open %s\n", path); return; }
        std::fwrite("CBTP", 1, 4, fp);
        uint32_t version = 1;
        std::fwrite(&version, sizeof(version), 1, fp);
        n_taps_offset = std::ftell(fp);
        std::fwrite(&n_taps, sizeof(n_taps), 1, fp);
        fprintf(stderr, "TapDumper: writing taps to %s\n", path);
    }

    void write_raw(const std::string & name, uint32_t dtype,
                   const std::vector<int64_t> & shape,
                   const void * data, size_t n_bytes) {
        if (!fp) return;
        uint32_t name_len = (uint32_t)name.size();
        std::fwrite(&name_len, sizeof(name_len), 1, fp);
        std::fwrite(name.data(), 1, name_len, fp);
        std::fwrite(&dtype, sizeof(dtype), 1, fp);
        uint32_t n_dims = (uint32_t)shape.size();
        std::fwrite(&n_dims, sizeof(n_dims), 1, fp);
        int64_t sh4[4] = {1, 1, 1, 1};
        for (uint32_t i = 0; i < n_dims && i < 4; ++i) sh4[i] = shape[i];
        std::fwrite(sh4, sizeof(int64_t), 4, fp);
        int64_t nb = (int64_t)n_bytes;
        std::fwrite(&nb, sizeof(nb), 1, fp);
        std::fwrite(data, 1, n_bytes, fp);
        n_taps++;
    }

    // Convenience: capture a ggml tensor from backend memory.
    void tap_tensor(const std::string & name, ggml_tensor * t) {
        if (!fp) return;
        std::vector<int64_t> shape;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (t->ne[i] == 1 && shape.size() >= 1 && i > 0) continue;
            shape.push_back(t->ne[i]);
        }
        while (shape.size() < 1) shape.push_back(1);
        uint32_t dtype = (t->type == GGML_TYPE_I32) ? 1u : 0u;
        std::vector<uint8_t> host(ggml_nbytes(t));
        ggml_backend_tensor_get(t, host.data(), 0, host.size());
        write_raw(name, dtype, shape, host.data(), host.size());
        fprintf(stderr, "  tap %-28s shape=[", name.c_str());
        for (size_t i = 0; i < shape.size(); ++i) fprintf(stderr, "%s%lld", i ? ", " : "", (long long)shape[i]);
        fprintf(stderr, "] dtype=%s nbytes=%zu\n", dtype ? "i32" : "f32", host.size());
    }

    void tap_f32(const std::string & name, const float * data, const std::vector<int64_t> & shape) {
        size_t n = 1; for (auto s : shape) n *= (size_t)s;
        write_raw(name, 0, shape, data, n * sizeof(float));
        fprintf(stderr, "  tap %-28s shape=[", name.c_str());
        for (size_t i = 0; i < shape.size(); ++i) fprintf(stderr, "%s%lld", i ? ", " : "", (long long)shape[i]);
        fprintf(stderr, "] dtype=f32 nbytes=%zu\n", n * sizeof(float));
    }

    void tap_i32(const std::string & name, const int32_t * data, const std::vector<int64_t> & shape) {
        size_t n = 1; for (auto s : shape) n *= (size_t)s;
        write_raw(name, 1, shape, data, n * sizeof(int32_t));
        fprintf(stderr, "  tap %-28s shape=[", name.c_str());
        for (size_t i = 0; i < shape.size(); ++i) fprintf(stderr, "%s%lld", i ? ", " : "", (long long)shape[i]);
        fprintf(stderr, "] dtype=i32 nbytes=%zu\n", n * sizeof(int32_t));
    }

    void close() {
        if (!fp) return;
        // Patch n_taps at header offset.
        std::fseek(fp, n_taps_offset, SEEK_SET);
        std::fwrite(&n_taps, sizeof(n_taps), 1, fp);
        std::fclose(fp);
        fp = nullptr;
        fprintf(stderr, "TapDumper: wrote %u taps\n", n_taps);
    }
};

#ifndef VE_PI
#define VE_PI 3.14159265358979323846f
#endif

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

// ============================================================================
// Section 1: GGUF KV helpers + config loading
// ============================================================================

static int32_t kv_u32(gguf_context * gc, const char * key, int32_t defval) {
    int64_t id = gguf_find_key(gc, key);
    return id < 0 ? defval : (int32_t)gguf_get_val_u32(gc, id);
}
static float   kv_f32(gguf_context * gc, const char * key, float defval) {
    int64_t id = gguf_find_key(gc, key);
    return id < 0 ? defval : gguf_get_val_f32(gc, id);
}
static bool    kv_bool(gguf_context * gc, const char * key, bool defval) {
    int64_t id = gguf_find_key(gc, key);
    return id < 0 ? defval : gguf_get_val_bool(gc, id);
}
static std::string kv_str(gguf_context * gc, const char * key, const std::string & defval) {
    int64_t id = gguf_find_key(gc, key);
    if (id < 0) return defval;
    return std::string(gguf_get_val_str(gc, id));
}

static VoiceEncoderConfig load_ve_config(gguf_context * gc) {
    VoiceEncoderConfig c;
    c.num_mels            = kv_u32 (gc, "clip.audio.ve.num_mels",           c.num_mels);
    c.sample_rate         = kv_u32 (gc, "clip.audio.ve.sample_rate",        c.sample_rate);
    c.n_fft               = kv_u32 (gc, "clip.audio.ve.n_fft",              c.n_fft);
    c.hop_size            = kv_u32 (gc, "clip.audio.ve.hop_size",           c.hop_size);
    c.win_size            = kv_u32 (gc, "clip.audio.ve.win_size",           c.win_size);
    c.fmin                = kv_u32 (gc, "clip.audio.ve.fmin",               c.fmin);
    c.fmax                = kv_u32 (gc, "clip.audio.ve.fmax",               c.fmax);
    c.mel_power           = kv_f32 (gc, "clip.audio.ve.mel_power",          c.mel_power);
    c.stft_magnitude_min  = kv_f32 (gc, "clip.audio.ve.stft_magnitude_min", c.stft_magnitude_min);
    c.preemphasis         = kv_f32 (gc, "clip.audio.ve.preemphasis",        c.preemphasis);
    c.normalized_mels     = kv_bool(gc, "clip.audio.ve.normalized_mels",    c.normalized_mels);
    c.mel_type            = kv_str (gc, "clip.audio.ve.mel_type",           c.mel_type);
    c.hidden_size         = kv_u32 (gc, "clip.audio.ve.hidden_size",        c.hidden_size);
    c.num_layers          = kv_u32 (gc, "clip.audio.ve.num_layers",         c.num_layers);
    c.speaker_embed_size  = kv_u32 (gc, "clip.audio.ve.speaker_embed_size", c.speaker_embed_size);
    c.partial_frames      = kv_u32 (gc, "clip.audio.ve.partial_frames",     c.partial_frames);
    c.final_relu          = kv_bool(gc, "clip.audio.ve.final_relu",         c.final_relu);
    c.cond_enc_n_channels = kv_u32 (gc, "clip.audio.cond_enc.n_channels",   c.cond_enc_n_channels);
    return c;
}

// S3Tok hparams (from clip.audio.s3tok.* if present, otherwise defaults from
// s3tokenizer.model_v2.ModelConfig).
static S3TokConfig load_s3tok_config(gguf_context * gc) {
    S3TokConfig c;
    c.n_mels       = kv_u32(gc, "clip.audio.s3tok.n_mels",       c.n_mels);
    c.n_fft        = kv_u32(gc, "clip.audio.s3tok.n_fft",        c.n_fft);
    c.hop          = kv_u32(gc, "clip.audio.s3tok.hop",          c.hop);
    c.sr           = kv_u32(gc, "clip.audio.s3tok.sample_rate",  c.sr);
    c.n_state      = kv_u32(gc, "clip.audio.s3tok.n_state",      c.n_state);
    c.n_head       = kv_u32(gc, "clip.audio.s3tok.n_head",       c.n_head);
    c.n_layer      = kv_u32(gc, "clip.audio.s3tok.n_layer",      c.n_layer);
    c.fsmn_kernel  = kv_u32(gc, "clip.audio.s3tok.fsmn_kernel",  c.fsmn_kernel);
    c.conv1_stride = kv_u32(gc, "clip.audio.s3tok.conv1_stride", c.conv1_stride);
    c.fsq_dim      = kv_u32(gc, "clip.audio.s3tok.fsq_dim",      c.fsq_dim);
    c.fsq_level    = kv_u32(gc, "clip.audio.s3tok.fsq_level",    c.fsq_level);
    return c;
}

// ============================================================================
// Section 2: Minimal PCM WAV loader
// ============================================================================

static bool load_wav_f32(const std::string & path, std::vector<float> & samples, int & sr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "load_wav: cannot open %s\n", path.c_str()); return false; }

    char hdr[12];
    f.read(hdr, 12);
    if (std::memcmp(hdr, "RIFF", 4) != 0 || std::memcmp(hdr + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "load_wav: not a RIFF/WAVE file\n"); return false;
    }

    uint16_t fmt_tag = 0, n_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0;
    std::vector<uint8_t> data_bytes;

    while (f.good()) {
        char chunk_id[4];
        uint32_t chunk_size = 0;
        f.read(chunk_id, 4);
        f.read(reinterpret_cast<char*>(&chunk_size), 4);
        if (!f.good()) break;

        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            std::vector<uint8_t> buf(chunk_size);
            f.read(reinterpret_cast<char*>(buf.data()), chunk_size);
            if (chunk_size < 16) { fprintf(stderr, "load_wav: fmt chunk too small\n"); return false; }
            std::memcpy(&fmt_tag,         buf.data() + 0,  2);
            std::memcpy(&n_channels,      buf.data() + 2,  2);
            std::memcpy(&sample_rate,     buf.data() + 4,  4);
            std::memcpy(&bits_per_sample, buf.data() + 14, 2);
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            data_bytes.resize(chunk_size);
            f.read(reinterpret_cast<char*>(data_bytes.data()), chunk_size);
        } else {
            f.seekg(chunk_size, std::ios::cur);
        }
    }

    if (data_bytes.empty()) { fprintf(stderr, "load_wav: no data chunk\n"); return false; }

    if (fmt_tag == 1 && bits_per_sample == 16) {
        size_t n = data_bytes.size() / 2;
        size_t total_samples = n / n_channels;
        samples.resize(total_samples);
        const int16_t * src = reinterpret_cast<const int16_t*>(data_bytes.data());
        for (size_t i = 0; i < total_samples; ++i) {
            int32_t acc = 0;
            for (uint16_t c = 0; c < n_channels; ++c) acc += src[i * n_channels + c];
            samples[i] = (float)acc / (float)(n_channels * 32768);
        }
    } else if (fmt_tag == 3 && bits_per_sample == 32) {
        size_t n = data_bytes.size() / 4;
        size_t total_samples = n / n_channels;
        samples.resize(total_samples);
        const float * src = reinterpret_cast<const float*>(data_bytes.data());
        for (size_t i = 0; i < total_samples; ++i) {
            float acc = 0.0f;
            for (uint16_t c = 0; c < n_channels; ++c) acc += src[i * n_channels + c];
            samples[i] = acc / (float)n_channels;
        }
    } else {
        fprintf(stderr, "load_wav: unsupported fmt_tag=%u, bits=%u\n", fmt_tag, bits_per_sample);
        return false;
    }

    sr = (int)sample_rate;
    return true;
}

// ============================================================================
// Section 3a: STFT primitives (used by both VE amp-mel and S3Tok log-mel)
// ============================================================================

static void reflect_pad(const std::vector<float> & src, std::vector<float> & dst, int pad) {
    int N = (int)src.size();
    dst.resize(N + 2 * pad);
    for (int i = 0; i < pad; ++i) dst[i]           = src[pad - i];
    for (int i = 0; i < N;   ++i) dst[i + pad]     = src[i];
    for (int i = 0; i < pad; ++i) dst[N + pad + i] = src[N - 2 - i];
}

enum class HannWindowType { Symmetric, Periodic };

static std::vector<float> hann_window(int N, HannWindowType type) {
    std::vector<float> w(N);
    // Symmetric: cos(2π n / (N-1)) — librosa / scipy default
    // Periodic:  cos(2π n / N)     — torch.hann_window default
    const float denom = (type == HannWindowType::Symmetric) ? (float)(N - 1) : (float)N;
    for (int i = 0; i < N; ++i) {
        w[i] = 0.5f * (1.0f - std::cos(2.0f * VE_PI * (float)i / denom));
    }
    return w;
}

// Precompute (twiddle_r[k*n_fft + n], twiddle_i[k*n_fft + n]) for a naive DFT.
static void precompute_twiddles(int n_fft, int n_freqs,
                                std::vector<float> & twiddle_r,
                                std::vector<float> & twiddle_i) {
    twiddle_r.assign((size_t)n_freqs * n_fft, 0.0f);
    twiddle_i.assign((size_t)n_freqs * n_fft, 0.0f);
    const float two_pi = 2.0f * VE_PI;
    for (int k = 0; k < n_freqs; ++k) {
        for (int n = 0; n < n_fft; ++n) {
            float phase = -two_pi * (float)k * (float)n / (float)n_fft;
            twiddle_r[(size_t)k * n_fft + n] = std::cos(phase);
            twiddle_i[(size_t)k * n_fft + n] = std::sin(phase);
        }
    }
}

// ============================================================================
// Section 3b: VE mel (40-bin amplitude-mel, symmetric Hann)
// ============================================================================
static std::vector<float> compute_ve_mel(
        const std::vector<float> & pcm,
        const float *              mel_filters,   // (n_mels, n_freqs) row-major
        const VoiceEncoderConfig & cfg,
        int &                      out_n_frames) {
    const int n_fft   = cfg.n_fft;
    const int hop     = cfg.hop_size;
    const int win_len = cfg.win_size;
    const int n_freqs = n_fft / 2 + 1;
    const int n_mels  = cfg.num_mels;
    const int pad     = n_fft / 2;

    // librosa.stft uses periodic hann (scipy.signal.get_window(..., fftbins=True)).
    auto window = hann_window(win_len, HannWindowType::Periodic);
    std::vector<float> padded; reflect_pad(pcm, padded, pad);
    const int n_frames = 1 + (int)pcm.size() / hop;

    std::vector<float> twiddle_r, twiddle_i;
    precompute_twiddles(n_fft, n_freqs, twiddle_r, twiddle_i);

    std::vector<float> mel_TM((size_t)n_frames * n_mels, 0.0f);
    std::vector<float> frame(n_fft);
    std::vector<float> mag_sq(n_freqs);

    for (int t = 0; t < n_frames; ++t) {
        const int frame_start = t * hop;
        const int win_off     = (n_fft - win_len) / 2;
        std::fill(frame.begin(), frame.end(), 0.0f);
        for (int i = 0; i < win_len; ++i) {
            frame[i + win_off] = padded[frame_start + i + win_off] * window[i];
        }
        for (int k = 0; k < n_freqs; ++k) {
            float re = 0.0f, im = 0.0f;
            const float * tr = &twiddle_r[(size_t)k * n_fft];
            const float * ti = &twiddle_i[(size_t)k * n_fft];
            for (int n = 0; n < n_fft; ++n) { re += frame[n] * tr[n]; im += frame[n] * ti[n]; }
            float m = re * re + im * im;
            if (cfg.mel_power != 2.0f) m = std::pow(std::sqrt(m), cfg.mel_power);
            mag_sq[k] = m;
        }
        float * mel_row = &mel_TM[(size_t)t * n_mels];
        for (int m = 0; m < n_mels; ++m) {
            const float * filt = &mel_filters[(size_t)m * n_freqs];
            float acc = 0.0f;
            for (int k = 0; k < n_freqs; ++k) acc += filt[k] * mag_sq[k];
            mel_row[m] = acc;
        }
    }
    out_n_frames = n_frames;
    return mel_TM;
}

// ============================================================================
// Section 3c: S3Tok mel (128-bin Whisper-style log-mel, periodic Hann)
// ============================================================================
// Matches s3tokenizer.S3Tokenizer.log_mel_spectrogram in the pip package.
// Returns row-major (n_frames, n_mels) with n_frames = ceil(padded_len / hop),
// where padded_len is a multiple of samples_per_tok (=640), yielding 4*n_tokens
// mel frames after dropping the trailing STFT frame.
static std::vector<float> compute_s3tok_mel(
        const std::vector<float> & pcm_padded,
        const float *              mel_filters,   // (n_mels=128, n_freqs=201)
        const S3TokConfig &        cfg,
        int &                      out_n_frames) {
    const int n_fft   = cfg.n_fft;
    const int hop     = cfg.hop;
    const int n_freqs = n_fft / 2 + 1;
    const int n_mels  = cfg.n_mels;
    const int pad     = n_fft / 2;

    auto window = hann_window(n_fft, HannWindowType::Periodic);
    std::vector<float> padded; reflect_pad(pcm_padded, padded, pad);

    // torch.stft with center=True gives 1 + L/hop frames; we drop the last.
    const int n_frames_full = 1 + (int)pcm_padded.size() / hop;
    const int n_frames      = n_frames_full - 1;  // matches stft[..., :-1] in PyTorch

    std::vector<float> twiddle_r, twiddle_i;
    precompute_twiddles(n_fft, n_freqs, twiddle_r, twiddle_i);

    // Compute (n_frames, n_mels) row-major, then Whisper-normalise.
    std::vector<float> mel_TM((size_t)n_frames * n_mels, 0.0f);
    std::vector<float> frame(n_fft);
    std::vector<float> mag_sq(n_freqs);

    for (int t = 0; t < n_frames; ++t) {
        const int frame_start = t * hop;
        for (int i = 0; i < n_fft; ++i) frame[i] = padded[frame_start + i] * window[i];

        for (int k = 0; k < n_freqs; ++k) {
            float re = 0.0f, im = 0.0f;
            const float * tr = &twiddle_r[(size_t)k * n_fft];
            const float * ti = &twiddle_i[(size_t)k * n_fft];
            for (int n = 0; n < n_fft; ++n) { re += frame[n] * tr[n]; im += frame[n] * ti[n]; }
            mag_sq[k] = re * re + im * im;
        }

        float * mel_row = &mel_TM[(size_t)t * n_mels];
        for (int m = 0; m < n_mels; ++m) {
            const float * filt = &mel_filters[(size_t)m * n_freqs];
            float acc = 0.0f;
            for (int k = 0; k < n_freqs; ++k) acc += filt[k] * mag_sq[k];
            mel_row[m] = acc;
        }
    }

    // Whisper normalisation: max(log10(clamp(x, 1e-10)), max_log10 - 8) then (x + 4) / 4.
    // Two-pass: first log10-clamp, find max, apply clip and offset.
    float max_v = -1e30f;
    for (auto & v : mel_TM) {
        v = std::log10(std::max(v, 1e-10f));
        if (v > max_v) max_v = v;
    }
    const float floor_v = max_v - 8.0f;
    for (auto & v : mel_TM) {
        if (v < floor_v) v = floor_v;
        v = (v + 4.0f) / 4.0f;
    }

    out_n_frames = n_frames;
    return mel_TM;
}

// ============================================================================
// Section 4: VE partial windowing
// ============================================================================
static std::vector<std::vector<float>> stride_partials(
        const std::vector<float> & mel_TM,
        int n_frames, int n_mels,
        int partial_frames, float rate,
        float min_coverage = 0.8f) {
    int frame_step = (int)std::round((16000.0f / rate) / (float)partial_frames);
    if (frame_step < 1)              frame_step = 1;
    if (frame_step > partial_frames) frame_step = partial_frames;

    const int win_size = partial_frames;
    const int step     = frame_step;
    const int over     = win_size - step;

    int base      = std::max(n_frames - win_size + step, 0);
    int n_wins    = base / step;
    int remainder = base - n_wins * step;
    if (n_wins == 0 || (float)(remainder + over) / (float)win_size >= min_coverage) n_wins += 1;

    const int target_n = win_size + step * (n_wins - 1);
    const int T_out    = std::max(target_n, n_frames);

    std::vector<float> padded_mel((size_t)T_out * n_mels, 0.0f);
    const int T_copy = std::min(n_frames, target_n);
    std::memcpy(padded_mel.data(), mel_TM.data(), (size_t)T_copy * n_mels * sizeof(float));

    std::vector<std::vector<float>> out(n_wins);
    for (int p = 0; p < n_wins; ++p) {
        int start = p * step;
        out[p].resize((size_t)partial_frames * n_mels);
        std::memcpy(out[p].data(),
                    padded_mel.data() + (size_t)start * n_mels,
                    (size_t)partial_frames * n_mels * sizeof(float));
    }
    return out;
}

// ============================================================================
// Section 5: VE LSTM cell (one timestep, one layer)
// ============================================================================
static std::pair<ggml_tensor *, ggml_tensor *> lstm_cell_step(
        ggml_context * ctx,
        ggml_tensor *  gates,   // (4H, B)
        ggml_tensor *  c_prev,  // (H,  B)
        int H, int B) {
    (void)B;
    const size_t elem = ggml_type_size(gates->type);
    // ggml_view_2d slices are strided (non-contiguous) — CUDA unary ops require
    // contiguous inputs, so materialise each gate slice before the activation.
    ggml_tensor * i = ggml_cont(ctx, ggml_view_2d(ctx, gates, H, B, gates->nb[1], (size_t)0 * H * elem));
    ggml_tensor * f = ggml_cont(ctx, ggml_view_2d(ctx, gates, H, B, gates->nb[1], (size_t)1 * H * elem));
    ggml_tensor * g = ggml_cont(ctx, ggml_view_2d(ctx, gates, H, B, gates->nb[1], (size_t)2 * H * elem));
    ggml_tensor * o = ggml_cont(ctx, ggml_view_2d(ctx, gates, H, B, gates->nb[1], (size_t)3 * H * elem));
    i = ggml_sigmoid(ctx, i);
    f = ggml_sigmoid(ctx, f);
    g = ggml_tanh   (ctx, g);
    o = ggml_sigmoid(ctx, o);
    ggml_tensor * c_new = ggml_add(ctx, ggml_mul(ctx, f, c_prev), ggml_mul(ctx, i, g));
    ggml_tensor * h_new = ggml_mul(ctx, o, ggml_tanh(ctx, c_new));
    return {h_new, c_new};
}

// ============================================================================
// Section 6: VE + cond_enc graph builder → 1024-dim cond_spkr tensor
// ============================================================================
// Materialise a fresh contiguous copy of `t` so the tap survives downstream ops
// that might reuse the original tensor's memory (especially critical for views
// and intermediate tensors that are used again later in the graph). We also
// stash the copy on the TapDumper so encode_reference can build_forward_expand
// each tap — mere ggml_set_output isn't enough since taps sit on side branches
// off the main compute path.
static void mark_tap(ggml_context * ctx, ggml_tensor * t, const char * name, TapDumper * taps) {
    if (!taps || !taps->enabled()) return;
    ggml_tensor * copy = ggml_cont(ctx, t);
    ggml_set_name(copy, name);
    ggml_set_output(copy);
    taps->pending.push_back({name, copy});
}

// Input x_mel_partials layout: (n_mels, B=n_partials, T=partial_frames).
static ggml_tensor * build_ve_graph(
        ggml_context *       ctx,
        const VoiceEncoder & ve,
        ggml_tensor *        x,
        TapDumper *          taps = nullptr) {
    const int T       = (int)x->ne[2];
    const int B       = (int)x->ne[1];
    const int n_mels  = (int)x->ne[0];
    const int H       = ve.ve_cfg.hidden_size;
    const int L       = ve.ve_cfg.num_layers;
    const int spk_out = ve.ve_cfg.speaker_embed_size;

    ggml_tensor * x_flat = ggml_reshape_2d(ctx, x, n_mels, B * T);
    ggml_tensor * ih_l0_all = ggml_add(ctx,
        ggml_mul_mat(ctx, ve.lstm[0].w_ih, x_flat),
        ve.lstm[0].b_ih);

    // Materialise a zero-filled (H, B) tensor for the initial LSTM state. The
    // ggml_view_2d slice into ih_l0_all is strided (nb1 = 4H*elem, not H*elem)
    // so we ggml_cont it before feeding to ggml_scale, which is a unary op and
    // requires contiguity on the CUDA backend.
    ggml_tensor * zero_HB = ggml_scale(ctx,
        ggml_cont(ctx, ggml_view_2d(ctx, ih_l0_all, H, B, ih_l0_all->nb[1], 0)),
        0.0f);

    std::vector<ggml_tensor *> h_prev(L, zero_HB);
    std::vector<ggml_tensor *> c_prev(L, zero_HB);

    for (int t = 0; t < T; ++t) {
        ggml_tensor * ih0 = ggml_view_2d(ctx, ih_l0_all,
            4 * H, B, ih_l0_all->nb[1],
            (size_t)t * B * ih_l0_all->nb[1]);
        ggml_tensor * hh0 = ggml_add(ctx,
            ggml_mul_mat(ctx, ve.lstm[0].w_hh, h_prev[0]),
            ve.lstm[0].b_hh);
        auto step0 = lstm_cell_step(ctx, ggml_add(ctx, ih0, hh0), c_prev[0], H, B);
        h_prev[0] = step0.first;
        c_prev[0] = step0.second;

        ggml_tensor * x_lower = h_prev[0];
        for (int l = 1; l < L; ++l) {
            ggml_tensor * ih_l = ggml_add(ctx, ggml_mul_mat(ctx, ve.lstm[l].w_ih, x_lower), ve.lstm[l].b_ih);
            ggml_tensor * hh_l = ggml_add(ctx, ggml_mul_mat(ctx, ve.lstm[l].w_hh, h_prev[l]), ve.lstm[l].b_hh);
            auto step_l = lstm_cell_step(ctx, ggml_add(ctx, ih_l, hh_l), c_prev[l], H, B);
            h_prev[l] = step_l.first;
            c_prev[l] = step_l.second;
            x_lower   = h_prev[l];
        }
    }

    ggml_tensor * h_final = h_prev[L - 1];
    mark_tap(ctx, h_final, "tap.ve_lstm_hidden_last", taps);

    ggml_tensor * emb = ggml_add(ctx, ggml_mul_mat(ctx, ve.proj_w, h_final), ve.proj_b);
    if (ve.ve_cfg.final_relu) emb = ggml_relu(ctx, emb);
    mark_tap(ctx, emb, "tap.ve_proj_relu", taps);

    // L2 normalise per partial.
    ggml_tensor * emb_norm = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, emb)));
    ggml_tensor * emb_normed = ggml_div(ctx, emb, emb_norm);
    mark_tap(ctx, emb_normed, "tap.ve_partial_embeds_norm", taps);

    // Mean across partials → (spk_embed,).
    ggml_tensor * emb_bt = ggml_cont(ctx, ggml_transpose(ctx, emb_normed));
    ggml_tensor * mean   = ggml_reshape_1d(ctx, ggml_mean(ctx, emb_bt), spk_out);
    mark_tap(ctx, mean, "tap.ve_mean_embed", taps);

    // L2 normalise the mean.
    ggml_tensor * ve_out = ggml_div(ctx, mean,
        ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, mean))));
    mark_tap(ctx, ve_out, "tap.ve_final_embed", taps);

    // cond_enc.spkr_enc Linear (spk_embed -> n_channels).
    ggml_tensor * cond = ggml_add(ctx,
        ggml_mul_mat(ctx, ve.cond_spkr_w, ve_out),
        ve.cond_spkr_b);
    ggml_set_name(cond, "cond_spkr");
    return cond;
}

// ============================================================================
// Section 7: S3Tok building blocks
// ============================================================================

// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta.
static ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x,
                                ggml_tensor * gamma, ggml_tensor * beta,
                                float eps = 1e-5f) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, gamma);
    if (beta) y = ggml_add(ctx, y, beta);
    return y;
}

// One S3Tok residual block: attn_ln → (Q,K,V linear + RoPE + FSMN + attn) → residual → mlp_ln → MLP → residual.
// Input `x` layout: (n_state, T), matches ggml matmul convention.
static ggml_tensor * s3tok_block(
        ggml_context *              ctx,
        ggml_tensor *               x,
        const VoiceEncoder::S3TokBlock & w,
        ggml_tensor *               positions,   // (T,) i32
        const S3TokConfig &         cfg,
        TapDumper *                 taps = nullptr,
        int                         blk_idx = -1) {
    // Only tap block 0's internals (avoid tap-name collisions across blocks).
    auto blk_tap = [&](ggml_tensor * t, const char * suffix) {
        if (blk_idx == 0 && taps && taps->enabled()) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "tap.s3_blk0_%s", suffix);
            mark_tap(ctx, t, nm, taps);
        }
    };
    const int T       = (int)x->ne[1];
    const int n_state = (int)x->ne[0];
    const int H       = cfg.n_head;
    const int D       = n_state / H;   // head_dim = 64

    // Pre-norm and QKV projection.
    ggml_tensor * xn = layer_norm(ctx, x, w.ln1_w, w.ln1_b);
    blk_tap(xn, "ln1_out");

    ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, w.attn_q_w, xn), w.attn_q_b);          // (n_state, T)
    ggml_tensor * k =                ggml_mul_mat(ctx, w.attn_k_w, xn);                       // (n_state, T)  no bias
    ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, w.attn_v_w, xn), w.attn_v_b);          // (n_state, T)
    blk_tap(q, "q_pre_rope");
    blk_tap(k, "k_pre_rope");
    blk_tap(v, "v_pre_rope");

    // FSMN residual on v (before RoPE): v_flat -> depthwise conv1d(kernel=31, groups=n_state) -> + v.
    // ggml_conv_1d_dw wants data as (T, n_channels) with ne0=T, ne1=n_channels.
    // v currently is (n_state, T); reshape to (T, n_state) via transpose+cont.
    ggml_tensor * v_TxC   = ggml_cont(ctx, ggml_transpose(ctx, v));                          // (T, n_state)
    ggml_tensor * fsm_conv = ggml_conv_1d_dw(ctx, w.fsmn_conv, v_TxC,
                                             /*s=*/1, /*p=*/cfg.fsmn_kernel / 2, /*d=*/1);   // (T, n_state, 1)
    fsm_conv = ggml_reshape_2d(ctx, fsm_conv, T, n_state);
    ggml_tensor * fsm_mem = ggml_add(ctx, fsm_conv, v_TxC);                                  // (T, n_state)
    blk_tap(fsm_mem, "fsm_mem_TxC");
    // Bring FSMN residual back to (n_state, T) layout to add to the attn output later.
    ggml_tensor * fsm_CxT = ggml_cont(ctx, ggml_transpose(ctx, fsm_mem));                    // (n_state, T)

    // Reshape Q/K/V for multi-head with RoPE.
    // (n_state, T) -> (D, H, T) with ne0=D=head_dim, ne1=H=n_head, ne2=T.
    q = ggml_reshape_3d(ctx, q, D, H, T);
    k = ggml_reshape_3d(ctx, k, D, H, T);
    v = ggml_reshape_3d(ctx, v, D, H, T);

    // RoPE (NEOX = half-rotation) on Q and K, matching s3tokenizer's apply_rotary_emb.
    q = ggml_rope_ext(ctx, q, positions, /*c=*/nullptr,
                      /*n_dims=*/D, /*mode=*/GGML_ROPE_TYPE_NEOX,
                      /*n_ctx_orig=*/0, /*freq_base=*/10000.0f, /*freq_scale=*/1.0f,
                      0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_ext(ctx, k, positions, /*c=*/nullptr,
                      D, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    blk_tap(q, "q_post_rope");
    blk_tap(k, "k_post_rope");

    // Rearrange for attention. Target shapes for ggml_mul_mat:
    //   q: (D, T, H) (per-head sequence)     -> permute (0,1,2)->(0,2,1)
    //   k: (D, T, H)                          -> permute (0,1,2)->(0,2,1)
    //   v: (T, D, H) — dim 0 will contract    -> permute (0,1,2)->(1,2,0)? Let me use the standard pattern:
    // Standard llama.cpp attention pattern (with matmuls on ne0 axis):
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));    // (D, T, H)
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));    // (D, T, H)
    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));    // (T, D, H) — dim 0 = T for contraction with scores

    // scores = k^T @ q → (T_k, T_q, H). ggml_mul_mat(a, b) contracts on ne0 of both.
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);          // (T, T, H)
    scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float)D));
    scores = ggml_soft_max(ctx, scores);                     // softmax over ne0 (keys)

    // out_h = v @ scores → (D, T_q, H). Both v (T, D, H) and scores (T_k, T_q, H) contract on ne0 = T_k = T_q.
    ggml_tensor * out_h = ggml_mul_mat(ctx, v, scores);      // (D, T, H)

    // (D, T, H) -> (D, H, T) -> (n_state, T)
    out_h = ggml_cont(ctx, ggml_permute(ctx, out_h, 0, 2, 1, 3));
    ggml_tensor * out_flat = ggml_reshape_2d(ctx, out_h, n_state, T);

    blk_tap(out_flat, "attn_pre_out");
    // Attention output projection + FSMN residual + skip.
    ggml_tensor * attn_out = ggml_add(ctx, ggml_mul_mat(ctx, w.attn_out_w, out_flat), w.attn_out_b);
    blk_tap(attn_out, "attn_out_no_fsmn");
    attn_out = ggml_add(ctx, attn_out, fsm_CxT);
    blk_tap(attn_out, "attn_out_with_fsmn");
    x = ggml_add(ctx, x, attn_out);
    blk_tap(x, "x_after_attn_residual");

    // MLP block: LayerNorm -> Linear(up) -> GELU -> Linear(down) -> residual.
    ggml_tensor * xn2 = layer_norm(ctx, x, w.ln2_w, w.ln2_b);
    blk_tap(xn2, "ln2_out");
    ggml_tensor * mlp = ggml_add(ctx, ggml_mul_mat(ctx, w.ffn_up_w, xn2), w.ffn_up_b);
    mlp = ggml_gelu(ctx, mlp);
    mlp = ggml_add(ctx, ggml_mul_mat(ctx, w.ffn_down_w, mlp), w.ffn_down_b);
    blk_tap(mlp, "mlp_out");
    x = ggml_add(ctx, x, mlp);
    return x;
}

// Build the S3Tok forward pass. Input tensor layout is (n_mels, T_mel) —
// ne0=n_mels, ne1=T_mel — which matches what compute_s3tok_mel writes into
// a plain row-major (T_mel, n_mels) host buffer under ggml's memory
// convention (ne0 is the fast-varying dimension).
//
// Returns FSQ pre-round logits h_scaled = tanh(project_down(x)) * 0.999 of
// shape (fsq_dim, T'), which the caller rounds + base-3 combines on CPU to
// produce token IDs.
static ggml_tensor * build_s3tok_graph(
        ggml_context *       ctx,
        const VoiceEncoder & ve,
        ggml_tensor *        mel_MxT,
        TapDumper *          taps = nullptr) {
    const S3TokConfig & cfg = ve.s3_cfg;
    const int n_mels  = cfg.n_mels;
    const int n_state = cfg.n_state;
    (void)n_mels;

    // ggml_conv_1d expects (T, in_channels) in ne0-major layout. Our host mel
    // buffer is (T, n_mels) row-major, which ggml interprets as (n_mels, T).
    // Transpose here to get to the conv-friendly layout.
    ggml_tensor * mel_TxM = ggml_cont(ctx, ggml_transpose(ctx, mel_MxT));

    // conv1 -> (T/2, n_state). Transpose immediately to (n_state, T/2) so the
    // 1-D bias broadcasts naturally on ne1 (the T axis), matching llama.cpp's
    // Linear-with-bias convention.
    ggml_tensor * h = ggml_conv_1d(ctx, ve.s3_conv1_w, mel_TxM, cfg.conv1_stride, 1, 1);
    h = ggml_cont(ctx, ggml_transpose(ctx, h));                 // (n_state, T/2)
    h = ggml_add(ctx, h, ve.s3_conv1_b);
    h = ggml_gelu(ctx, h);
    mark_tap(ctx, h, "tap.s3_conv1_out", taps);

    // For conv2 we need (T, n_state) again.
    h = ggml_cont(ctx, ggml_transpose(ctx, h));                 // (T/2, n_state)
    h = ggml_conv_1d(ctx, ve.s3_conv2_w, h, 2, 1, 1);           // (T/4, n_state)
    h = ggml_cont(ctx, ggml_transpose(ctx, h));                 // (n_state, T/4)
    h = ggml_add(ctx, h, ve.s3_conv2_b);
    h = ggml_gelu(ctx, h);
    mark_tap(ctx, h, "tap.s3_conv2_out", taps);

    // Attention convention: (n_state, T') already, ne0=n_state, ne1=T'.
    const int T_prime = (int)h->ne[1];

    // Positions tensor for RoPE (shared across all layers).
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_prime);
    ggml_set_name(positions, "s3_positions");
    ggml_set_input(positions);

    for (size_t i = 0; i < ve.s3_blocks.size(); ++i) {
        h = s3tok_block(ctx, h, ve.s3_blocks[i], positions, cfg, taps, (int)i);
        char nm[32];
        std::snprintf(nm, sizeof(nm), "tap.s3_blk%zu_out", i);
        mark_tap(ctx, h, nm, taps);
    }

    // FSQ head: project (n_state, T') -> (fsq_dim, T'), then tanh * 0.999.
    ggml_tensor * fsq_pre = ggml_add(ctx, ggml_mul_mat(ctx, ve.s3_fsq_w, h), ve.s3_fsq_b);   // (fsq_dim, T')
    mark_tap(ctx, fsq_pre, "tap.s3_fsq_pre", taps);
    ggml_tensor * fsq_h   = ggml_scale(ctx, ggml_tanh(ctx, fsq_pre), cfg.fsq_scale);
    ggml_set_name(fsq_h, "s3_fsq_h");
    return fsq_h;
}

// ============================================================================
// Section 8: Backend init / load_model / free
// ============================================================================

VoiceEncoder::VoiceEncoder(ggml_backend_t shared_backend) {
    if (shared_backend) { backend = shared_backend; owns_backend = false; }
    else                { owns_backend = true; }
}

VoiceEncoder::~VoiceEncoder() { free_model(); }

bool VoiceEncoder::init_backend() {
    if (backend) return true;
    const char * force_cpu = std::getenv("VE_FORCE_CPU");
    const bool cpu_only = force_cpu && *force_cpu && force_cpu[0] != '0';
#ifdef GGML_USE_CUDA
    if (!cpu_only) {
        backend = ggml_backend_cuda_init(0);
        if (backend) { fprintf(stderr, "VoiceEncoder: CUDA backend ok\n"); return true; }
    }
#endif
#ifdef GGML_USE_VULKAN
    if (!cpu_only) {
        backend = ggml_backend_vk_init(0);
        if (backend) { fprintf(stderr, "VoiceEncoder: Vulkan backend ok\n"); return true; }
    }
#endif
    backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "VoiceEncoder: failed to init CPU backend\n"); return false; }
    fprintf(stderr, "VoiceEncoder: CPU backend ok%s\n", cpu_only ? " (forced by VE_FORCE_CPU)" : "");
    return true;
}

static ggml_tensor * must_get(ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) fprintf(stderr, "VoiceEncoder: missing tensor '%s' in GGUF\n", name);
    return t;
}

bool VoiceEncoder::load_model(const std::string & gguf_path) {
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp_ctx };
    gguf_context * gc = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!gc) { fprintf(stderr, "VoiceEncoder: gguf_init_from_file failed for %s\n", gguf_path.c_str()); return false; }

    ve_cfg = load_ve_config(gc);
    s3_cfg = load_s3tok_config(gc);

    const int n_tensors = (int)gguf_get_n_tensors(gc);
    ggml_init_params params = {
        /*.mem_size=*/ ggml_tensor_overhead() * (n_tensors + 8),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc=*/ true,
    };
    if (!model_ctx) model_ctx = ggml_init(params);

    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gc, i);
        ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
        ggml_tensor * dst = ggml_dup_tensor(model_ctx, src);
        ggml_set_name(dst, name);
    }

    buffer = ggml_backend_alloc_ctx_tensors(model_ctx, backend);
    if (!buffer) { fprintf(stderr, "VoiceEncoder: failed to alloc ctx tensors\n"); gguf_free(gc); return false; }

    for (ggml_tensor * cur = ggml_get_first_tensor(model_ctx); cur; cur = ggml_get_next_tensor(model_ctx, cur)) {
        ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        if (src) ggml_backend_tensor_set(cur, ggml_get_data(src), 0, ggml_nbytes(src));
    }

    // VE tensors
    lstm.resize(ve_cfg.num_layers);
    for (int l = 0; l < ve_cfg.num_layers; ++l) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "a.lstm.ih.%d.weight", l); lstm[l].w_ih = must_get(model_ctx, nm);
        std::snprintf(nm, sizeof(nm), "a.lstm.ih.%d.bias",   l); lstm[l].b_ih = must_get(model_ctx, nm);
        std::snprintf(nm, sizeof(nm), "a.lstm.hh.%d.weight", l); lstm[l].w_hh = must_get(model_ctx, nm);
        std::snprintf(nm, sizeof(nm), "a.lstm.hh.%d.bias",   l); lstm[l].b_hh = must_get(model_ctx, nm);
    }
    proj_w      = must_get(model_ctx, "a.lstm.proj.weight");
    proj_b      = must_get(model_ctx, "a.lstm.proj.bias");
    ve_mels     = must_get(model_ctx, "a.ve.mel_filters");
    cond_spkr_w = must_get(model_ctx, "a.cond_enc.spkr_enc.weight");
    cond_spkr_b = must_get(model_ctx, "a.cond_enc.spkr_enc.bias");

    // S3Tok tensors
    s3_conv1_w = must_get(model_ctx, "a.conv1d.0.weight");
    s3_conv1_b = must_get(model_ctx, "a.conv1d.0.bias");
    s3_conv2_w = must_get(model_ctx, "a.conv1d.1.weight");
    s3_conv2_b = must_get(model_ctx, "a.conv1d.1.bias");
    s3_mels    = must_get(model_ctx, "a.s3tok.mel_filters");
    s3_fsq_w   = must_get(model_ctx, "a.quant.fsq.proj.weight");
    s3_fsq_b   = must_get(model_ctx, "a.quant.fsq.proj.bias");

    s3_blocks.resize(s3_cfg.n_layer);
    for (int l = 0; l < s3_cfg.n_layer; ++l) {
        char nm[80];
        auto get = [&](const char * suffix) -> ggml_tensor * {
            std::snprintf(nm, sizeof(nm), "a.blk.%d.%s", l, suffix);
            return must_get(model_ctx, nm);
        };
        s3_blocks[l].ln1_w      = get("ln1.weight");
        s3_blocks[l].ln1_b      = get("ln1.bias");
        s3_blocks[l].attn_q_w   = get("attn_q.weight");
        s3_blocks[l].attn_q_b   = get("attn_q.bias");
        s3_blocks[l].attn_k_w   = get("attn_k.weight");
        s3_blocks[l].attn_v_w   = get("attn_v.weight");
        s3_blocks[l].attn_v_b   = get("attn_v.bias");
        s3_blocks[l].attn_out_w = get("attn_out.weight");
        s3_blocks[l].attn_out_b = get("attn_out.bias");
        s3_blocks[l].fsmn_conv  = get("fsmn_conv.weight");
        s3_blocks[l].ln2_w      = get("ln2.weight");
        s3_blocks[l].ln2_b      = get("ln2.bias");
        s3_blocks[l].ffn_up_w   = get("ffn_up.weight");
        s3_blocks[l].ffn_up_b   = get("ffn_up.bias");
        s3_blocks[l].ffn_down_w = get("ffn_down.weight");
        s3_blocks[l].ffn_down_b = get("ffn_down.bias");
    }

    gguf_free(gc);
    ggml_free(tmp_ctx);

    // Sanity
    for (int l = 0; l < ve_cfg.num_layers; ++l) {
        if (!lstm[l].w_ih || !lstm[l].b_ih || !lstm[l].w_hh || !lstm[l].b_hh) return false;
    }
    if (!proj_w || !proj_b || !ve_mels || !cond_spkr_w || !cond_spkr_b) return false;
    if (!s3_conv1_w || !s3_conv1_b || !s3_conv2_w || !s3_conv2_b || !s3_mels || !s3_fsq_w || !s3_fsq_b) return false;
    for (const auto & b : s3_blocks) {
        if (!b.ln1_w || !b.ln1_b || !b.attn_q_w || !b.attn_q_b || !b.attn_k_w ||
            !b.attn_v_w || !b.attn_v_b || !b.attn_out_w || !b.attn_out_b ||
            !b.fsmn_conv || !b.ln2_w || !b.ln2_b ||
            !b.ffn_up_w || !b.ffn_up_b || !b.ffn_down_w || !b.ffn_down_b) return false;
    }

    fprintf(stderr, "VoiceEncoder: loaded %d tensors from %s\n", n_tensors, gguf_path.c_str());
    fprintf(stderr, "VoiceEncoder: VE T=%d H=%d L=%d spk=%d n_channels=%d n_mels=%d\n",
            ve_cfg.partial_frames, ve_cfg.hidden_size, ve_cfg.num_layers,
            ve_cfg.speaker_embed_size, ve_cfg.cond_enc_n_channels, ve_cfg.num_mels);
    fprintf(stderr, "VoiceEncoder: S3Tok n_state=%d n_head=%d n_layer=%d fsmn_k=%d\n",
            s3_cfg.n_state, s3_cfg.n_head, s3_cfg.n_layer, s3_cfg.fsmn_kernel);
    return true;
}

void VoiceEncoder::free_model() {
    if (buffer)    { ggml_backend_buffer_free(buffer); buffer = nullptr; }
    if (model_ctx) { ggml_free(model_ctx); model_ctx = nullptr; }
    if (backend && owns_backend) { ggml_backend_free(backend); backend = nullptr; }
}

// ============================================================================
// Section 9: encode_reference orchestrator
// ============================================================================

bool VoiceEncoder::encode_reference(const std::string & wav_path,
                                    std::map<std::string, staged_io> & outputs) {
    if (!backend || !model_ctx) {
        fprintf(stderr, "VoiceEncoder::encode_reference: model not loaded\n");
        return false;
    }
    outputs.clear();

    TapDumper taps;
    taps.open_from_env();

    // ---- 1) Load WAV ----
    std::vector<float> pcm;
    int wav_sr = 0;
    if (!load_wav_f32(wav_path, pcm, wav_sr)) return false;
    if (wav_sr != s3_cfg.sr) {
        fprintf(stderr, "VoiceEncoder: WAV sample rate %d != expected %d; preprocess with `ffmpeg -i in.wav -ar 16000 -ac 1 ref.wav`.\n",
                wav_sr, s3_cfg.sr);
        return false;
    }
    fprintf(stderr, "VoiceEncoder: WAV %zu samples @ %d Hz (%.2f s)\n",
            pcm.size(), wav_sr, (double)pcm.size() / wav_sr);
    if (taps.enabled()) {
        taps.tap_f32("tap.pcm_16k", pcm.data(), { (int64_t)pcm.size() });
    }

    // ---- 2) VE mel (unchanged) ----
    std::vector<float> ve_mel_filt_host((size_t)ve_cfg.num_mels * (ve_cfg.n_fft / 2 + 1));
    ggml_backend_tensor_get(ve_mels, ve_mel_filt_host.data(), 0, ggml_nbytes(ve_mels));

    int ve_n_frames = 0;
    auto ve_mel_TM = compute_ve_mel(pcm, ve_mel_filt_host.data(), ve_cfg, ve_n_frames);
    if (taps.enabled()) {
        // Buffer layout is (t, m) row-major with m innermost, i.e., ne0=n_mels,
        // ne1=T in ggml semantics. Record ggml-style shape so the compare
        // script can transpose to match PyTorch's (M, T).
        taps.tap_f32("tap.ve_mel_TM", ve_mel_TM.data(),
                     { (int64_t)ve_cfg.num_mels, (int64_t)ve_n_frames });
    }
    auto partials = stride_partials(ve_mel_TM, ve_n_frames, ve_cfg.num_mels,
                                    ve_cfg.partial_frames, /*rate=*/1.3f);
    const int VE_B = (int)partials.size();
    const int VE_T = ve_cfg.partial_frames;
    const int VE_M = ve_cfg.num_mels;
    if (VE_B == 0) { fprintf(stderr, "VoiceEncoder: audio too short for VE partials\n"); return false; }
    fprintf(stderr, "VoiceEncoder: VE mel %d frames, %d partials\n", ve_n_frames, VE_B);

    std::vector<float> ve_input((size_t)VE_M * VE_B * VE_T);
    for (int b = 0; b < VE_B; ++b) {
        const float * src = partials[b].data();
        for (int t = 0; t < VE_T; ++t)
            for (int m = 0; m < VE_M; ++m)
                ve_input[(size_t)m + (size_t)b * VE_M + (size_t)t * VE_M * VE_B] = src[t * VE_M + m];
    }

    // ---- 3) S3Tok mel: pad WAV to a multiple of samples_per_tok and log-mel ----
    const int spt         = s3_cfg.samples_per_tok;
    const int n_tokens    = (int)std::ceil((double)pcm.size() / (double)spt);
    const int padded_len  = n_tokens * spt;
    std::vector<float> pcm_padded(padded_len, 0.0f);
    std::memcpy(pcm_padded.data(), pcm.data(), pcm.size() * sizeof(float));

    std::vector<float> s3_mel_filt_host((size_t)s3_cfg.n_mels * (s3_cfg.n_fft / 2 + 1));
    ggml_backend_tensor_get(s3_mels, s3_mel_filt_host.data(), 0, ggml_nbytes(s3_mels));

    int s3_n_frames = 0;
    auto s3_mel_TM = compute_s3tok_mel(pcm_padded, s3_mel_filt_host.data(), s3_cfg, s3_n_frames);
    fprintf(stderr, "VoiceEncoder: S3Tok mel %d frames (%d tokens)\n", s3_n_frames, n_tokens);
    if (taps.enabled()) {
        taps.tap_f32("tap.s3_pcm_padded", pcm_padded.data(), { (int64_t)pcm_padded.size() });
        // Same layout as VE mel: m innermost -> ne0=n_mels, ne1=T.
        taps.tap_f32("tap.s3_mel_TM", s3_mel_TM.data(),
                     { (int64_t)s3_cfg.n_mels, (int64_t)s3_n_frames });
    }
    if (s3_n_frames < 4 * n_tokens) {
        fprintf(stderr, "VoiceEncoder: S3Tok mel frames %d less than 4*n_tokens=%d\n",
                s3_n_frames, 4 * n_tokens);
    }
    // Trim (or pad) to exactly 4*n_tokens frames so conv output is exactly n_tokens.
    const int target_frames = 4 * n_tokens;
    s3_mel_TM.resize((size_t)target_frames * s3_cfg.n_mels, 0.0f);
    s3_n_frames = target_frames;

    // ---- 4) Build graph ----
    // Rough node budget: LSTM unroll ≈ 20 * L * T + S3Tok ≈ 6 * ~40 + heads ≈ 300. Give plenty of headroom.
    const size_t graph_size = (size_t)ve_cfg.num_layers * VE_T * 25 + 2048;
    ggml_init_params gparams = {
        /*.mem_size=*/ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc=*/ true,
    };
    ggml_context * gctx = ggml_init(gparams);
    if (!gctx) { fprintf(stderr, "VoiceEncoder: ggml_init(graph ctx) failed\n"); return false; }

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, graph_size, /*grads=*/false);

    // VE input placeholder: (n_mels, B, T)
    ggml_tensor * ve_in = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, VE_M, VE_B, VE_T);
    ggml_set_name(ve_in, "ve_in"); ggml_set_input(ve_in);

    // S3Tok input placeholder. Buffer is (T_mel, n_mels) row-major, which
    // ggml sees as ne0=n_mels, ne1=T_mel (ne0 is the fast-varying axis).
    // build_s3tok_graph transposes internally before conv1d.
    ggml_tensor * s3_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, s3_cfg.n_mels, s3_n_frames);
    ggml_set_name(s3_in, "s3_in"); ggml_set_input(s3_in);

    ggml_tensor * cond_spkr_t = build_ve_graph   (gctx, *this, ve_in, &taps);
    ggml_tensor * fsq_h_t     = build_s3tok_graph(gctx, *this, s3_in, &taps);
    ggml_set_output(cond_spkr_t);
    ggml_set_output(fsq_h_t);
    ggml_build_forward_expand(gf, cond_spkr_t);
    ggml_build_forward_expand(gf, fsq_h_t);
    for (const auto & p : taps.pending) {
        ggml_build_forward_expand(gf, p.tensor);
    }

    // ---- 5) Allocate + populate + compute ----
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "VoiceEncoder: gallocr_alloc_graph failed\n");
        ggml_gallocr_free(alloc); ggml_free(gctx); return false;
    }

    ggml_backend_tensor_set(ve_in, ve_input.data(), 0, ggml_nbytes(ve_in));
    ggml_backend_tensor_set(s3_in, s3_mel_TM.data(), 0, ggml_nbytes(s3_in));

    // Positions for RoPE — need to fill 0..T'-1 where T' = s3_n_frames / (conv1_stride * 2).
    // conv1_stride == 2, conv2 stride 2 (fixed), so T' = s3_n_frames / 4.
    ggml_tensor * positions = ggml_get_tensor(gctx, "s3_positions");
    if (!positions) {
        fprintf(stderr, "VoiceEncoder: s3_positions tensor missing after graph build\n");
        ggml_gallocr_free(alloc); ggml_free(gctx); return false;
    }
    const int T_prime = (int)positions->ne[0];
    std::vector<int32_t> pos_data(T_prime);
    for (int i = 0; i < T_prime; ++i) pos_data[i] = i;
    ggml_backend_tensor_set(positions, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "VoiceEncoder: graph_compute failed\n");
        ggml_gallocr_free(alloc); ggml_free(gctx); return false;
    }

    // ---- Optional: capture named intermediate taps for offline diff vs PyTorch ----
    if (taps.enabled()) {
        for (const auto & p : taps.pending) {
            taps.tap_tensor(p.name, p.tensor);
        }
        taps.tap_tensor("tap.ve_cond_spkr",    cond_spkr_t);
        taps.tap_tensor("tap.s3_fsq_h_scaled", fsq_h_t);
    }

    // ---- 6) Read back cond_spkr ----
    staged_io cond_out;
    cond_out.type = GGML_TYPE_F32;
    cond_out.set_shape(ve_cfg.cond_enc_n_channels);
    cond_out.allocate();
    ggml_backend_tensor_get(cond_spkr_t, cond_out.data.data(), 0, cond_out.data.size());
    outputs["cond_spkr"] = std::move(cond_out);

    // ---- 7) Read back FSQ h_scaled (fsq_dim, T'), round + base-3 combine ----
    const int F     = s3_cfg.fsq_dim;
    const int Tprim = (int)fsq_h_t->ne[1];
    std::vector<float> fsq_host((size_t)F * Tprim);
    ggml_backend_tensor_get(fsq_h_t, fsq_host.data(), 0, fsq_host.size() * sizeof(float));

    // Build token IDs. fsq_h layout: ne0 = fsq_dim = F, ne1 = T', so token t occupies fsq_host[t*F .. t*F+F).
    staged_io tok_out;
    tok_out.type = GGML_TYPE_I32;
    tok_out.set_shape(Tprim);
    tok_out.allocate();
    int32_t * tok_ptr = tok_out.as<int32_t>();
    int32_t base_power[16];
    base_power[0] = 1;
    for (int i = 1; i < F && i < 16; ++i) base_power[i] = base_power[i - 1] * s3_cfg.fsq_level;
    for (int t = 0; t < Tprim; ++t) {
        int32_t id = 0;
        for (int c = 0; c < F; ++c) {
            float v = fsq_host[(size_t)t * F + c];
            // round + 1  ->  {0, 1, 2}
            int level = (int)std::lround(v) + 1;
            if (level < 0) level = 0;
            if (level >= s3_cfg.fsq_level) level = s3_cfg.fsq_level - 1;
            id += level * base_power[c];
        }
        tok_ptr[t] = id;
    }
    outputs["prompt_tokens"] = std::move(tok_out);

    if (taps.enabled()) {
        taps.tap_i32("tap.s3_tokens", tok_ptr, { (int64_t)Tprim });
        taps.close();
    }

    ggml_gallocr_free(alloc);
    ggml_free(gctx);

    double sum2 = 0.0;
    const float * cptr = outputs["cond_spkr"].as<float>();
    for (int i = 0; i < ve_cfg.cond_enc_n_channels; ++i) sum2 += (double)cptr[i] * cptr[i];
    fprintf(stderr, "VoiceEncoder: cond_spkr[%d] ||.||_2 = %.4f, first: %.4f %.4f %.4f %.4f\n",
            ve_cfg.cond_enc_n_channels, std::sqrt(sum2),
            cptr[0], cptr[1], cptr[2], cptr[3]);
    fprintf(stderr, "VoiceEncoder: prompt_tokens[%d] first10:", Tprim);
    for (int i = 0; i < std::min(10, Tprim); ++i) fprintf(stderr, " %d", tok_ptr[i]);
    fprintf(stderr, "\n");
    return true;
}
