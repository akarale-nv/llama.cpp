#pragma once

// Typed, named tensor blob for staging encoder / vocoder I/O.
//
// This struct is intentionally kept simple and public so it can carry
// heterogeneous outputs of an encoder pass — e.g. a Chatterbox reference-audio
// encode produces both a float32 (1024,) speaker conditioning row AND an int32
// (T',) vector of prompt speech tokens. `std::map<std::string, staged_io>`
// gives us a name-keyed multi-output surface without introducing a whole
// mtmd-side API extension in this iteration.
//
// Design note: shape follows ggml's fixed-arity convention (ne[0..3]). Trailing
// unused dims should be 1 (never 0), so `n_elements()` returns the correct
// product even for lower-rank tensors.
//
// This header will be lifted into mtmd.h with the same shape/semantics when
// we do the mtmd extension proper. Keeping it local for now keeps the diff
// contained to this example.

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

struct staged_io {
    ggml_type type = GGML_TYPE_F32;
    int32_t   shape[GGML_MAX_DIMS] = {1, 1, 1, 1};
    std::vector<uint8_t> data;

    template <typename T>
    const T * as() const {
        return reinterpret_cast<const T *>(data.data());
    }

    template <typename T>
    T * as() {
        return reinterpret_cast<T *>(data.data());
    }

    size_t n_elements() const {
        size_t p = 1;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            p *= (size_t)(shape[i] > 0 ? shape[i] : 1);
        }
        return p;
    }

    void set_shape(int32_t d0, int32_t d1 = 1, int32_t d2 = 1, int32_t d3 = 1) {
        shape[0] = d0; shape[1] = d1; shape[2] = d2; shape[3] = d3;
    }

    // Allocate `data` sized to hold `n_elements() * sizeof(dtype)` bytes.
    void allocate() {
        size_t bytes = n_elements() * ggml_type_size(type);
        data.assign(bytes, 0);
    }
};
