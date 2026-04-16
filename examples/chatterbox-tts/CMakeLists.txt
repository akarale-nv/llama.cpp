# T3 Speech Token Generation Example
# Self-contained: uses only llama.cpp public headers (no chatterbox.cpp dependency)

set(TARGET llama-chatterbox)

add_executable(${TARGET}
    t3-example.cpp
    s3gen.cpp
    decoder.cpp
    encoder.cpp
    s3gen_ops.cpp
    s3Mel2Wav.cpp
    s3Token2Mel.cpp
)

target_link_libraries(${TARGET} PRIVATE llama common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

install(TARGETS ${TARGET} RUNTIME)
