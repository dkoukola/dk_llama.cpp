# Build identity format. Increment the public ABI revision whenever a change to
# installed public declarations or data layouts is incompatible with a bridge
# built against an earlier revision.
set(LLAMA_PUBLIC_ABI_REVISION 1)

# These facilities are compiled unconditionally into libllama in this tree.
set(LLAMA_SPECULATIVE_CHECKPOINT_AVAILABLE ON)
set(LLAMA_DFLASH_AVAILABLE ON)

set(_llama_build_id_features "manifest_format=1\n")
string(APPEND _llama_build_id_features
    "llama.public_abi_revision=${LLAMA_PUBLIC_ABI_REVISION}\n")

# Keep this manifest to compatibility-relevant values. Optimization, kernel
# tuning, warning, test, and example options intentionally do not change the
# identity because they do not alter the libllama/bridge ABI contract.
set(_llama_build_id_boolean_features
    BUILD_SHARED_LIBS
    GGML_ACCELERATE
    GGML_CANN
    GGML_CPU_HBM
    GGML_CUDA
    GGML_CURL
    GGML_HIPBLAS
    GGML_METAL
    GGML_MUSA
    GGML_OPENMP
    GGML_RPC
    GGML_STATIC
    GGML_SYCL
    GGML_VULKAN
    LLAMA_DFLASH_AVAILABLE
    LLAMA_SPECULATIVE_CHECKPOINT_AVAILABLE)

list(SORT _llama_build_id_boolean_features)
foreach(_feature IN LISTS _llama_build_id_boolean_features)
    if(DEFINED ${_feature} AND ${_feature})
        set(_value 1)
    else()
        set(_value 0)
    endif()
    string(APPEND _llama_build_id_features "cmake.${_feature}=${_value}\n")
endforeach()

if(GGML_MAX_CONTEXTS)
    set(_llama_ggml_max_contexts "${GGML_MAX_CONTEXTS}")
else()
    set(_llama_ggml_max_contexts 64)
endif()

if(GGML_MAX_SRC)
    set(_llama_ggml_max_src "${GGML_MAX_SRC}")
else()
    set(_llama_ggml_max_src 16)
endif()

string(APPEND _llama_build_id_features
    "cmake.GGML_MAX_CONTEXTS=${_llama_ggml_max_contexts}\n"
    "cmake.GGML_MAX_SRC=${_llama_ggml_max_src}\n")

set(_llama_build_id_scalar_features
    CMAKE_BUILD_TYPE
    CMAKE_C_COMPILER_ABI
    CMAKE_C_EXTENSIONS
    CMAKE_C_COMPILER_ID
    CMAKE_C_STANDARD
    CMAKE_C_COMPILER_TARGET
    CMAKE_C_COMPILER_VERSION
    CMAKE_CXX_COMPILER_ABI
    CMAKE_CXX_EXTENSIONS
    CMAKE_CXX_COMPILER_ID
    CMAKE_CXX_STANDARD
    CMAKE_CXX_COMPILER_TARGET
    CMAKE_CXX_COMPILER_VERSION
    CMAKE_MSVC_RUNTIME_LIBRARY
    CMAKE_OSX_ARCHITECTURES
    CMAKE_SIZEOF_VOID_P
    CMAKE_SYSTEM_NAME
    CMAKE_SYSTEM_PROCESSOR)

list(SORT _llama_build_id_scalar_features)
foreach(_feature IN LISTS _llama_build_id_scalar_features)
    if(DEFINED ${_feature})
        set(_value "${${_feature}}")
    else()
        set(_value "<unset>")
    endif()
    string(REPLACE "\\" "\\\\" _value "${_value}")
    string(REPLACE "\n" "\\n" _value "${_value}")
    string(APPEND _llama_build_id_features "cmake.${_feature}=${_value}\n")
endforeach()

set(_llama_build_id_feature_file
    "${CMAKE_CURRENT_BINARY_DIR}/llama-build-id-features.txt")
file(WRITE "${_llama_build_id_feature_file}" "${_llama_build_id_features}")

set(LLAMA_BUILD_ID_SOURCE
    "${CMAKE_CURRENT_BINARY_DIR}/llama-build-id.cpp")
set(LLAMA_BUILD_ID_HEADER
    "${CMAKE_CURRENT_BINARY_DIR}/llama-build-id.h")
set(LLAMA_BUILD_ID_INCLUDE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}")

add_custom_target(llama-build-id
    COMMAND "${CMAKE_COMMAND}"
        "-DLLAMA_SOURCE_DIR:PATH=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DLLAMA_BINARY_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}"
        "-DLLAMA_BUILD_ID_FEATURE_FILE:FILEPATH=${_llama_build_id_feature_file}"
        "-DLLAMA_BUILD_ID_HEADER:FILEPATH=${LLAMA_BUILD_ID_HEADER}"
        "-DLLAMA_BUILD_ID_HEADER_TEMPLATE:FILEPATH=${CMAKE_CURRENT_LIST_DIR}/llama-build-id.h.in"
        "-DLLAMA_BUILD_ID_OUTPUT:FILEPATH=${LLAMA_BUILD_ID_SOURCE}"
        "-DLLAMA_BUILD_ID_TEMPLATE:FILEPATH=${CMAKE_CURRENT_LIST_DIR}/llama-build-id.cpp.in"
        "-DLLAMA_GIT_EXECUTABLE:FILEPATH=${GIT_EXECUTABLE}"
        -P "${CMAKE_CURRENT_LIST_DIR}/generate-llama-build-id.cmake"
    BYPRODUCTS
        "${LLAMA_BUILD_ID_HEADER}"
        "${LLAMA_BUILD_ID_SOURCE}"
    COMMENT "Generating deterministic libllama build identity"
    VERBATIM)

unset(_feature)
unset(_llama_build_id_boolean_features)
unset(_llama_build_id_feature_file)
unset(_llama_build_id_features)
unset(_llama_build_id_scalar_features)
unset(_llama_ggml_max_contexts)
unset(_llama_ggml_max_src)
unset(_value)
