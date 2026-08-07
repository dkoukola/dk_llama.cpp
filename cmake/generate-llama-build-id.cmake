cmake_minimum_required(VERSION 3.14)

set(_required_variables
    LLAMA_BINARY_DIR
    LLAMA_BUILD_ID_FEATURE_FILE
    LLAMA_BUILD_ID_HEADER
    LLAMA_BUILD_ID_HEADER_TEMPLATE
    LLAMA_BUILD_ID_OUTPUT
    LLAMA_BUILD_ID_TEMPLATE
    LLAMA_SOURCE_DIR)

foreach(_variable IN LISTS _required_variables)
    if(NOT DEFINED ${_variable} OR "${${_variable}}" STREQUAL "")
        message(FATAL_ERROR "${_variable} is required")
    endif()
endforeach()

file(READ "${LLAMA_BUILD_ID_FEATURE_FILE}" _feature_manifest)

set(_git_source_state "")
if(DEFINED LLAMA_GIT_EXECUTABLE AND NOT "${LLAMA_GIT_EXECUTABLE}" STREQUAL "")
    execute_process(
        COMMAND "${LLAMA_GIT_EXECUTABLE}" rev-parse --is-inside-work-tree
        WORKING_DIRECTORY "${LLAMA_SOURCE_DIR}"
        RESULT_VARIABLE _git_result
        OUTPUT_VARIABLE _inside_work_tree
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(_git_result EQUAL 0 AND _inside_work_tree STREQUAL "true")
        execute_process(
            COMMAND "${LLAMA_GIT_EXECUTABLE}" rev-parse --show-toplevel
            WORKING_DIRECTORY "${LLAMA_SOURCE_DIR}"
            RESULT_VARIABLE _top_level_result
            OUTPUT_VARIABLE _git_top_level
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        get_filename_component(
            _source_real_path "${LLAMA_SOURCE_DIR}" REALPATH)
        get_filename_component(
            _git_real_path "${_git_top_level}" REALPATH)
        file(TO_CMAKE_PATH "${_source_real_path}" _source_real_path)
        file(TO_CMAKE_PATH "${_git_real_path}" _git_real_path)
        if(WIN32)
            string(TOLOWER "${_source_real_path}" _source_real_path)
            string(TOLOWER "${_git_real_path}" _git_real_path)
        endif()
    endif()

    if(_git_result EQUAL 0 AND
       _inside_work_tree STREQUAL "true" AND
       _top_level_result EQUAL 0 AND
       _source_real_path STREQUAL _git_real_path)
        execute_process(
            COMMAND "${LLAMA_GIT_EXECUTABLE}" rev-parse --verify HEAD^{tree}
            WORKING_DIRECTORY "${LLAMA_SOURCE_DIR}"
            RESULT_VARIABLE _tree_result
            OUTPUT_VARIABLE _head_tree
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(
            COMMAND "${LLAMA_GIT_EXECUTABLE}"
                -c core.quotepath=true
                -c color.ui=false
                diff --binary --full-index --no-ext-diff --no-textconv
                --no-renames --no-color --ignore-submodules=none
                --diff-algorithm=myers --no-indent-heuristic HEAD -- .
            WORKING_DIRECTORY "${LLAMA_SOURCE_DIR}"
            RESULT_VARIABLE _diff_result
            OUTPUT_VARIABLE _tracked_diff
            ERROR_QUIET)

        if(_tree_result EQUAL 0 AND _diff_result EQUAL 0)
            string(SHA256 _diff_hash "${_tracked_diff}")
            string(CONCAT _git_source_state
                "source_state=git-v1\n"
                "head_tree=${_head_tree}\n"
                "diff_sha256=${_diff_hash}\n")
        endif()
    endif()
endif()

if(_git_source_state STREQUAL "")
    # Source archives do not have Git's tracked-file metadata. Hash their file
    # contents in stable path order, excluding an in-tree build directory.
    file(GLOB_RECURSE _source_files
        LIST_DIRECTORIES false
        RELATIVE "${LLAMA_SOURCE_DIR}"
        "${LLAMA_SOURCE_DIR}/*")
    list(SORT _source_files)

    get_filename_component(_source_dir "${LLAMA_SOURCE_DIR}" REALPATH)
    get_filename_component(_binary_dir "${LLAMA_BINARY_DIR}" REALPATH)
    file(TO_CMAKE_PATH "${_source_dir}" _source_dir)
    file(TO_CMAKE_PATH "${_binary_dir}" _binary_dir)
    string(REGEX REPLACE "/$" "" _source_dir "${_source_dir}")
    string(REGEX REPLACE "/$" "" _binary_dir "${_binary_dir}")
    if(WIN32)
        string(TOLOWER "${_source_dir}" _source_dir)
        string(TOLOWER "${_binary_dir}" _binary_dir)
    endif()

    if(_source_dir STREQUAL _binary_dir)
        message(FATAL_ERROR
            "A build without Git metadata must use a separate build directory")
    endif()

    string(FIND "${_binary_dir}/" "${_source_dir}/" _binary_in_source)
    set(_source_manifest "")

    foreach(_relative_path IN LISTS _source_files)
        if(_relative_path MATCHES "^\\.git(/|$)")
            continue()
        endif()

        set(_absolute_path "${_source_dir}/${_relative_path}")
        file(TO_CMAKE_PATH "${_absolute_path}" _absolute_path)
        if(_binary_in_source EQUAL 0)
            string(FIND "${_absolute_path}/" "${_binary_dir}/" _binary_prefix)
            if(_binary_prefix EQUAL 0)
                continue()
            endif()
        endif()

        if(IS_SYMLINK "${_absolute_path}")
            file(READ_SYMLINK "${_absolute_path}" _symlink_target)
            string(SHA256 _content_hash "symlink:${_symlink_target}")
        else()
            file(SHA256 "${_absolute_path}" _content_hash)
        endif()

        string(LENGTH "${_relative_path}" _path_length)
        string(APPEND _source_manifest
            "${_path_length}:${_relative_path}:${_content_hash}\n")
    endforeach()

    string(SHA256 _source_manifest_hash "${_source_manifest}")
    set(_git_source_state
        "source_state=archive-v1\nfiles_sha256=${_source_manifest_hash}\n")
endif()

string(CONCAT _identity_material
    "llama_build_id_format=1\n"
    "${_git_source_state}"
    "${_feature_manifest}")
string(SHA256 LLAMA_BUILD_ID "${_identity_material}")

string(LENGTH "${LLAMA_BUILD_ID}" _build_id_length)
if(NOT _build_id_length EQUAL 64 OR NOT LLAMA_BUILD_ID MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "Generated an invalid libllama build identity")
endif()

configure_file(
    "${LLAMA_BUILD_ID_TEMPLATE}"
    "${LLAMA_BUILD_ID_OUTPUT}"
    @ONLY)
configure_file(
    "${LLAMA_BUILD_ID_HEADER_TEMPLATE}"
    "${LLAMA_BUILD_ID_HEADER}"
    @ONLY)
