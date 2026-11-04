## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.

include_guard(GLOBAL)

# Shared by the FetchContent source build and Windows LLVM prebuild workflow.
# This patches sources only; an existing external LLVM install must be rebuilt.
function(hip_ep_apply_llvm_patches src_dir)
  find_package(Git QUIET REQUIRED)
  set(_patch "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/llvm-fix-null-thread-handle.patch")
  if(NOT EXISTS "${src_dir}/external/llvm-project/llvm/lib/Support/Windows/Threading.inc")
    message(FATAL_ERROR "Missing LLVM Windows threading source in ${src_dir}")
  endif()

  # Reconfigure safely without resetting unrelated changes in the checkout.
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --reverse --check --ignore-whitespace "${_patch}"
    WORKING_DIRECTORY "${src_dir}"
    RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_QUIET)
  if(_rc EQUAL 0)
    message(STATUS "LLVM null thread handle fix already applied")
    return()
  endif()

  execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --check --ignore-whitespace "${_patch}"
    WORKING_DIRECTORY "${src_dir}"
    RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "Cannot apply LLVM null thread handle fix in ${src_dir}: ${_out}${_err}")
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --ignore-whitespace "${_patch}"
    WORKING_DIRECTORY "${src_dir}"
    RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "Failed to apply LLVM null thread handle fix in ${src_dir}: ${_out}${_err}")
  endif()
  message(STATUS "Applied LLVM null thread handle fix (upstream e48cebf)")
endfunction()

# Standalone entry point for the CI prebuild, before configuring LLVM.
if(DEFINED HIP_EP_LLVM_PATCH_SOURCE_DIR)
  hip_ep_apply_llvm_patches("${HIP_EP_LLVM_PATCH_SOURCE_DIR}")
endif()
