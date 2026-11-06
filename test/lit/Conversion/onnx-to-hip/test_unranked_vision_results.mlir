// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

// Phi's vision graph has known input ranks but no output shape metadata on
// fused normalization and quantized MLP nodes. Refinement must propagate along
// the chain without interpreting an unranked tensor as a ranked tensor.
module {
  // CHECK-LABEL: func.func @vision_mlp
  // CHECK: hip.layer_norm
  // CHECK: hip.matmul_nbits
  // CHECK: hip.fast_gelu
  // CHECK: hip.matmul_nbits
  // CHECK: hip.add
  func.func @vision_mlp(%x: tensor<?x?x1152xf16>, %scale: tensor<1152xf16>, %bias: tensor<1152xf16>, %w1: tensor<4304x9x64xui8>, %s1: tensor<38736xf16>, %w2: tensor<1152x34x64xui8>, %s2: tensor<39168xf16>, %gelu_bias: tensor<4304xf16>) -> tensor<?x?x1152xf16> {
    %norm = "onnx.LayerNormalization"(%x, %scale, %bias) {axis = -1 : si64, epsilon = 1.0e-6 : f32, stash_type = 1 : si64} : (tensor<?x?x1152xf16>, tensor<1152xf16>, tensor<1152xf16>) -> tensor<*xf16>
    %fc1 = "onnx.Custom"(%norm, %w1, %s1) {domain_name = "com.microsoft", function_name = "MatMulNBits", K = 1152 : si64, N = 4304 : si64, bits = 4 : si64, block_size = 128 : si64} : (tensor<*xf16>, tensor<4304x9x64xui8>, tensor<38736xf16>) -> tensor<*xf16>
    %gelu = "onnx.Custom"(%fc1, %gelu_bias) {domain_name = "com.microsoft", function_name = "FastGelu"} : (tensor<*xf16>, tensor<4304xf16>) -> tensor<*xf16>
    %fc2 = "onnx.Custom"(%gelu, %w2, %s2) {domain_name = "com.microsoft", function_name = "MatMulNBits", K = 4304 : si64, N = 1152 : si64, bits = 4 : si64, block_size = 128 : si64} : (tensor<*xf16>, tensor<1152x34x64xui8>, tensor<39168xf16>) -> tensor<*xf16>
    %sum = "onnx.Add"(%x, %fc2) : (tensor<?x?x1152xf16>, tensor<*xf16>) -> tensor<*xf16>
    %ranked = tensor.cast %sum : tensor<*xf16> to tensor<?x?x1152xf16>
    "onnx.Return"(%ranked) : (tensor<?x?x1152xf16>) -> ()
  }

  // CHECK-LABEL: func.func @skip_norm
  // CHECK: hip.add
  // CHECK: hip.layer_norm
  func.func @skip_norm(%x: tensor<?x?x1152xf16>, %skip: tensor<?x?x1152xf16>, %scale: tensor<1152xf16>, %bias: tensor<1152xf16>) -> tensor<?x?x1152xf16> {
    %norm:4 = "onnx.Custom"(%x, %skip, %scale, %bias) {domain_name = "com.microsoft", function_name = "SkipLayerNormalization", epsilon = 1.0e-6 : f32} : (tensor<?x?x1152xf16>, tensor<?x?x1152xf16>, tensor<1152xf16>, tensor<1152xf16>) -> (tensor<*xf16>, none, none, tensor<*xf16>)
    %ranked = tensor.cast %norm#0 : tensor<*xf16> to tensor<?x?x1152xf16>
    "onnx.Return"(%ranked) : (tensor<?x?x1152xf16>) -> ()
  }
}
