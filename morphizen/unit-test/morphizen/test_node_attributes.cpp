/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./test_environment.hpp"
#include "morphizen/morphizen-ort-api-ext.hpp"
#include "morphizen/morphizen.hpp"

namespace {
class MLIRNodeAttributesTest : public ::testing::Test {
protected:
  void SetUp() override {
    auto *api = morphizen::get_the_global_api_unsafe();
    ASSERT_NE(api, nullptr);
    auto name = api->get_lib_name();
    if (!name.get() || *name != "morphizen-mlir-imp") {
      GTEST_SKIP() << "Requires the MLIR backend";
    }
  }
};

TEST_F(MLIRNodeAttributesTest, RepeatedOwnedAttributeLifetimes) {
  // Run under LeakSanitizer to catch detached placeholder operations that
  // survive NodeAttributesPtr destruction, including empty and unused attrs.
  for (int i = 0; i < 128; ++i) {
    auto empty = morphizen::NodeAttributesBuilder().build();
    ASSERT_NE(empty.get(), nullptr);
    auto attrs = morphizen::NodeAttributesBuilder()
                     .add("axis", int64_t(i))
                     .build();
    ASSERT_NE(morphizen::node_attributes_get(*attrs, "axis"), nullptr);
  }
  MORPHIZEN_ORT_API(node_attributes_delete)(nullptr);
}

TEST_F(MLIRNodeAttributesTest, NodeKeepsAttributesAfterTemporaryAndCloneRelease) {
  auto model = morphizen_cxx::Model::create(
      CMAKE_CURRENT_BINARY_PATH / "node_attributes_lifetime.onnx", {{"", 17}});
  auto graph = model->main_graph();
  auto input = graph.new_node_arg(
      "input", {2, 3}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto output = graph.new_node_arg(
      "output", {2, 3}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto node = graph.add_node(
      "softmax", "", "Softmax", "", {input}, {output},
      morphizen::NodeAttributesBuilder().add("axis", int64_t(1)).build());

  // The builder's owning pointer has already been destroyed by add_node.
  EXPECT_EQ(node.get_attr_int("axis"), 1);
  {
    auto cloned = morphizen::node_clone_attributes(*node.ptr());
    ASSERT_NE(morphizen::node_attributes_get(*cloned, "axis"), nullptr);
  }
  // A borrowed view remains part of the real node. Deleting the clone must
  // destroy only its detached placeholder, leaving the real node untouched.
  const auto &borrowed = morphizen::node_get_attributes_ref(*node.ptr());
  ASSERT_NE(morphizen::node_attributes_get(borrowed, "axis"), nullptr);
  EXPECT_EQ(node.get_attr_int("axis"), 1);
  graph.set_inputs({input});
  graph.set_outputs({output});
  graph.resolve();
}
} // namespace
