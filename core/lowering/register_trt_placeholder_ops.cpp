#include "torch/csrc/jit/runtime/custom_operator.h"

namespace torch {
namespace jit {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

#if TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 7  //>= 1.7
RegisterOperators trt_placeholder_ops_reg({
    /// Op marks a Tensor to be conveted from an Torch Tensor
    /// to a TRT constant Tensor
    Operator("trt::const(Tensor val) -> Tensor", [](Stack* stack) {}, aliasAnalysisFromSchema()),
});
#else 
RegisterOperators trt_placeholder_ops_reg({
  /// Op marks a Tensor to be conveted from an Torch Tensor
  /// to a TRT constant Tensor
  Operator(
    "trt::const(Tensor val) -> Tensor",
    [](Stack& stack) {
      return 0; //noop
    },
    aliasAnalysisFromSchema()),
});
#endif 


} // namespace jit
} // namespace torch
