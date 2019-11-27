// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include <torch/extension.h>
#include "SwapAlign2Nat/SwapAlign2Nat.h"

namespace tensormask {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "swap_align2nat_forward",
      &SwapAlign2Nat_forward,
      "SwapAlign2Nat_forward");
  m.def(
      "swap_align2nat_backward",
      &SwapAlign2Nat_backward,
      "SwapAlign2Nat_backward");
}

} // namespace tensormask
