// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename T>
__device__ inline T get_pixel_val(
    const T* tensor,
    const int idx,
    const int H,
    const int W,
    const int y,
    const int x,
    const int V,
    const int U,
    const int v,
    const int u,
    const T pad_val) {
  if ((y < 0) || (y >= H) || (x < 0) || (x >= W) || (v < 0) || (v >= V) ||
      (u < 0) || (u >= U)) {
    return pad_val;
  } else {
    return tensor[(((idx * V + v) * U + u) * H + y) * W + x];
  }
}

template <typename T>
__device__ inline void add_pixel_val(
    T* tensor,
    const T val,
    const int idx,
    const int H,
    const int W,
    const int y,
    const int x,
    const int V,
    const int U,
    const int v,
    const int u) {
  if ((val == 0.) || (y < 0) || (y >= H) || (x < 0) || (x >= W) || (v < 0) ||
      (v >= V) || (u < 0) || (u >= U)) {
    return;
  } else {
    atomicAdd(tensor + ((((idx * V + v) * U + u) * H + y) * W + x), val);
  }
}

template <typename T>
__global__ void SwapAlign2NatForwardFeat(
    const int nthreads,
    const T* bottom_data,
    const int Vout,
    const int Uout,
    const float hVout,
    const float hUout,
    const int Vin,
    const int Uin,
    const float lambda,
    const int Hin,
    const int Win,
    const int Hout,
    const int Wout,
    const T pad_val,
    T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int idx = index;
    const int x = idx % Wout;
    idx /= Wout;
    const int y = idx % Hout;
    idx /= Hout;
    const int u = idx % Uout;
    idx /= Uout;
    const int v = idx % Vout;
    idx /= Vout;

    const float ox = x * lambda + u - hUout + 0.5;
    const int xf = static_cast<int>(floor(ox));
    const int xc = static_cast<int>(ceil(ox));
    const float xwc = ox - xf;
    const float xwf = 1. - xwc;

    const float oy = y * lambda + v - hVout + 0.5;
    const int yf = static_cast<int>(floor(oy));
    const int yc = static_cast<int>(ceil(oy));
    const float ywc = oy - yf;
    const float ywf = 1. - ywc;

    const float ou = (u + 0.5) / lambda - 0.5;
    const int uf = static_cast<int>(floor(ou));
    const int uc = static_cast<int>(ceil(ou));
    const float uwc = ou - uf;
    const float uwf = 1. - uwc;

    const float ov = (v + 0.5) / lambda - 0.5;
    const int vf = static_cast<int>(floor(ov));
    const int vc = static_cast<int>(ceil(ov));
    const float vwc = ov - vf;
    const float vwf = 1. - vwc;

    T val = ywf * xwf * vwf * uwf *
            get_pixel_val(
                bottom_data, idx, Hin, Win, yf, xf, Vin, Uin, vf, uf, pad_val) +
        ywf * xwf * vwf * uwc *
            get_pixel_val(
                bottom_data, idx, Hin, Win, yf, xf, Vin, Uin, vf, uc, pad_val) +
        ywf * xwf * vwc * uwf *
            get_pixel_val(
                bottom_data, idx, Hin, Win, yf, xf, Vin, Uin, vc, uf, pad_val) +
        ywf * xwf * vwc * uwc *
            get_pixel_val(
                bottom_data, idx, Hin, Win, yf, xf, Vin, Uin, vc, uc, pad_val) +
        ywf * xwc * vwf * uwf *
            get_pixel_val(
                bottom_data, idx, Hin, Win, yf, xc, Vin, Uin, vf, uf, pad_val) +
        ywf * xwc * vwf * uwc *
            get_pixel_val(
                bottom_data, idx, Hin, Win, yf, xc, Vin, Uin, vf, uc, pad_val) +
        ywf * xwc * vwc * uwf *
            get_pixel_val(
                bottom_data, idx, Hin, Win, yf, xc, Vin, Uin, vc, uf, pad_val) +
        ywf * xwc * vwc * uwc *
            get_pixel_val(
                bottom_data, idx, Hin, Win, yf, xc, Vin, Uin, vc, uc, pad_val) +
        ywc * xwf * vwf * uwf *
            get_pixel_val(
                bottom_data, idx, Hin, Win, yc, xf, Vin, Uin, vf, uf, pad_val) +
        ywc * xwf * vwf * uwc *
            get_pixel_val(
                bottom_data, idx, Hin, Win, yc, xf, Vin, Uin, vf, uc, pad_val) +
        ywc * xwf * vwc * uwf *
            get_pixel_val(
                bottom_data, idx, Hin, Win, yc, xf, Vin, Uin, vc, uf, pad_val) +
        ywc * xwf * vwc * uwc *
            get_pixel_val(
                bottom_data, idx, Hin, Win, yc, xf, Vin, Uin, vc, uc, pad_val) +
        ywc * xwc * vwf * uwf *
            get_pixel_val(
                bottom_data, idx, Hin, Win, yc, xc, Vin, Uin, vf, uf, pad_val) +
        ywc * xwc * vwf * uwc *
            get_pixel_val(
                bottom_data, idx, Hin, Win, yc, xc, Vin, Uin, vf, uc, pad_val) +
        ywc * xwc * vwc * uwf *
            get_pixel_val(
                bottom_data, idx, Hin, Win, yc, xc, Vin, Uin, vc, uf, pad_val) +
        ywc * xwc * vwc * uwc *
            get_pixel_val(
                bottom_data, idx, Hin, Win, yc, xc, Vin, Uin, vc, uc, pad_val);

    top_data[index] = val;
  }
}

template <typename T>
__global__ void SwapAlign2NatBackwardFeat(
    const int nthreads,
    const T* top_diff,
    const int Vout,
    const int Uout,
    const float hVout,
    const float hUout,
    const int Vin,
    const int Uin,
    const float lambda,
    const int Hin,
    const int Win,
    const int Hout,
    const int Wout,
    T* bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int idx = index;
    const int x = idx % Wout;
    idx /= Wout;
    const int y = idx % Hout;
    idx /= Hout;
    const int u = idx % Uout;
    idx /= Uout;
    const int v = idx % Vout;
    idx /= Vout;

    const float ox = x * lambda + u - hUout + 0.5;
    const int xf = static_cast<int>(floor(ox));
    const int xc = static_cast<int>(ceil(ox));
    const float xwc = ox - xf;
    const float xwf = 1. - xwc;

    const float oy = y * lambda + v - hVout + 0.5;
    const int yf = static_cast<int>(floor(oy));
    const int yc = static_cast<int>(ceil(oy));
    const float ywc = oy - yf;
    const float ywf = 1. - ywc;

    const float ou = (u + 0.5) / lambda - 0.5;
    const int uf = static_cast<int>(floor(ou));
    const int uc = static_cast<int>(ceil(ou));
    const float uwc = ou - uf;
    const float uwf = 1. - uwc;

    const float ov = (v + 0.5) / lambda - 0.5;
    const int vf = static_cast<int>(floor(ov));
    const int vc = static_cast<int>(ceil(ov));
    const float vwc = ov - vf;
    const float vwf = 1. - vwc;

    const T grad = top_diff[index];

    add_pixel_val(
        bottom_diff,
        ywf * xwf * vwf * uwf * grad,
        idx,
        Hin,
        Win,
        yf,
        xf,
        Vin,
        Uin,
        vf,
        uf);
    add_pixel_val(
        bottom_diff,
        ywf * xwf * vwf * uwc * grad,
        idx,
        Hin,
        Win,
        yf,
        xf,
        Vin,
        Uin,
        vf,
        uc);
    add_pixel_val(
        bottom_diff,
        ywf * xwf * vwc * uwf * grad,
        idx,
        Hin,
        Win,
        yf,
        xf,
        Vin,
        Uin,
        vc,
        uf);
    add_pixel_val(
        bottom_diff,
        ywf * xwf * vwc * uwc * grad,
        idx,
        Hin,
        Win,
        yf,
        xf,
        Vin,
        Uin,
        vc,
        uc);
    add_pixel_val(
        bottom_diff,
        ywf * xwc * vwf * uwf * grad,
        idx,
        Hin,
        Win,
        yf,
        xc,
        Vin,
        Uin,
        vf,
        uf);
    add_pixel_val(
        bottom_diff,
        ywf * xwc * vwf * uwc * grad,
        idx,
        Hin,
        Win,
        yf,
        xc,
        Vin,
        Uin,
        vf,
        uc);
    add_pixel_val(
        bottom_diff,
        ywf * xwc * vwc * uwf * grad,
        idx,
        Hin,
        Win,
        yf,
        xc,
        Vin,
        Uin,
        vc,
        uf);
    add_pixel_val(
        bottom_diff,
        ywf * xwc * vwc * uwc * grad,
        idx,
        Hin,
        Win,
        yf,
        xc,
        Vin,
        Uin,
        vc,
        uc);
    add_pixel_val(
        bottom_diff,
        ywc * xwf * vwf * uwf * grad,
        idx,
        Hin,
        Win,
        yc,
        xf,
        Vin,
        Uin,
        vf,
        uf);
    add_pixel_val(
        bottom_diff,
        ywc * xwf * vwf * uwc * grad,
        idx,
        Hin,
        Win,
        yc,
        xf,
        Vin,
        Uin,
        vf,
        uc);
    add_pixel_val(
        bottom_diff,
        ywc * xwf * vwc * uwf * grad,
        idx,
        Hin,
        Win,
        yc,
        xf,
        Vin,
        Uin,
        vc,
        uf);
    add_pixel_val(
        bottom_diff,
        ywc * xwf * vwc * uwc * grad,
        idx,
        Hin,
        Win,
        yc,
        xf,
        Vin,
        Uin,
        vc,
        uc);
    add_pixel_val(
        bottom_diff,
        ywc * xwc * vwf * uwf * grad,
        idx,
        Hin,
        Win,
        yc,
        xc,
        Vin,
        Uin,
        vf,
        uf);
    add_pixel_val(
        bottom_diff,
        ywc * xwc * vwf * uwc * grad,
        idx,
        Hin,
        Win,
        yc,
        xc,
        Vin,
        Uin,
        vf,
        uc);
    add_pixel_val(
        bottom_diff,
        ywc * xwc * vwc * uwf * grad,
        idx,
        Hin,
        Win,
        yc,
        xc,
        Vin,
        Uin,
        vc,
        uf);
    add_pixel_val(
        bottom_diff,
        ywc * xwc * vwc * uwc * grad,
        idx,
        Hin,
        Win,
        yc,
        xc,
        Vin,
        Uin,
        vc,
        uc);
  }
}

namespace tensormask {

at::Tensor SwapAlign2Nat_forward_cuda(
    const at::Tensor& X,
    const int lambda_val,
    const float pad_val) {
  AT_ASSERTM(X.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(X.ndimension() == 4, "input must be a 4D tensor");
  AT_ASSERTM(lambda_val >= 1, "lambda should be greater or equal to 1");
  const int N = X.size(0);
  const int C = X.size(1);
  const int Vin = static_cast<int>(sqrt(static_cast<float>(C)));
  const int Uin = C / Vin;
  AT_ASSERTM(
      C == Vin * Uin && Vin == Uin, "#channels should be a square number");
  const int Vout = lambda_val * Vin;
  const int Uout = lambda_val * Uin;
  const int Hin = X.size(2);
  const int Win = X.size(3);
  const float lambda = static_cast<float>(lambda_val);
  const int Hout = static_cast<int>(ceil(Hin / lambda));
  const int Wout = static_cast<int>(ceil(Win / lambda));
  const float hVout = Vout / 2.;
  const float hUout = Uout / 2.;

  at::cuda::CUDAGuard device_guard(X.device());

  at::Tensor Y = at::empty({N, Vout * Uout, Hout, Wout}, X.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(at::cuda::ATenCeilDiv(Y.numel(), 512L), 4096L));
  dim3 block(512);

  if (Y.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return Y;
  }

  auto X_ = X.contiguous();
  AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "SwapAlign2Nat_forward", [&] {
    SwapAlign2NatForwardFeat<scalar_t><<<grid, block, 0, stream>>>(
        Y.numel(),
        X_.data_ptr<scalar_t>(),
        Vout,
        Uout,
        hVout,
        hUout,
        Vin,
        Uin,
        lambda,
        Hin,
        Win,
        Hout,
        Wout,
        pad_val,
        Y.data_ptr<scalar_t>());
  });
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());
  return Y;
}

at::Tensor SwapAlign2Nat_backward_cuda(
    const at::Tensor& gY,
    const int lambda_val,
    const int batch_size,
    const int channel,
    const int height,
    const int width) {
  AT_ASSERTM(gY.device().is_cuda(), "input gradient must be a CUDA tensor");
  AT_ASSERTM(gY.ndimension() == 4, "input gradient must be a 4D tensor");
  AT_ASSERTM(lambda_val >= 1, "lambda should be greater or equal to 1");
  const int Vin = static_cast<int>(sqrt(static_cast<float>(channel)));
  const int Uin = channel / Vin;
  const int Vout = lambda_val * Vin;
  const int Uout = lambda_val * Uin;
  const float hVout = Vout / 2.;
  const float hUout = Uout / 2.;
  const int Hout = gY.size(2);
  const int Wout = gY.size(3);

  at::cuda::CUDAGuard device_guard(gY.device());

  at::Tensor gX = at::zeros({batch_size, channel, height, width}, gY.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(at::cuda::ATenCeilDiv(gY.numel(), 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (gY.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return gX;
  }

  auto gY_ = gY.contiguous();
  AT_DISPATCH_FLOATING_TYPES(gY.scalar_type(), "SwapAlign2Nat_backward", [&] {
    SwapAlign2NatBackwardFeat<scalar_t><<<grid, block, 0, stream>>>(
        gY.numel(),
        gY_.data_ptr<scalar_t>(),
        Vout,
        Uout,
        hVout,
        hUout,
        Vin,
        Uin,
        static_cast<float>(lambda_val),
        height,
        width,
        Hout,
        Wout,
        gX.data_ptr<scalar_t>());
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return gX;
}

} // namespace tensormask
