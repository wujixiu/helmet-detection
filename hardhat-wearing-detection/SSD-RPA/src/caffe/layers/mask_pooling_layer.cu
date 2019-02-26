// --------------------------------------------------------
// Multitask Network Cascade
// Written by Haozhi Qi
// Copyright (c) 2016, Haozhi Qi
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------

#include "caffe/layers/mask_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaskPoolingForward(const int nthreads, const Dtype* bottom_data, const Dtype* bottom_masks,
			       Dtype* top_data, const int channels, const int height, const int width) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the top output
    int pw = index % width;
    int ph = (index / width) % height;
    // int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int mask_index = n * height * width + ph * width + pw;

    // top feature map has identical shape with bottom feature map, so we reuse index here
    top_data[index] = bottom_data[index] * bottom_masks[mask_index];
  }
}

template <typename Dtype>
void MaskPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
					   const vector<Blob<Dtype>*>& top) {
  // bottom[0] is feature maps, of shape (n x c x h x w)
  // bottom[1] is masks, of shape (n x 1 x h x w)
  // output(n, c, h, w) = input_feature(n, c, h, w) * input_mask(n, 1, h, w)
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_masks = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  MaskPoolingForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>>
  (count, bottom_data, bottom_masks, top_data, channels_, height_, width_);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void MaskPoolingBackwardFeature(const int nthreads, const Dtype* bottom_data, const Dtype* bottom_masks,
        Dtype* bottom_diff, const Dtype* top_diff, const int channels, const int height, const int width) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    // int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    // output w,h coordinate has the same size with input's w,h coordinate
    int mask_index = n * height * width + h * width + w;
    Dtype float_mask = bottom_masks[mask_index];
    bottom_diff[index] = top_diff[index] * float_mask;
  }
}

template <typename Dtype>
__global__ void MaskPoolingBackwardMask(const int nthreads, const Dtype* bottom_data, Dtype* bottom_diff,
  const Dtype* top_diff, const int channels, const int height, const int width) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, w, h) are index of mask element, with channel dim = 1
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height / 1;
    Dtype gradient = 0.0;
    for (int i = 0; i < channels; ++i) {
      int data_index = ((n * channels + i) * height + h) * width + w;
      gradient += top_diff[data_index] * bottom_data[data_index];
    }
    int mask_index = ((n * height) + h) * width + w;
    bottom_diff[mask_index] = gradient;
  }
}

template <typename Dtype>
void MaskPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
					    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_masks = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);

  if (propagate_down[0]) {
    MaskPoolingBackwardFeature<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>>
      (count, bottom_data, bottom_masks, bottom_diff, top_diff, channels_, height_, width_);
  }
  Dtype* bottom_mask_diff = bottom[1]->mutable_gpu_diff();
  count = bottom[1]->count();
  if (propagate_down[1]) {
    MaskPoolingBackwardMask<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>>
      (count, bottom_data, bottom_mask_diff, top_diff, channels_, height_, width_);
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(MaskPoolingLayer);

} // namespace caffe
