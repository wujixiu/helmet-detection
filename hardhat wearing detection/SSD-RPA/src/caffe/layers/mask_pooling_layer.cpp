// --------------------------------------------------------
// Multitask Network Cascade
// Written by Haozhi Qi
// Copyright (c) 2016, Haozhi Qi
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------

#include "caffe/layers/mask_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void MaskPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // This layer do not need any paramter
}

template <typename Dtype>
void MaskPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[0]->num(), channels_, height_, width_);

  CHECK_EQ(bottom[0]->height(), bottom[1]->height()) << "feature map height and mask height must be the same";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width()) << "feature map width and mask width must be the same";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "feature map num and mask num must be the same";
}

template <typename Dtype>
void MaskPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void MaskPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(MaskPoolingLayer);
#endif

INSTANTIATE_CLASS(MaskPoolingLayer);
REGISTER_LAYER_CLASS(MaskPooling);

} // namespace caffe
