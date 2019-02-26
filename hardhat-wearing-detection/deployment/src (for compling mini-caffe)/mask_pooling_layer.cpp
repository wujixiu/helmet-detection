// --------------------------------------------------------
// Multitask Network Cascade
// Written by Haozhi Qi
// Copyright (c) 2016, Haozhi Qi
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------

#include "./mask_pooling_layer.hpp"

namespace caffe {


void MaskPoolingLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) {

  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[0]->num(), channels_, height_, width_);

  CHECK_EQ(bottom[0]->height(), bottom[1]->height()) << "feature map height and mask height must be the same";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width()) << "feature map width and mask width must be the same";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "feature map num and mask num must be the same";
}

void MaskPoolingLayer::Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {

	const real_t* bottom_data = bottom[0]->cpu_data();
	const real_t* bottom_masks = bottom[1]->cpu_data();
	real_t* top_data = top[0]->mutable_cpu_data();
	int count = top[0]->count();
	for (int index = 0; index < count; index++){
		int pw = index % width_;
		int ph = (index / width_) % height_;
		int n = index / width_ / height_ / channels_;
		int mask_index = n * height_ * width_ + ph * width_ + pw;
		top_data[index] = bottom_data[index] * bottom_masks[mask_index];
	}
}
#ifndef USE_CUDA
STUB_GPU(MaskPoolingLayer);
#endif
static shared_ptr<Layer> CreateLayer(const LayerParameter &param) {
	return shared_ptr<Layer>(new MaskPoolingLayer(param));
}

REGISTER_LAYER_CREATOR(MaskPooling, CreateLayer);

} // namespace caffe
