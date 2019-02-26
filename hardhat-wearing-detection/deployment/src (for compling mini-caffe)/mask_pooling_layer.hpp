// --------------------------------------------------------
// Multitask Network Cascade
// Written by Haozhi Qi
// Copyright (c) 2016, Haozhi Qi
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------


#ifndef CAFFE_MASK_POOLING_LAYERS_HPP_
#define CAFFE_MASK_POOLING_LAYERS_HPP_


#include "../layer.hpp"

namespace caffe {

class MaskPoolingLayer : public Layer {
  public:
    explicit MaskPoolingLayer(const LayerParameter& param)
        : Layer(param) {}

    virtual void Reshape(const vector<Blob*>& bottom,
        const vector<Blob*>& top);

    virtual inline const char* type() const { return "MaskPooling"; }
    virtual inline int MinBottomBlobs() const { return 2; }
    virtual inline int MaxBottomBlobs() const { return 2; }
    virtual inline int MinTopBlobs() const { return 1; }
    virtual inline int MaxTopBlobs() const { return 1; }
  protected:
    virtual void Forward_cpu(const vector<Blob*>& bottom,
        const vector<Blob*>& top);

	virtual void Forward_gpu(const vector<Blob*>& bottom,
		const vector<Blob*>& top);

    int channels_;
    int height_;
    int width_;
};

}

#endif // CAFFE_MASK_POOLING_LAYERS_HPP_
