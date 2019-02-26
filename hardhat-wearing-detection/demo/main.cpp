#include <vector>
#include <algorithm>
#include <stdio.h>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace caffe;

struct BBox {
	float x1, y1, x2, y2, score;
	int label;
};
void preprocess(cv::Mat& frame, cv::Mat& preprocessed, cv::Size caffeInputSize, cv::Scalar mean, float inputscale);
void resizeImage(cv::Mat& frame, cv::Mat& dst, cv::Size dims);

int main(int argc, char* argv[]) {
	if (caffe::GPUAvailable()) {
		caffe::SetMode(caffe::GPU, 0);
	}
	
	vector<cv::Scalar> showColor = { cv::Scalar(0), cv::Scalar(0, 255, 255), cv::Scalar(0, 0, 255),
		cv::Scalar(255, 0, 0), cv::Scalar::all(255), cv::Scalar(0, 255, 0)};

	const char* kClassNames[] = { "__background__", "yellow", "red", "blue", "white", "none" };

	Net net("../models/Pelee_RPA.prototxt");
	net.CopyTrainedLayersFrom("../models/Pelee_RPA.caffemodel");
	Mat img = imread("../imgs/demo.jpg");

	Profiler* profiler = Profiler::Get();
	profiler->TurnON();
	uint64_t tic = profiler->Now();

	// preprocess
	const float kScoreThreshold = 0.1f;
	cv::Mat preprocessed;
	cv::Size caffeInputSize(304, 304);
	cv::Scalar mean(103.94, 116.78, 123.68);
	float inputscale = 0.017f;
	int height = img.rows;
	int width = img.cols;

	preprocess(img, preprocessed, caffeInputSize, mean, inputscale);

	vector<Mat> bgr;
	cv::split(preprocessed, bgr);

	// fill network input
	shared_ptr<Blob> data = net.blob_by_name("data");
	data->Reshape(1, 3, preprocessed.rows, preprocessed.cols);
	const int bias = data->offset(0, 1, 0, 0);
	const int bytes = bias * sizeof(float);
	memcpy(data->mutable_cpu_data() + 0 * bias, bgr[0].data, bytes);
	memcpy(data->mutable_cpu_data() + 1 * bias, bgr[1].data, bytes);
	memcpy(data->mutable_cpu_data() + 2 * bias, bgr[2].data, bytes);

	// forward
	net.Forward();

	// get output, shape is N x 7
	shared_ptr<Blob> result = net.blob_by_name("detection_out");
	const float* result_data = result->cpu_data();
	const int num_det = result->num();
	vector<BBox> detections;
	for (int k = 0; k < num_det; ++k) {
		if (result_data[0] != -1 && result_data[2] > kScoreThreshold) {
			// [image_id, label, score, xmin, ymin, xmax, ymax]
			BBox bbox;
			bbox.x1 = result_data[3] * width;
			bbox.y1 = result_data[4] * height;
			bbox.x2 = result_data[5] * width;
			bbox.y2 = result_data[6] * height;
			bbox.score = result_data[2];
			bbox.label = static_cast<int>(result_data[1]);
			detections.push_back(bbox);
		}
		result_data += 7;
	}

	// draw
	for (auto& bbox : detections) {
		cv::Rect rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1 + 1, bbox.y2 - bbox.y1 + 1);
		cv::rectangle(img, rect, showColor[bbox.label], 2);
		char buff[300];
		_snprintf_s(buff, sizeof(buff), "%s: %.2f", kClassNames[bbox.label], bbox.score);
		cv::putText(img, buff, cv::Point(bbox.x1, bbox.y1), FONT_HERSHEY_PLAIN, 1, showColor[bbox.label]);
	}
	uint64_t toc = profiler->Now();
	profiler->TurnOFF();
	profiler->DumpProfile("./ssd-profile.json");

	LOG(INFO) << "Costs " << (toc - tic) / 1000.f << " ms";
	cv::imwrite("./ssd-result.jpg", img);
	cv::imshow("result", img);
	cv::waitKey(0);
	return 0;
}

void preprocess(cv::Mat& frame, cv::Mat& preprocessed, cv::Size caffeInputSize, cv::Scalar mean, float inputscale)
{

	frame.convertTo(preprocessed, CV_32F, 1.0 / 255.0, 0.0);
	resizeImage(preprocessed, preprocessed, caffeInputSize);
	preprocessed.convertTo(preprocessed, CV_32F, 255.0, 0);
	cv::subtract(preprocessed, mean, preprocessed);
	preprocessed = preprocessed * inputscale;
}
void resizeImage(cv::Mat& frame, cv::Mat& dst, cv::Size dims)
{
	double max, min;
	int idx_min[2] = { 255, 255 }, idx_max[2] = { 255, 255 };
	cv::Mat tempImage;
	frame.copyTo(tempImage);
	tempImage = tempImage.reshape(1);
	cv::minMaxIdx(tempImage, &min, &max, idx_min, idx_max);

	cv::Mat imageStd;
	cv::Mat resizedStd;
	cv::subtract(frame, cv::Scalar::all(min), imageStd);

	imageStd = imageStd / (max - min);
	cv::resize(imageStd, resizedStd, dims);
	resizedStd = resizedStd * (max - min);
	cv::add(resizedStd, cv::Scalar::all(min), dst);
}
