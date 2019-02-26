#include<opencv2\opencv.hpp>
#include<iostream>
#include<fstream>
#include<sstream>
using namespace std;

#define CONFTHRESHOLD 0.1

string labelNames[5] = { "Y", "R", "B", "W", "N" };
vector<string> classes(labelNames, labelNames + 5);
vector<cv::Scalar> showColor = {cv::Scalar(0,255,255),cv::Scalar(0,0,255),
								cv::Scalar(255,0,0),cv::Scalar::all(255),
								cv::Scalar(0,255,0)};

vector<cv::String> getOutpusNames(const cv::dnn::Net& net);

void preprocess(cv::Mat& frame, cv::Mat& preprocessed, cv::Size caffeInputSize, cv::Scalar mean, float inputscale);
void postprocess(cv::Mat& frame, const vector<cv::Mat>& outs, cv::dnn::Net& net);
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
void resizeImage(cv::Mat& frame, cv::Mat& dst, cv::Size dims);

int main(){
	cv::dnn::Net peleeDetection;
	string prototxt = "../models/pelee/deploy_inference.prototxt";
	string weights = "../models/pelee/pelee_SSD_304x304_map78.caffemodel.caffemodel";
	string img = "../test_imgs/001.jpg";
	cv::Mat image = cv::imread(img);
	cv::Mat preprocessed;
	cv::Size caffeInputSize(304, 304);
	cv::Scalar mean(103.94, 116.78, 123.68);
	float inputscale = 0.017f;

	preprocess(image, preprocessed,caffeInputSize,mean,inputscale);
	
	vector<cv::Mat> detections;
	peleeDetection = cv::dnn::readNetFromCaffe(prototxt, weights);
	cv::Mat inputBlob = cv::dnn::blobFromImage(preprocessed, 1.0, caffeInputSize,cv::Scalar::all(0),false, false);
	peleeDetection.setInput(inputBlob);
	peleeDetection.forward(detections, getOutpusNames(peleeDetection));

	postprocess(image, detections, peleeDetection);
	cv::imshow("results", image);
	cv::waitKey();
	return 0;
}

vector<cv::String> getOutpusNames(const cv::dnn::Net& net)
{
	static vector<cv::String> names;
	if (names.empty())
	{
		vector<int> outLayers = net.getUnconnectedOutLayers();
		vector<cv::String> layersNames = net.getLayerNames();
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

void preprocess(cv::Mat& frame, cv::Mat& preprocessed, cv::Size caffeInputSize, cv::Scalar mean, float inputscale)
{

	frame.convertTo(preprocessed, CV_32F, 1.0 / 255.0, 0.0);
	resizeImage(preprocessed, preprocessed, caffeInputSize);
	preprocessed.convertTo(preprocessed, CV_32F, 255.0, 0);
	cv::subtract(preprocessed, mean, preprocessed);
	preprocessed = preprocessed * inputscale;
}

void postprocess(cv::Mat& frame, const vector<cv::Mat>& outs, cv::dnn::Net& net)
{
	static vector<int> outLayers = net.getUnconnectedOutLayers();
	static string outLayerType = net.getLayer(outLayers[0])->type;

	vector<int> classIds;
	vector<float> confidences;
	vector<cv::Rect> boxes;

	CV_Assert(outs.size() == 1);
	float* data = (float*)outs[0].data;
	for (size_t i = 0; i < outs[0].total(); i += 7)
	{
		float confidence = data[i + 2];
		if (confidence > CONFTHRESHOLD)
		{
			int left = (int)(data[i + 3] * frame.cols);
			int top = (int)(data[i + 4] * frame.rows);
			int right = (int)(data[i + 5] * frame.cols);
			int bottom = (int)(data[i + 6] * frame.rows);
			int width = right - left + 1;
			int height = bottom - top + 1;
			int classId = (int)(data[i + 1]) - 1;

			drawPred(classId, confidence, left, top, right, bottom, frame);
			classIds.push_back(classId);
			boxes.push_back(cv::Rect(left, top, width, height));
			confidences.push_back(confidence);
		}
	}

}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
	
	cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), showColor[classId]);
	string label = classes[classId] + ":" + cv::format("%.2f",conf);
	int baseLine;
	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	cv::rectangle(frame, cv::Point(left, top - labelSize.height),
		cv::Point(left + labelSize.width, top + baseLine), showColor[classId], cv::FILLED);
	cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
}

void resizeImage(cv::Mat& frame, cv::Mat& dst, cv::Size dims)
{
	double max, min;
	int idx_min[2] = { 255, 255 }, idx_max[2] = { 255, 255 };
	cv::Mat tempImage;
	frame.copyTo(tempImage);
	tempImage = tempImage.reshape(1);
	cv::minMaxIdx(tempImage, &min, &max, idx_min,idx_max);

	cv::Mat imageStd;
	cv::Mat resizedStd;
	cv::subtract(frame, cv::Scalar::all(min), imageStd);

	imageStd = imageStd / (max - min);
	cv::resize(imageStd, resizedStd, dims);
	resizedStd = resizedStd * (max - min);
	cv::add(resizedStd, cv::Scalar::all(min), dst);

}