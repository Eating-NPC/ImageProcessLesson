#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

Mat Load();
VideoCapture LCapture();
Mat grayImage(Mat& image);
Mat Hist(Mat& grayimage);
Mat equalizeHist(Mat& grayimage);
double generateGaussianNoise(double mu, double sigma);
Mat addSalt(int n);
Mat fixedThreshold(Mat& grayImage, int threshold);
Mat OTSU(Mat& grayImage);
Mat Kittle(Mat& grayimg);
void singleGauss();