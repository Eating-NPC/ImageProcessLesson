#include "functions.h"
#include <stdio.h>

Mat Load()
{
	Mat image = imread("C:\\Users\\book\\Desktop\\lena.jpg");
	return image;
}

VideoCapture LCapture()
{
	VideoCapture capture;
	capture.open("C:\\Users\\book\\Desktop\\cat.avi");
	if (!capture.isOpened())
	{
		std::cout << "video not open." << std::endl;
	}
	return capture;
}

Mat grayImage(Mat& image)
{
	Mat grayimage;
	grayimage.create(image.size(), CV_8UC1);
	for (int i = 0; i < grayimage.rows; i++) {
		for (int j = 0; j < grayimage.cols; j++) {
			grayimage.at<uchar>(i, j) = image.at<Vec3b>(i, j)[0] * 0.11 + image.at<Vec3b>(i, j)[1] * 0.59 + image.at<Vec3b>(i, j)[2] * 0.3;
		}
	}
	return grayimage;
}

Mat Hist(Mat& grayimage)
{
	Mat hist = Mat::zeros(256,256,CV_8UC1);

	int count[256] = { 0 };
	int num = 0;
	for (int i = 0; i < grayimage.rows; i++) {
		for (int j = 0; j < grayimage.cols; j++) {
			count[grayimage.at<uchar>(i, j)]++;
		}
	}
	int size = grayimage.cols * grayimage.rows;

	for (int i = 0; i < 256; i++) {
		num = 255 - 10000* ((double)count[i] / size);
		line(hist, Point(i, num), Point(i, 255), Scalar(255,0,0), 1);
	}
	return hist;
}

Mat equalizeHist(Mat& grayimage)
{
	Mat hist = Hist(grayimage);
	imshow("hist", hist);
	Mat equalizehist = Mat::zeros(256, 256, CV_8UC1);
	int size = grayimage.rows * grayimage.cols;
	int count[256] = { 0 };
	int p[256] = { 0 };
	for (int i = 0; i < grayimage.rows; i++) {
		for (int j = 0; j < grayimage.cols; j++) {
			count[grayimage.at<uchar>(i, j)]++;
		}
	}
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < i; j++) {
			p[i] += count[j];
		}
	}
	
	return equalizehist;
	
}

double generateGaussianNoise(double mu, double sigma)
{
	static double V1, V2, S;
	static int phase = 0;
	double X;
	double U1, U2;
	if (phase == 0) {
		do {
			U1 = (double)rand() / RAND_MAX;
			U2 = (double)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while (S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	}
	else {
		X = V2 * sqrt(-2 * log(S) / S);
	}
	phase = 1 - phase;
	return mu + sigma * X;
}

Mat addSalt(int n)
{
	Mat dst = Load();
	for (int k = 0; k < n; k++)
	{
		//随机取值行列
		int i = rand() % dst.rows;
		int j = rand() % dst.cols;
		//图像通道判定
		if (dst.channels() == 1)
		{
			dst.at<uchar>(i, j) = 255;
		}
		else
		{
			dst.at<Vec3b>(i, j)[0] = 255;
			dst.at<Vec3b>(i, j)[1] = 255;
			dst.at<Vec3b>(i, j)[2] = 255;
		}
	}
	return dst;
}

Mat fixedThreshold(Mat& grayImage, int threshold)
{
	Mat dst;
	dst.create(grayImage.size(), grayImage.type());
	for (int i = 0; i < grayImage.rows; i++)
	{
		for (int j = 0; j < grayImage.cols; j++)
		{
			if (grayImage.at<uchar>(i, j) < threshold)
			{
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	return dst;
}

Mat OTSU(Mat& grayImage)
{
	int nCols = grayImage.cols;
	int nRows = grayImage.rows;
	int threshold = 0;
	//init the parameters
	int nSumPix[256];
	float nProDis[256];
	for (int i = 0; i < 256; i++)
	{
		nSumPix[i] = 0;
		nProDis[i] = 0;
	}

	//统计灰度集中每个像素在整幅图像中的个数
	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{
			nSumPix[(int)grayImage.at<uchar>(i, j)]++;
		}
	}

	//计算每个灰度级占图像中的概率分布
	for (int i = 0; i < 256; i++)
	{
		nProDis[i] = (float)nSumPix[i] / (nCols * nRows);
	}

	//遍历灰度级【0，255】，计算出最大类间方差下的阈值

	float w0, w1, u0_temp, u1_temp, u0, u1, delta_temp;
	double delta_max = 0.0;
	for (int i = 0; i < 256; i++)
	{
		//初始化相关参数
		w0 = w1 = u0 = u1 = u0_temp = u1_temp = delta_temp = 0;
		for (int j = 0; j < 256; j++)
		{
			//背景部分
			if (j <= i)
			{
				w0 += nProDis[j];
				u0_temp += j * nProDis[j];
			}
			//前景部分
			else
			{
				w1 += nProDis[j];
				u1_temp += j * nProDis[j];
			}
		}
		//计算两个分类的平均灰度
		u0 = u0_temp / w0;
		u1 = u1_temp / w1;
		//依次找到最大类间方差下的阈值
		delta_temp = (float)(w0 * w1 * pow((u0 - u1), 2)); //前景与背景之间的方差(类间方差)
		if (delta_temp > delta_max)
		{
			delta_max = delta_temp;
			threshold = i;
		}
	}
	return fixedThreshold(grayImage, threshold);
}

void singleGauss()
{
	double alpha = 0.05;    //背景建模alpha值
	double std_init = 20;    //初始标准差
	double var_init = std_init * std_init;    //初始方差    
	double lamda = 2.5 * 1.2;    //背景更新参数

	Mat frame, frame_u, frame_d, frame_var, frame_std;
	VideoCapture capture;
	capture.open("C:\\Users\\book\\Desktop\\cat.avi");
	if (!capture.isOpened())
	{
		std::cout << "video not open." << std::endl;
	}
	double rate = capture.get(CAP_PROP_FPS);
	int delay = 1000 / rate;
	while (frame.empty()) {
		capture.read(frame);
		waitKey(delay);
	}
	frame_u.create(frame.size(), frame.type());
	frame_d.create(frame.size(), frame.type());
	frame_var.create(frame.size(), frame.type());
	frame_std.create(frame.size(), frame.type());
	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++)
		{
			frame_u.at<Vec3b>(i, j)[0] = frame.at<Vec3b>(i, j)[0];
			frame_u.at<Vec3b>(i, j)[1] = frame.at<Vec3b>(i, j)[1];
			frame_u.at<Vec3b>(i, j)[2] = frame.at<Vec3b>(i, j)[2];
			frame_d.at<Vec3b>(i, j)[0] = 0;
			frame_d.at<Vec3b>(i, j)[1] = 0;
			frame_d.at<Vec3b>(i, j)[2] = 0;
			frame_var.at<Vec3b>(i, j)[0] = var_init;
			frame_var.at<Vec3b>(i, j)[1] = var_init;
			frame_var.at<Vec3b>(i, j)[2] = var_init;
			frame_std.at<Vec3b>(i, j)[0] = std_init;
			frame_std.at<Vec3b>(i, j)[1] = std_init;
			frame_std.at<Vec3b>(i, j)[2] = std_init;
		}
	}
	while (capture.read(frame))
	{
		for (int i = 0; i < frame.rows; i++)
		{
			for (int j = 0; j < frame.cols; j++)
			{
				if (abs(frame.at<Vec3b>(i, j)[0] - frame_u.at<Vec3b>(i, j)[0]) < lamda * std_init &&
					abs(frame.at<Vec3b>(i, j)[1] - frame_u.at<Vec3b>(i, j)[1]) < lamda * std_init &&
					abs(frame.at<Vec3b>(i, j)[2] - frame_u.at<Vec3b>(i, j)[2]) < lamda * std_init)
				{
					frame_u.at<Vec3b>(i, j)[0] = (1 - alpha) * frame_u.at<Vec3b>(i, j)[0] + alpha * frame.at<Vec3b>(i, j)[0];
					frame_u.at<Vec3b>(i, j)[1] = (1 - alpha) * frame_u.at<Vec3b>(i, j)[1] + alpha * frame.at<Vec3b>(i, j)[1];
					frame_u.at<Vec3b>(i, j)[2] = (1 - alpha) * frame_u.at<Vec3b>(i, j)[2] + alpha * frame.at<Vec3b>(i, j)[2];
					frame_var.at<Vec3b>(i, j)[0] = (1 - alpha) * frame_var.at<Vec3b>(i, j)[0] + alpha * (frame.at<Vec3b>(i, j)[0] - frame_u.at<Vec3b>(i, j)[0]) * (frame.at<Vec3b>(i, j)[0] - frame_u.at<Vec3b>(i, j)[0]);
					frame_var.at<Vec3b>(i, j)[1] = (1 - alpha) * frame_var.at<Vec3b>(i, j)[1] + alpha * (frame.at<Vec3b>(i, j)[1] - frame_u.at<Vec3b>(i, j)[1]) * (frame.at<Vec3b>(i, j)[1] - frame_u.at<Vec3b>(i, j)[1]);
					frame_var.at<Vec3b>(i, j)[2] = (1 - alpha) * frame_var.at<Vec3b>(i, j)[2] + alpha * (frame.at<Vec3b>(i, j)[2] - frame_u.at<Vec3b>(i, j)[2]) * (frame.at<Vec3b>(i, j)[2] - frame_u.at<Vec3b>(i, j)[2]);
					frame_std.at<Vec3b>(i, j)[0] = sqrt(frame_var.at<Vec3b>(i, j)[0] * 1.0);
					frame_std.at<Vec3b>(i, j)[1] = sqrt(frame_var.at<Vec3b>(i, j)[1] * 1.0);
					frame_std.at<Vec3b>(i, j)[2] = sqrt(frame_var.at<Vec3b>(i, j)[2] * 1.0);
				}
				else
				{
					frame_d.at<Vec3b>(i, j)[0] = frame.at<Vec3b>(i, j)[0] - frame_u.at<Vec3b>(i, j)[0];
					frame_d.at<Vec3b>(i, j)[1] = frame.at<Vec3b>(i, j)[1] - frame_u.at<Vec3b>(i, j)[1];
					frame_d.at<Vec3b>(i, j)[2] = frame.at<Vec3b>(i, j)[2] - frame_u.at<Vec3b>(i, j)[2];
				}
			}
		}


		imshow("orgin", frame);

		imshow("background", frame_u);
		imshow("foreground", frame_d);
		for (int i = 0; i < frame.rows; i++)
		{
			for (int j = 0; j < frame.cols; j++)
			{
				frame_d.at<Vec3b>(i, j)[0] = 0;
				frame_d.at<Vec3b>(i, j)[1] = 0;
				frame_d.at<Vec3b>(i, j)[2] = 0;
			}
		}
		waitKey(delay);
	}
}