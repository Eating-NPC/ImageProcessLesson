#include "Experiment.h"
#include "functions.h"


Experiment::Experiment(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
}

//图像采集
void Experiment::on_Load_clicked()
{
    imshow("Load", Load());
}

void Experiment::on_Capture_clicked()
{
	Mat frame;
    VideoCapture capture = LCapture();
	double rate = capture.get(CAP_PROP_FPS);
	int delay = 1000 / rate;
	while (capture.read(frame))
	{
		waitKey(delay);
		imshow("frame", frame);
	}
}

//图像预处理
void Experiment::on_grayImage_clicked()
{
	imshow("grayImage",grayImage(Load()));
}

void Experiment::on_Hist_clicked()
{
	imshow("Hist", Hist(grayImage(Load())));
}

void Experiment::on_equalizeHist_clicked()
{
	Mat src, dst;
	src = grayImage(Load());
	dst.create(src.size(), src.type());
	imshow("equalizeHist", equalizeHist(grayImage(Load())));
}

void Experiment::on_equalizeHistCV_clicked()
{
	Mat src,dst;
	src = grayImage(Load());
	dst.create(src.size(), src.type());
	equalizeHist(src, dst);
	imshow("hist", Hist(src));
	imshow("hist-equal", Hist(dst));
	imshow("src", src);
	imshow("equalizeCV", dst);
}
void Experiment::on_GradientSharp_clicked()
{
	Mat grad,result;
	Mat gray = grayImage(Load());
	grad.create(gray.size(), gray.type());
	result.create(gray.size(), gray.type());
	for (int i = 1; i < gray.rows - 1; i++)
	{
		for (int j = 1; j < gray.cols - 1; j++)
		{
			grad.at<uchar>(i, j) = saturate_cast<uchar>(fabs(gray.at<uchar>(i, j) - gray.at<uchar>(i + 1, j)) + fabs(gray.at<uchar>(i, j) - gray.at<uchar>(i, j + 1)));
			result.at<uchar>(i, j) = saturate_cast<uchar>(gray.at<uchar>(i, j) + (fabs(gray.at<uchar>(i, j) - gray.at<uchar>(i + 1, j)) + fabs(gray.at<uchar>(i, j) - gray.at<uchar>(i, j + 1))));
		}
	}
	imshow("gray", gray);
	imshow("result", result);
	imshow("grad", grad);
}

void Experiment::on_Laplace_clicked()
{
	Mat result1, laplaceimg;
	Mat gray;
	gray = grayImage(Load());
	imshow("gray", gray);
	result1.create(gray.size(), gray.type());
	laplaceimg.create(gray.size(), gray.type());
	for (int i = 1; i < gray.rows - 1; i++)
	{
		for (int j = 1; j < gray.cols - 1; j++)
		{
			result1.at<uchar>(i, j) = saturate_cast<uchar>(5 * gray.at<uchar>(i, j) - gray.at<uchar>(i + 1, j) - gray.at<uchar>(i - 1, j) - gray.at<uchar>(i, j + 1) - gray.at<uchar>(i, j - 1));
			laplaceimg.at<uchar>(i, j) = saturate_cast<uchar>(4 * gray.at<uchar>(i, j) - gray.at<uchar>(i + 1, j) - gray.at<uchar>(i - 1, j) - gray.at<uchar>(i, j + 1) - gray.at<uchar>(i, j - 1));
		}
	}
	imshow("result1", result1);
	imshow("laplaceimg", laplaceimg);
}

void Experiment::on_SaltAndPepper_clicked()
{
	imshow("src", Load());
	imshow("椒盐", addSalt(3000));
}

void Experiment::on_Monocular_clicked()
{

}

void Experiment::on_Binocular_clicked()
{

}

void Experiment::on_Stereo_clicked()
{

}

void Experiment::on_Robert_clicked()
{
	Mat  src, Roberts;
	src = grayImage(Load());
	imshow("gray", src);
	Roberts.create(src.size(), src.type());
	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			Roberts.at<uchar>(i, j) = saturate_cast<uchar>(fabs(src.at<uchar>(i, j) - src.at<uchar>(i + 1, j + 1)) + fabs(src.at<uchar>(i + 1, j) - src.at<uchar>(i, j + 1)));
		}
	}
	imshow("Roberts", Roberts);
}

void Experiment::on_Soble_clicked()
{
	Mat  src, Sobelx, Sobely, Sobel;
	src = grayImage(Load());
	imshow("image", Load());
	Sobel.create(src.size(), src.type());
	Sobelx.create(src.size(), src.type());
	Sobely.create(src.size(), src.type());
	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			Sobelx.at<uchar>(i, j) = saturate_cast<uchar>(fabs(src.at<uchar>(i + 1, j - 1) + 2 * src.at<uchar>(i + 1, j) + src.at<uchar>(i + 1, j + 1) - src.at<uchar>(i - 1, j - 1) - 2 * src.at<uchar>(i - 1, j) - src.at<uchar>(i - 1, j + 1)));
			Sobely.at<uchar>(i, j) = saturate_cast<uchar>(fabs(src.at<uchar>(i - 1, j + 1) + 2 * src.at<uchar>(i, j + 1) + src.at<uchar>(i + 1, j + 1) - src.at<uchar>(i - 1, j - 1) - 2 * src.at<uchar>(i, j - 1) - src.at<uchar>(i + 1, j - 1)));
			Sobel.at<uchar>(i, j) = saturate_cast<uchar>(fabs(Sobelx.at<uchar>(i, j)) + fabs(Sobely.at<uchar>(i, j)));
		}
	}
	imshow("Sobelx", Sobelx);
	imshow("Sobely", Sobely);
	imshow("Sobel", Sobel);
}

void Experiment::on_Canny_clicked()
{
	Mat gray, canny;
	int tl_value = 50;
	gray = grayImage(Load());
	imshow("gray", gray);
	canny.create(gray.size(), gray.type());
	Canny(gray, canny, tl_value, tl_value * 2, 3, false);
	imshow("canny", canny);
}

void Experiment::on_Gauss_clicked()
{
	Mat dst = Load();
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			if (dst.channels() == 1)
			{
				dst.at<uchar>(i, j) = saturate_cast<uchar>(dst.at<uchar>(i, j) + generateGaussianNoise(2, 0.8) * 32);
			}
			else
			{
				dst.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(dst.at<Vec3b>(i, j)[0] + generateGaussianNoise(2, 0.8) * 32);
				dst.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(dst.at<Vec3b>(i, j)[1] + generateGaussianNoise(2, 0.8) * 32);
				dst.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(dst.at<Vec3b>(i, j)[2] + generateGaussianNoise(2, 0.8) * 32);
			}
		}
	}
	imshow("Gauss", dst);
	imshow("src", Load());
}

void Experiment::on_LinearFilter_clicked()
{
	Mat out;
	Mat salting = addSalt(3000);
	boxFilter(salting, out, -1, Size(3, 3));
	Mat out_blur;
	blur(salting, out_blur, Size(3, 3));
	Mat out_gauss;
	GaussianBlur(salting, out_gauss, Size(3, 3), 0, 0);
	Mat out_median;
	medianBlur(salting, out_median, 5);
	imshow("salting", salting);
	imshow("boxFilter", out);
	imshow("blur", out_blur);
	imshow("GaussianBlur", out_gauss);
	imshow("medianBlur", out_median);
}

void Experiment::on_MorphologicalFilter_clicked()
{

}

void Experiment::on_EdgeFilter_clicked()
{

}

void Experiment::on_AffineTrans_clicked()
{
	Point2f srcTri[3], dstTri[3]; //二维坐标下的点，类型为浮点
	Mat rot_mat(2, 3, CV_32FC1); //单通道矩阵
	Mat warp_mat(2, 3, CV_32FC1);
	Mat src5, dst5;
	src5 = imread("C://Users//qw//Desktop//tutu.png", 1);
	dst5 = Mat::zeros(src5.rows, src5.cols, src5.type());
	//计算矩阵仿射变换
	srcTri[0] = Point2f(0, 0);
	srcTri[1] = Point2f(src5.cols - 1, 0); //缩小一个像素
	srcTri[2] = Point2f(0, src5.rows - 1);
	//改变目标图像大小
	dstTri[0] = Point2f(src5.cols * 0.0, src5.rows * 0.33);
	dstTri[1] = Point2f(src5.cols * 0.85, src5.rows * 0.25);
	dstTri[2] = Point2f(src5.cols * 0.15, src5.rows * 0.7);
	//由三对点计算仿射变换
	warp_mat = getAffineTransform(srcTri, dstTri);
	//对图像做仿射变换
	warpAffine(src5, dst5, warp_mat, src5.size());
	imshow("src5", src5);
	waitKey(0);
	imshow("dst5", dst5);
}

void Experiment::on_PerspectiveTrans_clicked()
{
	Point2f srcQuad[4], dstQuad[4];
	Mat warp_matrix(3, 3, CV_32FC1);
	Mat src6, dst6;
	src6 = imread("C://Users//qw//Desktop//tutu.png", 1);
	dst6 = Mat::zeros(src6.rows, src6.cols, src6.type());
	srcQuad[0] = Point2f(0, 0); //src top left
	srcQuad[1] = Point2f(src6.cols - 1, 0); //src top right
	srcQuad[2] = Point2f(0, src6.rows - 1); //src bottom left
	srcQuad[3] = Point2f(src6.cols - 1, src6.rows - 1); //src bot right
	dstQuad[0] = Point2f(src6.cols * 0.05, src6.rows * 0.33); //dst top left
	dstQuad[1] = Point2f(src6.cols * 0.9, src6.rows * 0.25); //dst top right
	dstQuad[2] = Point2f(src6.cols * 0.2, src6.rows * 0.7); //dst bottom left
	dstQuad[3] = Point2f(src6.cols * 0.8, src6.rows * 0.9); //dst bot right
	warp_matrix = getPerspectiveTransform(srcQuad, dstQuad);
	warpPerspective(src6, dst6, warp_matrix, src6.size());
	imshow("src6", src6);
	waitKey(0);
	imshow("dst6", dst6);
}

//目标检测
void Experiment::on_ThresholdSeg_clicked()
{
	imshow("src", grayImage(Load()));
	imshow("threshold",fixedThreshold(grayImage(Load()), 50));
}

void Experiment::on_OTSU_clicked()
{
	imshow("src", grayImage(Load()));
	imshow("OTSU", OTSU(grayImage(Load())));
}

void Experiment::on_Kittle_clicked()
{

}

void Experiment::on_ThresholdSegCV_clicked()
{

}

void Experiment::on_InterframeDiff_clicked()
{

}

void Experiment::on_MixedGauss_clicked()
{

}

void Experiment::on_MixedGaussVideo_clicked()
{

}

void Experiment::on_SIFT_clicked()
{

}

void Experiment::on_Brisk_clicked()
{

}

void Experiment::on_Brisk1_clicked()
{

}

void Experiment::on_ORB_clicked()
{

}

void Experiment::on_ORB1_clicked()
{

}

void Experiment::on_haar_clicked()
{

}

void Experiment::on_ssd_clicked()
{

}

void Experiment::on_yolo_clicked()
{

}

void Experiment::on_svm_clicked()
{

}

void Experiment::on_carcascade_clicked()
{

}

void Experiment::on_num_clicked()
{

}

void Experiment::on_tran_clicked()
{

}

void Experiment::on_camshift_clicked()
{

}

void Experiment::on_singleGauss_clicked()
{
	singleGauss();
}