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
	Mat imaout;
	Mat grayImg = grayImage(Load());
	int minval = 0;
	vector<int>h0(8);//存储各区域均值
	vector<int>h1(8);//存储（像素值-均值）的平方
	vector<int>h2(8);//存储各区域中心值
	imaout.create(grayImg.rows - 2, grayImg.cols - 2, CV_8UC1);
	for (int i = 1; i < grayImg.rows - 1; i++)
	{
		for (int j = 1; j < grayImg.cols - 1; j++)
		{
			//左上
			if (i != 0 && j != 0)
			{
				h0[0] = (grayImg.at<uchar>(i - 1, j - 1) + grayImg.at<uchar>(i - 1, j) + grayImg.at<uchar>(i, j - 1) + grayImg.at<uchar>(i, j)) / 4;
				h2[0] = grayImg.at<uchar>(i - 1, j - 1);
			}
			else
			{
				h0[0] = 255;
			}
			//上
			if (i != 0 && j != 0 && j != grayImg.cols - 1)
			{
				h0[1] = (grayImg.at<uchar>(i - 1, j - 1) + grayImg.at<uchar>(i - 1, j) + grayImg.at<uchar>(i - 1, j + 1) + grayImg.at<uchar>(i, j - 1) + grayImg.at<uchar>(i, j) + grayImg.at<uchar>(i, j + 1)) / 6;
				h2[1] = grayImg.at<uchar>(i - 1, j);
			}
			else
			{
				h0[1] = 255;
			}
			//右上
			if (i != 0 && j != grayImg.cols - 1)
			{
				h0[2] = (grayImg.at<uchar>(i - 1, j + 1) + grayImg.at<uchar>(i - 1, j) + grayImg.at<uchar>(i, j + 1) + grayImg.at<uchar>(i, j)) / 4;
				h2[2] = grayImg.at<uchar>(i - 1, j + 1);
			}
			else
			{
				h0[2] = 255;
			}
			//左
			if (i != 0 && i != grayImg.rows - 1 && j != 0)
			{
				h0[3] = (grayImg.at<uchar>(i - 1, j - 1) + grayImg.at<uchar>(i - 1, j) + grayImg.at<uchar>(i, j - 1) + grayImg.at<uchar>(i, j) + grayImg.at<uchar>(i + 1, j - 1) + grayImg.at<uchar>(i + 1, j)) / 6;
				h2[3] = grayImg.at<uchar>(i, j - 1);
			}
			else
			{
				h0[3] = 255;
			}
			//右
			if (i != 0 && i != grayImg.rows - 1 && j != grayImg.cols - 1)
			{
				h0[4] = (grayImg.at<uchar>(i - 1, j + 1) + grayImg.at<uchar>(i - 1, j) + grayImg.at<uchar>(i, j + 1) + grayImg.at<uchar>(i, j) + grayImg.at<uchar>(i + 1, j + 1) + grayImg.at<uchar>(i + 1, j)) / 6;
				h2[4] = grayImg.at<uchar>(i, j + 1);
			}
			else
			{
				h0[4] = 255;
			}
			//左下
			if (i != grayImg.rows - 1 && j != 0)
			{
				h0[5] = (grayImg.at<uchar>(i, j - 1) + grayImg.at<uchar>(i, j) + grayImg.at<uchar>(i + 1, j - 1) + grayImg.at<uchar>(i + 1, j)) / 4;
				h2[5] = grayImg.at<uchar>(i + 1, j - 1);
			}
			else
			{
				h0[5] = 255;
			}
			//下
			if (i != grayImg.rows && j != 0 && j != grayImg.cols - 1)
			{
				h0[6] = (grayImg.at<uchar>(i + 1, j - 1) + grayImg.at<uchar>(i + 1, j) + grayImg.at<uchar>(i + 1, j + 1) + grayImg.at<uchar>(i, j - 1) + grayImg.at<uchar>(i, j) + grayImg.at<uchar>(i, j + 1)) / 6;
				h2[6] = grayImg.at<uchar>(i + 1, j);
			}
			else
			{
				h0[6] = 255;
			}
			//右下
			if (i != grayImg.rows - 1 && j != grayImg.cols - 1)
			{
				h0[7] = (grayImg.at<uchar>(i, j + 1) + grayImg.at<uchar>(i, j) + grayImg.at<uchar>(i + 1, j + 1) + grayImg.at<uchar>(i + 1, j)) / 4;
				h2[7] = grayImg.at<uchar>(i + 1, j + 1);
			}
			else
			{
				h0[7] = 255;
			}
			for (int z = 0; z < 8; z++)
			{
				h1[z] = sqrt(grayImg.at<uchar>(i, j) - h0[z]);
			}
			minval = min_element(h1.begin(), h1.end()) - h1.begin();

			cou = saturate_cast<uchar>(h2[minval]);
			imaout.at<uchar>(i - 1, j - 1) = cou;
		}
	}
	imshow("SWF", imaout);
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
	imshow("src", grayImage(Load()));
	imshow("Kittle", Kittle(grayImage(Load())));
}

void Experiment::on_ThresholdSegCV_clicked()
{
	Mat dst;
	threshold(grayImage(Load()), dst, 100, 255, THRESH_OTSU);
	imshow("dst", dst);
}

void Experiment::on_InterframeDiff_clicked()
{

}

void Experiment::on_MixedGauss_clicked()
{

}

void Experiment::on_MixedGaussVideo_clicked()
{
	Mat greyimg;
	Mat foreground, foreground2;
	Ptr<BackgroundSubtractorKNN> ptrKNN = createBackgroundSubtractorKNN(100, 400, true);
	Ptr<BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2(100, 25, true);
	namedWindow("Extracted Foreground");
	VideoCapture pCapture;
	Mat pframe;
	pCapture = VideoCapture("E:/Code/C++/Open_cv/Example/pets2001.avi");

	while (1)
	{
		pCapture >> pframe;
		if (pframe.data == NULL)
			return;
		cvtColor(pframe, greyimg, CV_BGR2GRAY);
		long long t = getTickCount();
		ptrKNN->apply(pframe, foreground, 0.01);
		long long t1 = getTickCount();
		mog2->apply(greyimg, foreground2, -1);
		long long t2 = getTickCount();
		_cprintf("t1 = %f t2 = %f\n", (t1 - t) / getTickFrequency(), (t2 - t1) / getTickFrequency());
		imshow("Extracted Foreground", foreground);
		imshow("Extracted Foreground2", foreground2);
		imshow("video", pframe);
		waitKey(10);
	}
}

void Experiment::on_SIFT_clicked()
{
	Mat src1 = imread("E:/Code/C++/Open_cv/Example/1.1.jpg", 1);
	Mat src2 = imread("E:/Code/C++/Open_cv/Example/1.2.jpg", 1);
	imshow("src1", src1);
	imshow("src2", src2);

	if (!src1.data || !src2.data)
	{
		_cprintf(" --(!) Error reading images \n");
		return;
	}

	//sift feature detect  
	Ptr<SIFT> siftdetector = SIFT::create();
	vector<KeyPoint> kp1, kp2;

	siftdetector->detect(src1, kp1);
	siftdetector->detect(src2, kp2);
	Mat des1, des2;//descriptor  
	siftdetector->compute(src1, kp1, des1);
	siftdetector->compute(src2, kp2, des2);
	Mat res1, res2;

	drawKeypoints(src1, kp1, res1);//在内存中画出特征点  
	drawKeypoints(src2, kp2, res2);

	_cprintf("size of description of Img1: %d\n", kp1.size());
	_cprintf("size of description of Img2: %d\n", kp2.size());

	Mat transimg1, transimg2;
	transimg1 = res1.clone();
	transimg2 = res2.clone();

	char str1[20], str2[20];
	sprintf_s(str1, "%d", kp1.size());
	sprintf_s(str2, "%d", kp2.size());

	const char* str = str1;
	putText(transimg1, str1, Point(280, 230), 0, 1.0, Scalar(255, 0, 0), 2);//在图片中输出字符   

	str = str2;
	putText(transimg2, str2, Point(280, 230), 0, 1.0, Scalar(255, 0, 0), 2);//在图片中输出字符   

																			//imshow("Description 1",res1);  
	imshow("descriptor1", transimg1);
	imshow("descriptor2", transimg2);

	BFMatcher matcher(NORM_L2, true);
	vector<DMatch> matches;
	matcher.match(des1, des2, matches);
	Mat img_match;
	drawMatches(src1, kp1, src2, kp2, matches, img_match);//,Scalar::all(-1),Scalar::all(-1),vector<char>(),drawmode);  
	_cprintf("number of matched points: %d\n", matches.size());
	imshow("matches", img_match);
}

void Experiment::on_Brisk_clicked()
{
	Mat src1 = imread("E:/Code/C++/Open_cv/Example/1.1.jpg");
	Mat src2 = imread("E:/Code/C++/Open_cv/Example/1.2.jpg");
	// feature detect
	Ptr<BRISK> detector = BRISK::create();
	vector<KeyPoint> kp1, kp2;
	double start = GetTickCount();
	detector->detect(src1, kp1);
	detector->detect(src2, kp2);
	//cv::BRISK extractor;  

	Mat des1, des2;//descriptor  
	detector->compute(src1, kp1, des1);
	detector->compute(src2, kp2, des2);

	Mat res1, res2;
	int drawmode = DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
	drawKeypoints(src1, kp1, res1, Scalar::all(-1), drawmode);//画出特征点  
	drawKeypoints(src2, kp2, res2, Scalar::all(-1), drawmode);
	_cprintf("size of description of Img1: %d\n", kp1.size());
	_cprintf("size of description of Img2: %d\n", kp2.size());

	BFMatcher matcher(NORM_HAMMING);
	vector<DMatch> matches;
	matcher.match(des1, des2, matches);
	double end = GetTickCount();
	_cprintf("run time: %f ms\n", (end - start));
	Mat img_match;
	drawMatches(src1, kp1, src2, kp2, matches, img_match);
	_cprintf("number of matched points: %d\n", matches.size());
	imshow("matches", img_match);
}

void Experiment::on_Brisk1_clicked()
{

}

void Experiment::on_ORB_clicked()
{
	Mat img1 = imread("E:/Code/C++/Open_cv/Example/1.1.jpg");
	Mat img2 = imread("E:/Code/C++/Open_cv/Example/1.2.jpg");
	// 1 初始化特征点和描述子,ORB
	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	Ptr<ORB> orb = ORB::create();
	// 2 提取 Oriented FAST 特征点
	orb->detect(img1, keypoints1);
	orb->detect(img2, keypoints2);
	// 3 根据角点位置计算 BRIEF 描述子
	orb->compute(img1, keypoints1, descriptors1);
	orb->compute(img2, keypoints2, descriptors2);
	// 4 对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
	vector<DMatch> matches;
	BFMatcher bfmatcher(NORM_HAMMING, true);
	bfmatcher.match(descriptors1, descriptors2, matches);
	// 5 绘制匹配结果
	Mat img_match;
	drawMatches(img1, keypoints1, img2, keypoints2, matches, img_match);
	imshow("所有匹配点对", img_match);
}

void Experiment::on_ORB1_clicked()
{
	Mat obj = imread("E:/Code/C++/Open_cv/Example/1.1.jpg");   //载入目标图像
	Mat scene = imread("E:/Code/C++/Open_cv/Example/1.2.jpg"); //载入场景图像
	if (obj.empty() || scene.empty())
	{
		cout << "Can't open the picture!\n";
		return;
	}
	vector<KeyPoint> obj_keypoints, scene_keypoints;
	Mat obj_descriptors, scene_descriptors;
	Ptr<ORB> detector = ORB::create();

	detector->detect(obj, obj_keypoints);
	detector->detect(scene, scene_keypoints);
	detector->compute(obj, obj_keypoints, obj_descriptors);
	detector->compute(scene, scene_keypoints, scene_descriptors);

	BFMatcher matcher(NORM_HAMMING, true); //汉明距离做为相似度度量
	vector<DMatch> matches;
	matcher.match(obj_descriptors, scene_descriptors, matches);
	Mat match_img;
	drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, match_img);
	imshow("滤除误匹配前", match_img);

	//保存匹配对序号
	vector<int> queryIdxs(matches.size()), trainIdxs(matches.size());
	for (size_t i = 0; i < matches.size(); i++)
	{
		queryIdxs[i] = matches[i].queryIdx;
		trainIdxs[i] = matches[i].trainIdx;
	}

	Mat H12;   //变换矩阵

	vector<Point2f> points1;
	KeyPoint::convert(obj_keypoints, points1, queryIdxs);
	vector<Point2f> points2;
	KeyPoint::convert(scene_keypoints, points2, trainIdxs);
	int ransacReprojThreshold = 5;  //拒绝阈值


	H12 = findHomography(Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold);
	vector<char> matchesMask(matches.size(), 0);
	Mat points1t;
	perspectiveTransform(Mat(points1), points1t, H12);
	for (size_t i1 = 0; i1 < points1.size(); i1++)  //保存‘内点’
	{
		if (norm(points2[i1] - points1t.at<Point2f>((int)i1, 0)) <= ransacReprojThreshold) //给内点做标记
		{
			matchesMask[i1] = 1;
		}
	}
	Mat match_img2;   //滤除‘外点’后
	drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, match_img2, Scalar(0, 0, 255), Scalar::all(-1), matchesMask);

	//画出目标位置
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0, 0); obj_corners[1] = Point(obj.cols, 0);
	obj_corners[2] = Point(obj.cols, obj.rows); obj_corners[3] = Point(0, obj.rows);
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H12);
	//line( match_img2, scene_corners[0] + Point2f(static_cast<float>(obj.cols), 0),scene_corners[1] + Point2f(static_cast<float>(obj.cols), 0),Scalar(0,0,255),2);
	//line( match_img2, scene_corners[1] + Point2f(static_cast<float>(obj.cols), 0),scene_corners[2] + Point2f(static_cast<float>(obj.cols), 0),Scalar(0,0,255),2);
	//line( match_img2, scene_corners[2] + Point2f(static_cast<float>(obj.cols), 0),scene_corners[3] + Point2f(static_cast<float>(obj.cols), 0),Scalar(0,0,255),2);
	//line( match_img2, scene_corners[3] + Point2f(static_cast<float>(obj.cols), 0),scene_corners[0] + Point2f(static_cast<float>(obj.cols), 0),Scalar(0,0,255),2);
	line(match_img2, Point2f((scene_corners[0].x + static_cast<float>(obj.cols)), (scene_corners[0].y)), Point2f((scene_corners[1].x + static_cast<float>(obj.cols)), (scene_corners[1].y)), Scalar(0, 0, 255), 2);
	line(match_img2, Point2f((scene_corners[1].x + static_cast<float>(obj.cols)), (scene_corners[1].y)), Point2f((scene_corners[2].x + static_cast<float>(obj.cols)), (scene_corners[2].y)), Scalar(0, 0, 255), 2);
	line(match_img2, Point2f((scene_corners[2].x + static_cast<float>(obj.cols)), (scene_corners[2].y)), Point2f((scene_corners[3].x + static_cast<float>(obj.cols)), (scene_corners[3].y)), Scalar(0, 0, 255), 2);
	line(match_img2, Point2f((scene_corners[3].x + static_cast<float>(obj.cols)), (scene_corners[3].y)), Point2f((scene_corners[0].x + static_cast<float>(obj.cols)), (scene_corners[0].y)), Scalar(0, 0, 255), 2);

	float A_th;
	A_th = atan(abs((scene_corners[3].y - scene_corners[0].y) / (scene_corners[3].x - scene_corners[0].x)));
	A_th = 90 - 180 * A_th / 3.14;
	_cprintf("angle=%f\n", A_th);

	imshow("滤除误匹配后", match_img2);

	//line( scene, scene_corners[0],scene_corners[1],Scalar(0,0,255),2);
	//line( scene, scene_corners[1],scene_corners[2],Scalar(0,0,255),2);
	//line( scene, scene_corners[2],scene_corners[3],Scalar(0,0,255),2);
	//line( scene, scene_corners[3],scene_corners[0],Scalar(0,0,255),2);

	imshow("场景图像", scene);

	Mat rotimage;
	Mat rotate = getRotationMatrix2D(Point(scene.cols / 2, scene.rows / 2), A_th, 1);
	warpAffine(scene, rotimage, rotate, scene.size());
	imshow("rotimage", rotimage);


	//方法三 透视变换  
	Point2f src_point[4];
	Point2f dst_point[4];
	src_point[0].x = scene_corners[0].x;
	src_point[0].y = scene_corners[0].y;
	src_point[1].x = scene_corners[1].x;
	src_point[1].y = scene_corners[1].y;
	src_point[2].x = scene_corners[2].x;
	src_point[2].y = scene_corners[2].y;
	src_point[3].x = scene_corners[3].x;
	src_point[3].y = scene_corners[3].y;


	dst_point[0].x = 0;
	dst_point[0].y = 0;
	dst_point[1].x = obj.cols;
	dst_point[1].y = 0;
	dst_point[2].x = obj.cols;
	dst_point[2].y = obj.rows;
	dst_point[3].x = 0;
	dst_point[3].y = obj.rows;

	Mat newM(3, 3, CV_32FC1);
	newM = getPerspectiveTransform(src_point, dst_point);

	Mat dst = scene.clone();

	warpPerspective(scene, dst, newM, obj.size());

	imshow("obj", obj);
	imshow("dst", dst);

	Mat resultimg = dst.clone();

	absdiff(obj, dst, resultimg);//当前帧跟前面帧相减

	imshow("result", resultimg);

	imshow("dst", dst);
	imshow("src", obj);
}

void Experiment::on_haar_clicked()
{
	CascadeClassifier faceCascade;
	faceCascade.load("E://Code//C++//Open_cv//opencv-4.1.0//sources//data//haarcascades//haarcascade_frontalface_alt2.xml");//加载分类器
	VideoCapture capture;
	capture.open(0);// 打开摄像头
	//      capture.open("video.avi");    // 打开视频
	if (!capture.isOpened())
	{
		_cprintf("open camera failed. \n");
		return;
	}
	Mat img, imgGray;
	vector<Rect> faces;
	while (1)
	{
		capture >> img;// 读取图像至img
		if (img.empty())continue;
		if (img.channels() == 3)
			cvtColor(img, imgGray, CV_RGB2GRAY);
		else
		{
			imgGray = img;
		}
		double start = GetTickCount();
		faceCascade.detectMultiScale(imgGray, faces, 1.2, 6, 0, Size(0, 0));// 检测人脸
		double end = GetTickCount();
		_cprintf("run time: %f ms\n", (end - start));
		if (faces.size() > 0)
		{
			for (int i = 0; i < faces.size(); i++)
			{
				rectangle(img, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 1, 8);
			}
		}
		imshow("CamerFace", img); // 显示
		if (waitKey(1) != -1)
			break;// delay ms 等待按键退出
	}

}

//face-ssd
void CProjectDlg::OnBnClickedButton38()
{
	size_t inWidth = 300;
	size_t inHeight = 300;
	double inScaleFactor = 1.0;
	Scalar meanVal(104.0, 177.0, 123.0);
	float min_confidence = 0.5;
	String modelConfiguration = "E://Code//C++//Open_cv//opencv-4.1.0//samples//dnn//face_detector//deploy.prototxt";
	String modelBinary = "E://Code//C++//Open_cv//opencv-4.1.0//samples//dnn//face_detector//res10_300x300_ssd_iter_140000.caffemodel";
	//! [Initialize network]
	dnn::Net net = dnn::readNetFromCaffe(modelConfiguration, modelBinary);
	//! [Initialize network]
	if (net.empty())
	{
		cerr << "Can't load network by using the following files: " << endl;
		cerr << "prototxt:   " << modelConfiguration << endl;
		cerr << "caffemodel: " << modelBinary << endl;
		cerr << "Models are available here:" << endl;
		cerr << "<OPENCV_SRC_DIR>/samples/dnn/face_detector" << endl;
		cerr << "or here:" << endl;
		cerr << "https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector" << endl;
		exit(-1);
	}
	VideoCapture cap(0);//must be -1
	if (!cap.isOpened())
	{
		_cprintf("Couldn't open camera : \n");
		return;
	}
	for (;;)//while(1)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera/video or read image
		if (frame.empty())break;
		if (frame.channels() == 4)cvtColor(frame, frame, COLOR_BGRA2BGR);
		//! [Prepare blob]
		Mat inputBlob = dnn::blobFromImage(frame, inScaleFactor,
			Size(inWidth, inHeight),
			meanVal, false, false); //Convert Mat to batch of images
									//! [Set input blob]
		net.setInput(inputBlob, "data"); //set the network input
										 //! [Make forward pass]
		Mat detection = net.forward("detection_out"); //compute output

		vector<double> layersTimings;
		double freq = getTickFrequency() / 1000;
		double time = net.getPerfProfile(layersTimings) / freq;
		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
		ostringstream ss;
		ss << "FPS: " << 1000 / time << " ; time: " << time << " ms";
		putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(255, 0, 255));

		float confidenceThreshold = min_confidence;
		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);
			if (confidence > confidenceThreshold)
			{
				auto xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
				auto yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
				auto xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
				auto yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
				Rect object(xLeftBottom, yLeftBottom,
					(xRightTop - xLeftBottom),
					(yRightTop - yLeftBottom));
				rectangle(frame, object, Scalar(0, 255, 0));
				ss.str("");
				ss << confidence;
				String conf(ss.str());
				String label = "Face: " + conf;
				int baseLine = 0;
				Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
				rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
					Size(labelSize.width, labelSize.height + baseLine)),
					Scalar(255, 255, 255), CV_FILLED);
				putText(frame, label, Point(xLeftBottom, yLeftBottom),
					FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
			}
		}
		imshow("detections", frame);
		if (waitKey(1) != -1) break;
	}
}

void Experiment::on_ssd_clicked()
{

}

void Experiment::on_yolo_clicked()
{

}

void Experiment::on_svm_clicked()
{
	int iWidth = 512, iheight = 512;
	Mat matImg = Mat::zeros(iheight, iWidth, CV_8UC3);//三色通道
															  //1.获取样本
	int labels[5] = { 1.0, -1.0, -1.0, -1.0,1.0 }; //样本数据  
	Mat labelsMat(5, 1, CV_32SC1, labels);     //样本标签  
	float trainingData[5][2] = { { 501, 300 },{ 255, 10 },{ 501, 255 },{ 10, 501 },{ 450,500 } }; //Mat结构特征数据  
	Mat trainingDataMat(5, 2, CV_32FC1, trainingData);   //Mat结构标签  
														 //2.设置SVM参数
	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);//可以处理非线性分割的问题
	svm->setKernel(ml::SVM::POLY);//径向基函数SVM::LINEAR
										/*svm->setGamma(0.01);
										svm->setC(10.0);*/
										//算法终止条件
	svm->setDegree(1.0);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 100, 1e-6));
	//3.训练支持向量
	svm->train(trainingDataMat, ml::SampleTypes::ROW_SAMPLE, labelsMat);
	//4.保存训练器
	svm->save("mnist_svm.xml");
	//5.导入训练器
	//Ptr<SVM> svm1 = StatModel::load<SVM>("mnist_dataset/mnist_svm.xml");

	//读取测试数据
	Vec3b green(0, 255, 0), blue(255, 0, 0);
	for (int i = 0; i < matImg.rows; i++)
	{
		for (int j = 0; j < matImg.cols; j++)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << j, i);
			float fRespone = svm->predict(sampleMat);
			if (fRespone == 1)
			{
				matImg.at<cv::Vec3b>(i, j) = green;
			}
			else if (fRespone == -1)
			{
				matImg.at<cv::Vec3b>(i, j) = blue;
			}
		}
	}
	// Show the training data  
	int thickness = -1;
	int lineType = 8;
	for (int i = 0; i < trainingDataMat.rows; i++)
	{
		if (labels[i] == 1)
		{
			circle(matImg, Point(trainingData[i][0], trainingData[i][1]), 5, Scalar(0, 0, 0), thickness, lineType);
		}
		else
		{
			circle(matImg, Point(trainingData[i][0], trainingData[i][1]), 5, Scalar(255, 255, 255), thickness, lineType);
		}
	}

	//显示支持向量点
	thickness = 2;
	lineType = 8;
	Mat vec = svm->getSupportVectors();
	int nVarCount = svm->getVarCount();//支持向量的维数
	_cprintf("vec.rows=%d vec.cols=%d\n", vec.rows, vec.cols);
	for (int i = 0; i < vec.rows; ++i)
	{
		int x = (int)vec.at<float>(i, 0);
		int y = (int)vec.at<float>(i, 1);
		_cprintf("vec.at=%d %f,%f\n", i, vec.at<float>(i, 0), vec.at<float>(i, 1));
		_cprintf("x=%d,y=%d\n", x, y);
		circle(matImg, Point(x, y), 6, Scalar(0, 0, 255), thickness, lineType);
	}


	imshow("circle", matImg); // show it to the user 
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