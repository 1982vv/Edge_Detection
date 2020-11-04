

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include "stdafx.h"
#include "imatrix.h"
#include "ETF.h"
#include "fdog.h"
#include "myvec.h"

#define N 3
using namespace cv;
using namespace std;

void conv2D(InputArray src, InputArray kernel, OutputArray dst, int ddepth,
	Point anchor = Point(-1, -1), int borderType = BORDER_DEFAULT)
{
		//卷积运算的第一步:卷积核逆时针翻转180°
		Mat kernelFlip;
		flip(kernel, kernelFlip, -1);
		//卷积运算的第二步
		filter2D(src, dst,ddepth, kernelFlip, anchor,0.0, BORDER_DEFAULT);
}

/*可分离的离散二维卷积,先进行垂直方向上的卷积,然后进行水平方向上的卷积*/
void sepConv2D_Y_X(InputArray src, OutputArray src_kerY_kerX, int ddepth,InputArray kernelY, InputArray kernelX, Point anchor = Point(-1, -1), 
	int borderType = BORDER_DEFAULT)
{
	//输入矩阵与垂直方向上的卷积核的卷积
	Mat src_kerY;
	conv2D(src, kernelY, src_kerY, ddepth, anchor, borderType);
	//上面得到的卷积结果,接着和水平方向上的卷积核卷积的
	conv2D(src_kerY, kernelX, src_kerY_kerX, ddepth, anchor, borderType);
}

/*可分离的离散二维卷积,先进行水平方向上的卷积,然后进行垂直方向上的卷积*/
void sepConv2D_X_Y(InputArray src, OutputArray src_kerX_kerY, int ddepth,
	InputArray kernelX, InputArray kernelY, Point anchor = Point(-1, -1), int
	borderType = BORDER_DEFAULT)
{
	//输入矩阵与垂直方向上的卷积核的卷积
	Mat src_kerX;
	conv2D(src, kernelX, src_kerX, ddepth, anchor, borderType);
	//上面得到的卷积结果,接着和水平方向上的卷积核的卷积
	conv2D(src_kerX, kernelX, src_kerX_kerY, ddepth, anchor, borderType);
}

/*prewitt边缘提取*/
void prewitt(InputArray src, OutputArray dst, int ddepth, int x, int y, int borderType = BORDER_DEFAULT)
{
	CV_Assert(!(x == 0 && y == 0));
	//如果x不等于零, src和 prewitt_x卷积核进行卷积运算
	if (x != 0 && y == 0)
	{
		// 可分 prewitt离的卷积核
		Mat prewitt_x_y = (Mat_<float>(3, 1) << 1, 1, 1);
		Mat prewitt_x_x = (Mat_<float>(1, 3) << 1, 0, -1);
		// 可分离的离散的二维卷积
		sepConv2D_Y_X(src, dst, ddepth, prewitt_x_y, prewitt_x_x, Point(-1, -1), borderType);
	}
	//如果x等于零且y不等于零, src和 prewitt_y卷积核进行卷积运算
	if (y != 0 && x == 0)
	{		
		// 可分离的prewitt卷积核
		Mat prewitt_y_x = (Mat_<float>(3, 1) << 1, 0, -1);
		Mat prewitt_y_y = (Mat_<float>(1, 3) << 1, 1, 1);
		// 可分离的离散的二维卷积
		sepConv2D_X_Y(src, dst, ddepth, prewitt_y_x, prewitt_y_y, Point(-1, -1), borderType);
	}
}

int main()
{
	std::string image_names[N] = {"test","lena","TJU"};
	Mat images[N];
	Mat grey_images[N];
	Mat Prewitt_edges[N];
	Mat Sobel_edges[N];
	Mat Canny_edges[N];
	ETF e[N];
	imatrix img[N];
	Mat FDoG_edges[N];
	images[0] = imread("./test.jpg");
	images[1] = imread("./lena.jpg");
	images[2] = imread("./TJU.jpg");

	//Prewitt实现
	for (int i = 0;i < N;i++)
	{
		if (!images[i].data)
		{
			std::cout << "没有图片" << std::endl;
			return -1;
		}
		cvtColor(images[i], grey_images[i], COLOR_BGR2GRAY);
		/*第二步: prewitt卷积*/
		//图像矩阵和prewittx卷积核卷积
		Mat img_prewitt_x;
		prewitt(grey_images[i], img_prewitt_x, CV_32FC1, 1, 0);
		//图像矩阵与prewitt_y卷积核卷积
		Mat img_prewitt_y;
		prewitt(grey_images[i], img_prewitt_y, CV_32FC1, 0, 1);
		/*第三步:水平方向和垂直方向上的边缘强度*/
		//数据类型转换,边缘强度的灰度级显示
		Mat abs_img_prewitt_x, abs_img_prewitt_y;
		convertScaleAbs(img_prewitt_x, abs_img_prewitt_x, 1, 0);
		convertScaleAbs(img_prewitt_y, abs_img_prewitt_y, 1, 0);
		/*第四步:通过第三步得到的两个方向上的边缘强度,求出最终的边缘强度*/
		//这里采用平方根的方式
		Mat img_prewitt_x2, image_prewitt_y2;
		
		pow(img_prewitt_x, 2.0, img_prewitt_x2);
		pow(img_prewitt_y, 2.0, image_prewitt_y2);
		sqrt(img_prewitt_x2 + image_prewitt_y2, Prewitt_edges[i]);
		//数据类型转换,边缘强度的灰度级显示
		Prewitt_edges[i].convertTo(Prewitt_edges[i], CV_8UC1);
		
	}

	for (int i = 0;i < N;i++)
	{
		//这里直接调用opencv中的Sobel函数，不做具体实现，因为原理和上面写的prewitt函数类似
		Mat img_Sobel_x;
		Sobel(grey_images[i], img_Sobel_x, CV_32FC1, 1, 0);
		Mat img_Sobel_y;
		Sobel(grey_images[i], img_Sobel_y, CV_32FC1, 0, 1);
		Mat abs_img_Sobel_x, abs_img_Sobel_y;
		convertScaleAbs(img_Sobel_x, abs_img_Sobel_x, 1, 0);
		convertScaleAbs(img_Sobel_y, abs_img_Sobel_y, 1, 0);
		Mat img_Sobel_x2, image_Sobel_y2;
		pow(img_Sobel_x, 2.0, img_Sobel_x2);
		pow(img_Sobel_y, 2.0, image_Sobel_y2);
		sqrt(img_Sobel_x2 + image_Sobel_y2, Sobel_edges[i]);
		Sobel_edges[i].convertTo(Sobel_edges[i], CV_8UC1);
	}

	for (int i = 0;i < N;i++)
	{
		Canny(grey_images[i], Canny_edges[i], 80, 200);
	}

	for (int i = 0;i < N;i++)
	{
		imshow("prewitt_"+image_names[i], Prewitt_edges[i]);
		imshow("Sobel_" + image_names[i], Sobel_edges[i]);
		imshow("Canny_" + image_names[i], Canny_edges[i]);
	}
	
	//FDoG实现
	double tao = 0.99;
	double thres = 0.7;
	double sigma = 1.0;
	double sigma3 = 3.0;

	int wz = 15;
	double colSig = 15.0;
	double spaSig = 10.0;
	int iterFDoG = 2;
	
	for (int s = 0;s < N;s++)
	{
		if (!images[s].data)
		{
			std::cout << "没有图片" << std::endl;
			return -1;
		}
		//bilateral filtering
		Mat filterImg;
		//color sigma对降噪起了作用
		bilateralFilter(grey_images[s], filterImg, wz, colSig, spaSig);
	
		
		grey_images[s] = filterImg;
	
		cout << "rows: " << images[s].rows << "   " << "cols: " << images[s].cols << endl;
		
		int index;
		img[s].init(images[s].rows, images[s].cols);
	
		for (int i = 0; i < images[s].rows; i++) {
			img[s].p[i] = new int[images[s].cols];
			for (int j = 0; j < images[s].cols; j++) {
				index = i * images[s].cols + j;
				img[s].p[i][j] = grey_images[s].data[index];
			}
		}
	
		int image_x = img[s].getRow();
		int image_y = img[s].getCol();
		
		cout << e[s].Nr << " " << e[s].Nc<<endl;
		e[s].init(image_x, image_y);
		cout << e[s].Nr << " " << e[s].Nc << endl;
	
		e[s].set(img[s]);
	
		e[s].Smooth(4, 2);

		GetFDoG(img[s], e[s], sigma, sigma3, tao);
	
		int pxVal;
		for (int i = 0; i < iterFDoG; i++)
		{
			for (int j = 0; j < img[s].getRow(); j++)
			{
				for (int k = 0; k < img[s].getCol(); k++)
				{
					pxVal = (int)img[s][j][k] + (int)(grey_images[s].data[j * img[s].getCol() + k]);
					if (pxVal > 255)
						pxVal = 255;
					img[s][j][k] = pxVal;
				}
			}
			GetFDoG(img[s], e[s], sigma, sigma3, tao);
		}
		GrayThresholding(img[s], thres);

		FDoG_edges[s] = grey_images[s];
		for (int i = 0; i < images[s].rows; i++) {
			for (int j = 0; j < images[s].cols; j++) {
				index = i * images[s].cols + j;
				FDoG_edges[s].data[index] = img[s].p[i][j];
			}
		}
		imshow("FDog_"+ image_names[s], FDoG_edges[s]);
	}
	
	waitKey(0);
	return 0;

}