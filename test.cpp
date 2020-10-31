

#include <opencv2/opencv.hpp>
#include <iostream>
#define N 3
using namespace cv;

void conv2D(InputArray src, InputArray kernel, OutputArray dst, int ddepth,
	Point anchor = Point(-1, -1), int borderType = BORDER_DEFAULT)
{
		//�������ĵ�һ��:�������ʱ�뷭ת180��
		Mat kernelFlip;
		flip(kernel, kernelFlip, -1);
		//�������ĵڶ���
		filter2D(src, dst,ddepth, kernelFlip, anchor,0.0, BORDER_DEFAULT);
}

/*�ɷ������ɢ��ά���,�Ƚ��д�ֱ�����ϵľ��,Ȼ�����ˮƽ�����ϵľ��*/
void sepConv2D_Y_X(InputArray src, OutputArray src_kerY_kerX, int ddepth,InputArray kernelY, InputArray kernelX, Point anchor = Point(-1, -1), 
	int borderType = BORDER_DEFAULT)
{
	//��������봹ֱ�����ϵľ���˵ľ��
	Mat src_kerY;
	conv2D(src, kernelY, src_kerY, ddepth, anchor, borderType);
	//����õ��ľ�����,���ź�ˮƽ�����ϵľ���˾����
	conv2D(src_kerY, kernelX, src_kerY_kerX, ddepth, anchor, borderType);
}

/*�ɷ������ɢ��ά���,�Ƚ���ˮƽ�����ϵľ��,Ȼ����д�ֱ�����ϵľ��*/
void sepConv2D_X_Y(InputArray src, OutputArray src_kerX_kerY, int ddepth,
	InputArray kernelX, InputArray kernelY, Point anchor = Point(-1, -1), int
	borderType = BORDER_DEFAULT)
{
	//��������봹ֱ�����ϵľ���˵ľ��
	Mat src_kerX;
	conv2D(src, kernelX, src_kerX, ddepth, anchor, borderType);
	//����õ��ľ�����,���ź�ˮƽ�����ϵľ���˵ľ��
	conv2D(src_kerX, kernelX, src_kerX_kerY, ddepth, anchor, borderType);
}

/*prewitt��Ե��ȡ*/
void prewitt(InputArray src, OutputArray dst, int ddepth, int x, int y, int borderType = BORDER_DEFAULT)
{
	CV_Assert(!(x == 0 && y == 0));
	//���x��������, src�� prewitt_x����˽��о������
	if (x != 0 && y == 0)
	{
		// �ɷ� prewitt��ľ����
		Mat prewitt_x_y = (Mat_<float>(3, 1) << 1, 1, 1);
		Mat prewitt_x_x = (Mat_<float>(1, 3) << 1, 0, -1);
		// �ɷ������ɢ�Ķ�ά���
		sepConv2D_Y_X(src, dst, ddepth, prewitt_x_y, prewitt_x_x, Point(-1, -1), borderType);
	}
	//���x��������y��������, src�� prewitt_y����˽��о������
	if (y != 0 && x == 0)
	{		
		// �ɷ����prewitt�����
		Mat prewitt_y_x = (Mat_<float>(3, 1) << 1, 0, -1);
		Mat prewitt_y_y = (Mat_<float>(1, 3) << 1, 1, 1);
		// �ɷ������ɢ�Ķ�ά���
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
	images[0] = imread("./test.jpg");
	images[1] = imread("./lena.jpg");
	images[2] = imread("./TJU.jpg");

	//Prewittʵ��
	for (int i = 0;i < N;i++)
	{
		if (!images[i].data)
		{
			std::cout << "û��ͼƬ" << std::endl;
			return -1;
		}
		cvtColor(images[i], grey_images[i], COLOR_BGR2GRAY);
		/*�ڶ���: prewitt���*/
		//ͼ������prewittx����˾��
		Mat img_prewitt_x;
		prewitt(grey_images[i], img_prewitt_x, CV_32FC1, 1, 0);
		//ͼ�������prewitt_y����˾��
		Mat img_prewitt_y;
		prewitt(grey_images[i], img_prewitt_y, CV_32FC1, 0, 1);
		/*������:ˮƽ����ʹ�ֱ�����ϵı�Եǿ��*/
		//��������ת��,��Եǿ�ȵĻҶȼ���ʾ
		Mat abs_img_prewitt_x, abs_img_prewitt_y;
		convertScaleAbs(img_prewitt_x, abs_img_prewitt_x, 1, 0);
		convertScaleAbs(img_prewitt_y, abs_img_prewitt_y, 1, 0);
		/*���Ĳ�:ͨ���������õ������������ϵı�Եǿ��,������յı�Եǿ��*/
		//�������ƽ�����ķ�ʽ
		Mat img_prewitt_x2, image_prewitt_y2;
		
		pow(img_prewitt_x, 2.0, img_prewitt_x2);
		pow(img_prewitt_y, 2.0, image_prewitt_y2);
		sqrt(img_prewitt_x2 + image_prewitt_y2, Prewitt_edges[i]);
		//��������ת��,��Եǿ�ȵĻҶȼ���ʾ
		Prewitt_edges[i].convertTo(Prewitt_edges[i], CV_8UC1);
		
	}

	for (int i = 0;i < N;i++)
	{
		//����ֱ�ӵ���opencv�е�Sobel��������������ʵ�֣���Ϊԭ�������д��prewitt��������
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
		imshow("prewitt_"+image_names[i], Prewitt_edges[i]);
		imshow("Sobel_" + image_names[i], Sobel_edges[i]);
	}
	
		
		
	waitKey(0);
	return 0;

}