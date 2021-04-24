/*
* create_mask.cpp
*
* Author:
* Siddharth Kherada <siddharthkherada27[at]gmail[dot]com>
*
* This tutorial demonstrates how to make mask image (black and white).
* The program takes as input a source image and outputs its corresponding
* mask image.
*/

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/cudafeatures2d.hpp>
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"

using namespace std;
using namespace cv;

cv::cuda::GpuMat srcDevice, mask, finalImg;
bool found = false;

Mat src, img1;

Point point;
vector<Point> pts;

//vector<Point> pts= { Point(65,100) , Point(569,60), Point(906,193),
//		Point(1223,349), Point(1218,613), Point(1192,698), Point(54,685),
//		Point(32,225), Point(64,103) };

//[65, 100;
// 569, 60;
// 906, 193;
// 1223, 349;
// 1218, 613;
// 1192, 698;
// 54, 685;
// 32, 225;
// 64, 103]


int drag = 0;
int var = 0;
int flag = 0;

void mouseHandler(int, int, int, int, void*);


cv::cuda::GpuMat doMask(cv::cuda::GpuMat srci, cv::cuda::GpuMat d, bool preview=true) {
	cv::cuda::GpuMat desti = d;
	if(desti.rows == 0){
		desti = cv::cuda::GpuMat(srci.rows, srci.cols, CV_8UC3);
		desti.setTo(Scalar::all(0));
		//desti.copyTo(desti);

	}

	if(mask.rows == 0){
		Mat tmp_mask(srci.size(), CV_8UC1);
		tmp_mask= Mat::zeros(srci.size(), CV_8UC1);
		fillPoly(tmp_mask, pts, Scalar(255, 255, 255), 8, 0);
		cout <<  pts << endl;
		mask.upload(tmp_mask);
		cout << tmp_mask.size() << tmp_mask.type() << endl;

	}

	cout << mask.size() << mask.type() << endl;
	cv::cuda::GpuMat gray;
	cv::cuda::cvtColor(srci, gray, cv::COLOR_BGR2GRAY);
	cv::cuda::bitwise_and(gray, gray, desti, mask);
	finalImg = desti;
	if(preview){
		Mat mTmp, fTmp;
		mask.download(mTmp);
		finalImg.download(fTmp);
		imshow("Mask", mTmp);
		imshow("Result", fTmp);
		imshow("Source", img1);
	}

	return mask;
}

cv::cuda::GpuMat doMask(cv::cuda::GpuMat srci, bool preview=true) {
	cv::cuda::GpuMat d;
	return doMask(srci, d, preview);
}

void mouseHandler(int event, int x, int y, int, void*)
{

    if (event == EVENT_LBUTTONDOWN && !drag)
    {
        if (flag == 0)
        {
            if (var == 0)
                img1 = src.clone();
            point = Point(x, y);
            circle(img1, point, 2, Scalar(0, 0, 255), -1, 8, 0);
            pts.push_back(point);
            var++;
            drag  = 1;

            if (var > 1)
                line(img1,pts[var-2], point, Scalar(0, 0, 255), 2, 8, 0);

            imshow("Source", img1);
        }
    }

    if (event == EVENT_LBUTTONUP && drag)
    {
        imshow("Source", img1);
        drag = 0;
    }

    if (event == EVENT_RBUTTONDOWN)
    {
        flag = 1;
        img1 = src.clone();

        if (var != 0)
        {
            polylines( img1, pts, 1, Scalar(0,0,0), 2, 8, 0);
        }

        imshow("Source", img1);
    }

    if (event == EVENT_RBUTTONUP)
    {
    	srcDevice.upload(src);
		doMask(srcDevice, true);
    }

    if (event == EVENT_MBUTTONDOWN)
    {
        pts.clear();
        var = 0;
        drag = 0;
        flag = 0;
        imshow("Source", src);
    }


}

cv::cuda::GpuMat getMask(cv::cuda::GpuMat srci)
{
	srcDevice = srci;
	if(found)
		return doMask(srcDevice, false);
    namedWindow("Source", WINDOW_AUTOSIZE);
    setMouseCallback("Source", mouseHandler, NULL);
    srcDevice.download(src);
    imshow("Source", src);
	if((char)cv::waitKey(0) == (char)27)
		cout << "returning by key" << endl;
	    found = true;
		cv::destroyAllWindows();
		return finalImg;

	cout << "returning" << endl;
    return mask;
}


int main2(int argc, char **argv) {

    CommandLineParser parser(argc, argv, "{@input | lena.jpg | input image}");
    parser.about("This program demonstrates using mouse events\n");
    parser.printMessage();
    cout << "\n\tleft mouse button - set a point to create mask shape\n"
        "\tright mouse button - create mask from points\n"
        "\tmiddle mouse button - reset\n";
    String input_image = parser.get<String>("@input");

    src = imread(samples::findFile(input_image));

    //cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);

    srcDevice.upload(src);

	getMask(srcDevice);
}

