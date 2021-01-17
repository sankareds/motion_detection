#include <stdio.h>
#include <iostream>
#include "opencv2/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <string>
#include <cuda/cuda_runtime.h>
#include <opencv2/cudafeatures2d.hpp>
#include "opencv2/imgproc/types_c.h"
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudalegacy.hpp"
#include "opencv2/cudawarping.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <cuda/device_launch_parameters.h>
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudabgsegm.hpp"

using namespace cv;
using namespace std;
using namespace cv::cuda;


cuda::GpuMat getMask(cuda::GpuMat);
int detectMotion(const GpuMat &, Mat &, Mat &,
                 int, int, int, int,
                 int ,
                 Scalar &, Scalar);
bool saveImg(Mat, const string, const string, const char *, const char *);


int main( int argc, char** argv )
{
    system("clear");
    float time;
    Mat img;
    cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Input", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed", cv::WINDOW_AUTOSIZE);

    const char* gst =  "rtspsrc location=rtsp://admin:@cam2/ch0_0.264 name=r latency=0 protocols=tcp ! application/x-rtp,payload=96,encoding-name=H264 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw(memory:NVMM), format=BGRx ! nvvidconv ! videoconvert ! video/x-raw, format=BGR, framerate=15/1 ! appsink max-buffers=5 drop=true";
    cv::VideoCapture cap(gst, cv::CAP_GSTREAMER);
    if ( !cap.isOpened() )
    {
        cout << "Cannot open the video" << endl;
        return -1;
    }

    int scale = 1;
    Size s = img.size(); //calcolo dimensioni frame
    cout<<"Dimensioni originali "<<s.width<<" x "<<s.height<<endl;
    int N = s.height/scale;
    int M = s.width/scale;
    cout<<"Dimensioni elaborazione "<<M<<" x "<<N<<endl;


    Mat resized (N, M, CV_8UC1, Scalar(0,0,0)); //resized image
    GpuMat resized_device(N, M, CV_8UC3, Scalar(0,0,0)), out_device;
    Mat element = getStructuringElement( MORPH_RECT, Size(3, 3), Point( 1, 1) );
    Mat masked;
    Mat inframe;


    //cap.set(15, -8.0);
    unsigned int width  = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    unsigned int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    unsigned int fps    = cap.get(cv::CAP_PROP_FPS);
    unsigned int pixels = width*height;
    std::cout <<"Frame size : "<<width<<" x "<<height<<", "<<pixels<<" Pixels "<<fps<<" FPS"<<std::endl;



    int N_FRAME=1;
    unsigned int frameByteSize = pixels * 3;
    std::cout << "Before Cuda Call" << std::endl;


    void *unified_ptr;
    cudaMallocManaged(&unified_ptr, frameByteSize);
    cv::cuda::GpuMat mask,mask_device(height, width, CV_8UC3, unified_ptr);
    Mat cpu_in_device(height, width, CV_8UC3, unified_ptr);

    int number_of_changes, number_of_sequence=0;
    const string DIR = "/tmp/motion/pics/"; // directory where the images will be stored
    const string EXT = ".jpg"; // extension of the images
    const int DELAY = 500; // in mseconds, take a picture every 1/2 second
    const string LOGFILE = "/tmp/motion/log";
    // Format of directory
    string DIR_FORMAT = "%d%h%Y"; // 1Jan1970
    string FILE_FORMAT = DIR_FORMAT + "/" + "%d%h%Y_%H%M%S"; // 1Jan1970/1Jan1970_12153
    string CROPPED_FILE_FORMAT = DIR_FORMAT + "/cropped/" + "%d%h%Y_%H%M%S"; // 1Jan1970/cropped/1Jan1970_121539


    Ptr<BackgroundSubtractor> mog2 = cuda::createBackgroundSubtractorMOG2(700,16,false);
    Ptr<BackgroundSubtractorFGD> fgd = cuda::createBackgroundSubtractorFGD();
    Ptr<cuda::CLAHE> clahe = cv::cuda::createCLAHE(200, Size(8,8));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );

    while(1)
    {
        {
            bool bSuccesss = cap.read(inframe);



            GpuMat in_device, mask_device, yuv(N, M, CV_8UC3, Scalar(0,0,0));
            mask_device.upload(inframe);

            if(mask.rows == 0){
            	mask = getMask(mask_device);
            }

            in_device.setTo(Scalar(0,0,0));
            mask_device.copyTo(in_device, mask);

            //cv::cuda::cvtColor(yuv, mask_device, cv::COLOR_BGR2GRAY);
            // mask area
            //cv::cuda::threshold(mask_device, mask_device, 127, 255, 0);


//            cv::Ptr< cv::cuda::Filter > filter3 = cv::cuda::createSobelFilter(CV_8UC3, CV_8UC3, 1, 1, 1, 1, cv::BORDER_DEFAULT);
//            filter3->apply(in_device, in_device);

            // normalization
//            cv::cuda::cvtColor(in_device, yuv, cv::COLOR_BGR2YUV);
//            std::vector<GpuMat> channels;
//            cv::cuda::split(yuv, channels);
////			cv::cuda::equalizeHist(channels[0], channels[0]);
////			cv::cuda::equalizeHist(channels[1], channels[1]);
////			cv::cuda::equalizeHist(channels[2], channels[2]);
//			clahe->apply(channels[0], channels[0]);
//			clahe->apply(channels[1], channels[1]);
//			clahe->apply(channels[2], channels[2]);


            //cv::cuda::merge(channels, yuv);
            //clahe->apply(yuv, yuv);
            //cv::cuda::equalizeHist(yuv, yuv);
            //cv::cuda::cvtColor(yuv, yuv, cv::COLOR_YUV2BGR);
            //cv::cuda::threshold(yuv, yuv, 127, 255, CV_THRESH_BINARY);



            //cv::cuda::threshold(yuv, yuv, 100, 255, CV_THRESH_BINARY_INV);




            //cuda::resize(yuv, resized_device, Size(M,N));
            //cv::cuda::cvtColor(yuv, resized_device, cv::COLOR_BGR2GRAY);
            in_device.copyTo(resized_device);
            cv::Ptr<cv::cuda::Filter> filter = cuda::createGaussianFilter(resized_device.type(), resized_device.type(), Size(3, 3),0);
            filter->apply(resized_device, resized_device);
            // mog already does gaussion


            mog2->apply(resized_device, out_device);



            //changeDetection<<<M, N>>>(resized_device, buffer_device, disc_device, out_device, k_device, buff_size, old_disc_device);
            //cv::cuda::threshold(out_device, out_device, 127, 255, CV_THRESH_BINARY);



            filter = cuda::createMorphologyFilter(CV_MOP_OPEN, out_device.type(), element);
            filter->apply(out_device, out_device);


//            cv::Ptr< cv::cuda::Filter > filter3 = cv::cuda::createSobelFilter(CV_8UC1, CV_8UC1, 1, 1, 1, 1);
//            filter3->apply(out_device, out_device);
//            cv::cuda::threshold(out_device, out_device, 100, 255, CV_THRESH_BINARY_INV);

            GpuMat thresh;
            //cv::cuda::threshold(out_device, out_device, 100, 255, 0);
            //cv::cuda::threshold(fgMat, fgMat, 100, 255, 0);
            Mat result_host(out_device);

            //foreground feature detection
            GpuMat fgMat(N, M, CV_8UC1, Scalar(0));;
            std::vector<Mat> features;
            vector<Vec4i> hierarchy;
//            fgd->apply(out_device, fgMat);
//            fgd->getForegroundRegions(features);
        	cv::findContours(result_host, features, hierarchy, CV_RETR_CCOMP,
        			CV_CHAIN_APPROX_SIMPLE);
            int contourSize=0;
            for (auto it = begin (features); it != end (features); ++it) {
                int area = it->rows * it->cols;
                if(area > contourSize){
                	contourSize = area;
                }
            }


            Mat in_host(in_device);
            Mat processed(resized_device);
            Mat result_cropped;
            // Detect motion in window
            int x_start = 0, x_stop = inframe.cols;
            int y_start = 0, y_stop = inframe.rows;
            Scalar color(0,255,255);
            cv::Scalar mean, std;
			Mat in_device_cpu;
            //in_device.download(in_device_cpu);
            cv::cuda::meanStdDev(out_device, mean, std);
            cout << "number of changes" << number_of_changes;
            cout << "number of changes=" << features.size() << "|| Max Contour=" << contourSize<< endl;

            int captureCount=0;
            if((N_FRAME > 50 && contourSize > 500 ))
            {
                //cout << "--------------****motion detected------------**************" << number_of_changes << endl;
                if(number_of_sequence % 1 == 0){
					//cout << "writing image to disk" << inframe.rows << endl;
					saveImg( processed , DIR,EXT,DIR_FORMAT.c_str(),FILE_FORMAT.c_str());
					//saveImg(result_cropped,DIR,EXT,DIR_FORMAT.c_str(),CROPPED_FILE_FORMAT.c_str());
                }
                if(captureCount  == 0){
                	captureCount = 5;
                }else{
                	captureCount--;
                }

                number_of_sequence++;
            }else
            {
                number_of_sequence = 0;
            }
            cv::imshow("Output",result_host);
            cv::imshow("Input",in_host);
            cv::imshow("Processed",processed);
    		if((char)cv::waitKey(1) == (char)27)
    			break;
            N_FRAME++;
        }
    }
    return 0;
}

