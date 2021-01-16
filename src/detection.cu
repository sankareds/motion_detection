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

__global__ void trainKernel(cuda::PtrStepSz<uchar>in_device,cuda::PtrStepSz<uchar>buffer, int k); //fill initial buffer
__global__ void calculateDiscr(cuda::PtrStepSz<uchar>in_device, cuda::PtrStepSz<uchar>buffer,cuda::PtrStepSz<uchar>disc_device, const int buff_size,cuda::PtrStepSz<uchar>old_disc_device); //calculate initial backgorund
__global__ void changeDetection(cuda::PtrStepSz<uchar>in_device, cuda::PtrStepSz<uchar>buffer_device, cuda::PtrStepSz<uchar>disc_device, cuda::PtrStepSz<uchar>out_device, cuda::PtrStepSz<uchar>k_device, int buff_size,cuda::PtrStepSz<uchar>old_disc_device); //motion detection
__device__ int sort_and_median(int arr[], int length); //calculate new dicriminator for a pixel

cuda::GpuMat maskImage(cuda::GpuMat);
int detectMotion(const GpuMat &, Mat &, Mat &,
                 int, int, int, int,
                 int ,
                 Scalar &, Scalar);
bool saveImg(Mat, const string, const string, const char *, const char *);

int getContourSize(const Mat &result_host) {
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;
	cv::findContours(result_host, contours, hierarchy, CV_RETR_CCOMP,
			CV_CHAIN_APPROX_SIMPLE);
	int largest_area = 0;

		int largest_contour_index = 0;

		for (int i = 0; i < contours.size(); i++) {
			double a = cv::contourArea(contours[i], false);
			cout << "contour area=" << a << endl;
			if (a > largest_area) {
				largest_area = a;

				largest_contour_index = i;
			}
	}

	return largest_area;
}

int main( int argc, char** argv )
{
    system("clear");
    int scale=1;
    const int buff_size=150; //buffer size in numer of frame
    int TRAIN=true; //flag for training fase
    cudaEvent_t start, stop;
    float time;
    Mat img;
    cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Background", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Masked", cv::WINDOW_AUTOSIZE);
    const char* gst =  "rtspsrc location=rtsp://admin:@cam1/ch0_0.264 name=r latency=0 protocols=tcp ! application/x-rtp,payload=96,encoding-name=H264 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw(memory:NVMM), format=BGRx ! nvvidconv ! videoconvert ! video/x-raw, format=BGR, framerate=5/1 ! appsink max-buffers=5 drop=true";
    cv::VideoCapture cap(gst, cv::CAP_GSTREAMER);
    if ( !cap.isOpened() )
    {
        cout << "Cannot open the video" << endl;
        return -1;
    }
    bool bSuccess = cap.read(img);
    if (!bSuccess)
    {

        cout << "Impossibile leggere frame di input" << endl;
        return 0;

    }
    Size s = img.size(); //calcolo dimensioni frame
    cout<<"Dimensioni originali "<<s.width<<" x "<<s.height<<endl;
    int N = s.height/scale;
    int M = s.width/scale;
    cout<<"Dimensioni elaborazione "<<M<<" x "<<N<<endl;


    cv::VideoWriter oVW ("output.avi", cv::VideoWriter::fourcc('M','P','4','2'), 5, Size(M,N), false); //inizializza oggetto VideoWriter
    if ( !oVW.isOpened() ) //if not initialize the VideoWriter successfully, exit the program
    {
        cout << "ERROR: Failed to write the video" << endl;
        return -1;
    }

	cv::VideoWriter bVW ("background.avi", cv::VideoWriter::fourcc('M','P','4','2'), 5, Size(M,N), false); //inizializza oggetto VideoWriter
	if ( !bVW.isOpened() ) //if not initialize the VideoWriter successfully, exit the program
    {
        cout << "ERROR: Failed to write the video" << endl;
        return -1;
    }


    Mat outframe(N, M, CV_8UC1, Scalar(0)); //output frame
    Mat disc(N, M, CV_8UC3, Scalar(0,0,0)); //background
    Mat old_disc(N, M, CV_8UC3, Scalar(0,0,0)); //old backgorund for comparison
    Mat resized (N, M, CV_8UC1, Scalar(0,0,0)); //resized image
    Mat k(N, M, CV_8UC1,Scalar(0)); //last buffer element pointer
    Mat buffer(N*buff_size,M, CV_8UC3 , Scalar(0,0,0)); //buffer for motion memorization
    Mat element = getStructuringElement( MORPH_RECT, Size(3, 3), Point( 1, 1) );
    Mat masked;


    //cap.set(15, -8.0);
    unsigned int width  = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    unsigned int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    unsigned int fps    = cap.get(cv::CAP_PROP_FPS);
    unsigned int pixels = width*height;
    std::cout <<"Frame size : "<<width<<" x "<<height<<", "<<pixels<<" Pixels "<<fps<<" FPS"<<std::endl;

    Mat inframe;


    unsigned int frameByteSize = pixels * 3;
    std::cout << "Before Cuda Call" << std::endl;


    void *unified_ptr;
    cudaMallocManaged(&unified_ptr, frameByteSize);
    cv::cuda::GpuMat mask_device(height, width, CV_8UC3, unified_ptr);
    Mat cpu_in_device(height, width, CV_8UC3, unified_ptr);

    cuda::GpuMat out_device, disc_device, buffer_device, k_device, old_disc_device, resized_device; //same Mats on Gpu
	//GpuMats upload on Gpu
    disc_device.upload(disc);
    old_disc_device.upload(old_disc);
    resized_device.upload(resized);
    buffer_device.upload(buffer);
    k_device.upload(k);
    int N_FRAME=1; // frame counter
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );
    TRAIN = false;

    int number_of_changes, number_of_sequence=0;
    const string DIR = "/tmp/motion/pics/"; // directory where the images will be stored
    const string EXT = ".jpg"; // extension of the images
    const int DELAY = 500; // in mseconds, take a picture every 1/2 second
    const string LOGFILE = "/tmp/motion/log";

    // Format of directory
    string DIR_FORMAT = "%d%h%Y"; // 1Jan1970
    string FILE_FORMAT = DIR_FORMAT + "/" + "%d%h%Y_%H%M%S"; // 1Jan1970/1Jan1970_12153
    string CROPPED_FILE_FORMAT = DIR_FORMAT + "/cropped/" + "%d%h%Y_%H%M%S"; // 1Jan1970/cropped/1Jan1970_121539


    Ptr<BackgroundSubtractor> mog2 = cuda::createBackgroundSubtractorMOG2(500,32,true);
    Ptr<BackgroundSubtractorFGD> fgd = cuda::createBackgroundSubtractorFGD();
    Ptr<cuda::CLAHE> clahe = cv::cuda::createCLAHE(4, Size(8,8));

    while(1)
    {
        outframe.setTo(0);
        out_device.upload(outframe);
        if (TRAIN==true) // training phase for initial background
        {
            cout<<"Training sfondo"<<endl;
            for(int y=0; y<buff_size; y++)
            {
                    bool bSuccess = cap.read(inframe);
                    if (!bSuccess)
                    {
                        cout << "Impossibile leggere frame di input" << endl;
                        return -1;

                    }
                    if (inframe.empty()) //check whether the image is loaded or not
                    {
                        cout << "Cannot open frame for train" << endl;
                        return -2;
                    }
                    N_FRAME++;
                    //inframe.copyTo(cpu_in_device);
                    mask_device.upload(inframe);
                    cv::cuda::cvtColor(mask_device, mask_device, cv::COLOR_BGR2GRAY);
                    //cv::cuda::threshold(mask_device, mask_device, 127, 255, 0);
                    cv::cuda::GpuMat in_device = maskImage(mask_device);
                    cv::cuda::cvtColor(in_device, in_device, cv::COLOR_GRAY2BGR);


                    cuda::resize(in_device, resized_device, Size(M,N));
                    cv::Ptr<cv::cuda::Filter> filter = cuda::createGaussianFilter(resized_device.type(), resized_device.type(), Size(3, 3),0);
                                filter->apply(resized_device, resized_device);
                    filter -> apply(resized_device, resized_device);
                    cout << in_device.size() << in_device.type() <<endl;
                    trainKernel<<<M, N>>>(resized_device, buffer_device, y);
                    cout << "end" <<endl;

            }
            cout<<"Calcolo sfondo iniziale"<<endl;
            calculateDiscr<<<M, N>>>(resized_device, buffer_device, disc_device, buff_size, old_disc_device);
            cout<<"Elaborazione"<<endl;
            TRAIN=false;
        }
        else
        {
            bool bSuccesss = cap.read(inframe);
            if (!bSuccesss)
            {
                cout << "Impossibile leggere frame di input" << endl;
                cudaEventRecord( stop, 0 );
                cudaEventSynchronize( stop );
                cudaEventElapsedTime( &time, start, stop );
                printf("execution time %8.2f ms\nAvg. FPS: %8.2f for %d frame\n",time, N_FRAME/(time/1000), N_FRAME);
                cudaEventDestroy( start );
                cudaEventDestroy( stop );
                return -1;

            }
            if (inframe.empty()) //check whether the image is loaded or not
            {
                cout << "Video ended" << endl;
                cudaEventRecord( stop, 0 );
                cudaEventSynchronize( stop );
                cudaEventElapsedTime( &time, start, stop );
                printf("execution time %8.2f ms\nAvg. FPS: %8.2f for %d frame\n",time, N_FRAME/(time/1000), N_FRAME);
                cudaEventDestroy( start );
                cudaEventDestroy( stop );
                return -1;
            }

            GpuMat yuv;
            yuv.upload(inframe);



            cv::cuda::cvtColor(yuv, yuv, cv::COLOR_BGR2Lab);
            std::vector<GpuMat> channels;
            cv::cuda::split(yuv, channels);

            clahe->apply(channels[0], channels[0]);
            //cv::cuda::equalizeHist(channels[0], channels[0]);
            cv::cuda::merge(channels, yuv);
            cv::cuda::cvtColor(yuv, yuv, cv::COLOR_Lab2BGR);
            cv::cuda::cvtColor(yuv, mask_device, cv::COLOR_BGR2GRAY);

            //cv::cuda::threshold(mask_device, mask_device, 127, 255, 0);
            cv::cuda::GpuMat in_device = maskImage(mask_device);

            GpuMat fgMat;
            std::vector<Mat> features;
            //cv::cuda::cvtColor(in_device, fgMat, cv::COLOR_GRAY2BGR);
            //fgd->apply(fgMat, fgMat);
            //fgd->getForegroundRegions(features);

            //cout << features.size() << endl;

            cuda::resize(in_device, resized_device, Size(M,N));
            cv::Ptr<cv::cuda::Filter> filter = cuda::createGaussianFilter(resized_device.type(), resized_device.type(), Size(3, 3),0);
            filter->apply(resized_device, resized_device);
            // mog already does gaussion
            mog2->apply(resized_device, out_device);


            //changeDetection<<<M, N>>>(resized_device, buffer_device, disc_device, out_device, k_device, buff_size, old_disc_device);
            cv::cuda::threshold(out_device, out_device, 127, 255, CV_THRESH_BINARY);



            filter = cuda::createMorphologyFilter(CV_MOP_OPEN, out_device.type(), element);
            filter->apply(out_device, out_device);



            GpuMat thresh;
            //cv::cuda::threshold(out_device, thresh, 127, 255, 0);
            Mat result_host(out_device);




            Mat back_host(yuv);

            Mat result_cropped;
            // Detect motion in window
            int x_start = 0, x_stop = inframe.cols;
            int y_start = 0, y_stop = inframe.rows;
            Scalar color(0,255,255);
            cv::Scalar mean, std;
			Mat in_device_cpu;
			cv::cuda::cvtColor(in_device, in_device, cv::COLOR_GRAY2BGR);
            in_device.download(in_device_cpu);
            cv::cuda::meanStdDev(out_device, mean, std);
            number_of_changes = detectMotion(out_device, in_device_cpu, result_cropped,  x_start, x_stop, y_start, y_stop, 20, color, std);
            int contourSize = getContourSize(result_host);
            cout << "number of changes" << number_of_changes;
            cout << "stddev=" << std[0] << "|| number of changes=" << number_of_changes << "|| Contour=" << contourSize<< endl;

            int captureCount=0;
            if((N_FRAME > 50 && number_of_changes > 300 && contourSize > 100) || captureCount > 0 )
            {
                //cout << "--------------****motion detected------------**************" << number_of_changes << endl;
                if(number_of_sequence % 1 == 0){
					//cout << "writing image to disk" << inframe.rows << endl;
					//saveImg( in_device_cpu , DIR,EXT,DIR_FORMAT.c_str(),FILE_FORMAT.c_str());
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
            cv::imshow("Background",back_host);
    		if((char)cv::waitKey(1) == (char)27)
    			break;
            oVW.write(result_host);
            bVW.write(back_host);
            N_FRAME++;
        }
    }
    return 0;
}


__global__ void trainKernel(cuda::PtrStepSz<uchar>in_device,cuda::PtrStepSz<uchar>buffer, int k)
{
    int y=threadIdx.x;
    int x=blockIdx.x;
    buffer.ptr(y+k*in_device.rows)[x*3]=in_device.ptr(y)[x*3];
    buffer.ptr(y+k*in_device.rows)[x*3+1]=in_device.ptr(y)[x*3+1];
    buffer.ptr(y+k*in_device.rows)[x*3+2]=in_device.ptr(y)[x*3+2];
    __syncthreads();
}

__global__ void calculateDiscr(cuda::PtrStepSz<uchar>in_device, cuda::PtrStepSz<uchar>buffer_device,cuda::PtrStepSz<uchar>disc_device, const int buff_size,cuda::PtrStepSz<uchar>old_disc_device)
{
    int y=threadIdx.x;
    int x=blockIdx.x;
    int arrR[150];
    int arrG[150];
    int arrB[150];
    //scorro lista per calcolare il giusto discriminatore per ogni pixel
    for(int k=0; k<buff_size; k++)
    {
        // il discrminatore é dato dalla media dei valori RGB del pixel
        arrR[k]=buffer_device.ptr(y+k*in_device.rows)[x*3+2];
        arrG[k]=buffer_device.ptr(y+k*in_device.rows)[x*3+1];
        arrB[k]=buffer_device.ptr(y+k*in_device.rows)[x*3];
    }
    disc_device.ptr(y)[x*3+2]=sort_and_median(arrR, buff_size);
    disc_device.ptr(y)[x*3+1]=sort_and_median(arrG, buff_size);
    disc_device.ptr(y)[x*3]=sort_and_median(arrB, buff_size);
    old_disc_device.ptr(y)[x*3+2]=disc_device.ptr(y)[x*3+2];
    old_disc_device.ptr(y)[x*3+1]=disc_device.ptr(y)[x*3+1];
    old_disc_device.ptr(y)[x*3]=disc_device.ptr(y)[x*3];
    __syncthreads();
}

__global__ void changeDetection(cuda::PtrStepSz<uchar>in_device, cuda::PtrStepSz<uchar>buffer_device, cuda::PtrStepSz<uchar>disc_device, cuda::PtrStepSz<uchar>out_device, cuda::PtrStepSz<uchar>k_device, int buff_size,cuda::PtrStepSz<uchar>old_disc_device)
{
    int y=threadIdx.x;
    int x=blockIdx.x;
    int R_diff= abs(disc_device.ptr(y)[x*3+2]-in_device.ptr(y)[x*3+2]);
    int G_diff= abs(disc_device.ptr(y)[x*3+1]-in_device.ptr(y)[x*3+1]);
    int B_diff= abs(disc_device.ptr(y)[x*3]-in_device.ptr(y)[x*3]);
    int Sim=(R_diff+G_diff+B_diff)/3;
    if(Sim>=10)  //   if movement
    {
        int R_old_diff= abs(old_disc_device.ptr(y)[x*3+2]-in_device.ptr(y)[x*3+2]);
        int G_old_diff= abs(old_disc_device.ptr(y)[x*3+1]-in_device.ptr(y)[x*3+1]);
        int B_old_diff= abs(old_disc_device.ptr(y)[x*3]-in_device.ptr(y)[x*3]);
        int old_Sim=(R_old_diff+G_old_diff+B_old_diff)/3;
        if(old_Sim<=10)  //if movement less than the old
        {
            disc_device.ptr(y)[x*3+2]=old_disc_device.ptr(y)[x*3+2];
            disc_device.ptr(y)[x*3+1]=old_disc_device.ptr(y)[x*3+1];
            disc_device.ptr(y)[x*3]=old_disc_device.ptr(y)[x*3];
            k_device.ptr(y)[x]=0;
        }
        else
        {
            out_device.ptr(y)[x]=255;
            if(k_device.ptr(y)[x]>=buff_size)
            {
                int arrR[150];
                int arrG[150];
                int arrB[150];
                //scroll through the list to calculate the right discriminator for each pixel
                for(int k=0; k<buff_size; k++)
                {
                    // the discriminator is given by the average of the RGB values ​​of the pixel
                    arrR[k]=buffer_device.ptr(y+k*in_device.rows)[x*3+2];
                    arrG[k]=buffer_device.ptr(y+k*in_device.rows)[x*3+1];
                    arrB[k]=buffer_device.ptr(y+k*in_device.rows)[x*3];
                }
                old_disc_device.ptr(y)[x*3+2]=disc_device.ptr(y)[x*3+2];
                old_disc_device.ptr(y)[x*3+1]=disc_device.ptr(y)[x*3+1];
                old_disc_device.ptr(y)[x*3]=disc_device.ptr(y)[x*3];

                disc_device.ptr(y)[x*3+2]=sort_and_median(arrR, buff_size);
                disc_device.ptr(y)[x*3+1]=sort_and_median(arrG, buff_size);
                disc_device.ptr(y)[x*3]=sort_and_median(arrB, buff_size);
                k_device.ptr(y)[x]=0;
            }
            else
            {
                int k=k_device.ptr(y)[x];
                buffer_device.ptr(y+k*in_device.rows)[x*3+2]=in_device.ptr(y)[x*3+2];
                buffer_device.ptr(y+k*in_device.rows)[x*3+1]=in_device.ptr(y)[x*3+1];
                buffer_device.ptr(y+k*in_device.rows)[x*3]=in_device.ptr(y)[x*3];
                k_device.ptr(y)[x]++;
            }
        }
    }
    else
    {
        k_device.ptr(y)[x]=0;
    }
    __syncthreads();
}



__device__ int sort_and_median(int arr[], int length)
{
    int i, j, tmp;
    for (i = 1; i < length; i++)
    {
        j = i;
        while (j > 0 && arr[j - 1] > arr[j])
        {
            tmp = arr[j];
            arr[j] = arr[j - 1];
            arr[j - 1] = tmp;
            j--;
        }
    }
    return arr[length/2];
}
