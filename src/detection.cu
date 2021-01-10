#include <stdio.h>
#include <iostream>
#include "opencv4/opencv2/imgproc.hpp"
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <math.h>
#include <string>
#include <cuda/cuda_runtime.h>
#include <opencv4/opencv2/cudafeatures2d.hpp>
#include "opencv4/opencv2/imgproc/types_c.h"
#include "opencv4/opencv2/cudacodec.hpp"
#include "opencv4/opencv2/cudaimgproc.hpp"
#include "opencv4/opencv2/cudafilters.hpp"
#include "opencv4/opencv2/cudawarping.hpp"
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <cuda/device_launch_parameters.h>

using namespace cv;
using namespace std;
using namespace cv::cuda;

__global__ void trainKernel(cuda::PtrStepSz<uchar>in_device,cuda::PtrStepSz<uchar>buffer, int k); //fill initial buffer
__global__ void calculateDiscr(cuda::PtrStepSz<uchar>in_device, cuda::PtrStepSz<uchar>buffer,cuda::PtrStepSz<uchar>disc_device, const int buff_size,cuda::PtrStepSz<uchar>old_disc_device); //calculate initial backgorund
__global__ void changeDetection(cuda::PtrStepSz<uchar>in_device, cuda::PtrStepSz<uchar>buffer_device, cuda::PtrStepSz<uchar>disc_device, cuda::PtrStepSz<uchar>out_device, cuda::PtrStepSz<uchar>k_device, int buff_size,cuda::PtrStepSz<uchar>old_disc_device); //motion detection
__device__ int sort_and_median(int arr[], int length); //calculate new dicriminator for a pixel


int main( int argc, char** argv )
{
    system("clear");
    int scale=1;
    const int buff_size=150; //buffer size in numer of frame
    int TRAIN=true; //flag for training fase
    cudaEvent_t start, stop;
    float time;
    Mat img;
    cv::namedWindow("MyCameraPreview", cv::WINDOW_AUTOSIZE);
    const char* gst =  "rtspsrc location=rtsp://admin:@cam1/ch0_0.264 name=r latency=0 protocols=tcp ! application/x-rtp,payload=96,encoding-name=H264 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw(memory:NVMM), format=BGRx ! nvvidconv ! videoconvert ! video/x-raw, format=BGR, framerate=5/1 ! appsink";
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
    Mat inframe; //input frame
    Mat outframe(N, M, CV_8UC1, Scalar(0)); //output frame
    Mat disc(N, M, CV_8UC3, Scalar(0,0,0)); //background
    Mat old_disc(N, M, CV_8UC3, Scalar(0,0,0)); //old backgorund for comparison
    Mat resized (N, M, CV_8UC3, Scalar(0,0,0)); //resized image
    Mat k(N, M, CV_8UC1,Scalar(0)); //last buffer element pointer
    Mat buffer(N*buff_size,M, CV_8UC3 , Scalar(0,0,0)); //buffer for motion memorization
    Mat element = getStructuringElement( MORPH_RECT, Size(3, 3), Point( 1, 1) );
    cuda::GpuMat in_device, out_device, disc_device, buffer_device, k_device, old_disc_device, resized_device; //same Mats on Gpu
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
                    in_device.upload(inframe);
                    cuda::resize(in_device, resized_device, Size(M,N));
                    cv::Ptr<cv::cuda::Filter> filter = cuda::createGaussianFilter(resized_device.type(), resized_device.type(), Size(3, 3),0);
                                filter->apply(resized_device, resized_device);
                    filter -> apply(resized_device, resized_device);
                    trainKernel<<<M, N>>>(resized_device, buffer_device, y);
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
            in_device.upload(inframe); //carico frame input su GPU
            cuda::resize(in_device, resized_device, Size(M,N));
            cv::Ptr<cv::cuda::Filter> filter = cuda::createGaussianFilter(resized_device.type(), resized_device.type(), Size(3, 3),0);
            filter->apply(resized_device, resized_device);
            changeDetection<<<M, N>>>(resized_device, buffer_device, disc_device, out_device, k_device, buff_size, old_disc_device);
            filter = cuda::createMorphologyFilter(CV_MOP_OPEN, out_device.type(), element);
            filter->apply(out_device, out_device);
            Mat result_host(out_device);
            cout << "cols=" << disc_device.cols << endl;
            cout << "rows=" << disc_device.rows << endl;
            Mat back_host(disc_device);
            cv::imshow("MyCameraPreview",result_host);
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
