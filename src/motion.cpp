#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/cudafeatures2d.hpp>
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;

std::string exec(const char*);

int detectMotion(const cv::cuda::GpuMat & motion, Mat & result, Mat & result_cropped,
                 int x_start, int x_stop, int y_start, int y_stop,
                 int max_deviation,
                 Scalar & color, Scalar stddev)
{
    // calculate the standard deviation
    // if not to much changes then the motion is real (neglect agressive snow, temporary sunlight)


    if(stddev[0] > 5 and  stddev[0] < 35)
    {
    	int number_of_changes = cv::cuda::countNonZero(motion);
//        int number_of_changes = 0;
//        int min_x = motion.cols, max_x = 0;
//        int min_y = motion.rows, max_y = 0;
//        // loop over image and detect changes
//        for(int j = 0; j < motion.rows; j+=2){ // height
//            for(int i = 0; i < motion.cols; i+=2){ // width
//                // check if at pixel (j,i) intensity is equal to 255
//                // this means that the pixel is different in the sequence
//                // of images (prev_frame, current_frame, next_frame)
//                if(static_cast<int>(motion.at<uchar>(j,i)) == 255)
//                {
//                    number_of_changes++;
//                    if(min_x>i) min_x = i;
//                    if(max_x<i) max_x = i;
//                    if(min_y>j) min_y = j;
//                    if(max_y<j) max_y = j;
//                }
//
//                if(number_of_changes > 300) {
//                	break;
//                }
//            }
//        }
//        if(number_of_changes){
//            //check if not out of bounds
//            if(min_x-10 > 0) min_x -= 10;
//            if(min_y-10 > 0) min_y -= 10;
//            if(max_x+10 < result.cols-1) max_x += 10;
//            if(max_y+10 < result.rows-1) max_y += 10;
//            // draw rectangle round the changed pixel
//            Point x(min_x,min_y);
//            Point y(max_x,max_y);
//            Rect rect(x,y);
//            Mat cropped = result(rect);
//            cropped.copyTo(result_cropped);
//            rectangle(result,rect,color,1);
//        }
        return number_of_changes;
    }
    return 0;
}


inline void directoryExistsOrCreate(const char* pzPath)
{
    DIR *pDir;
    // directory doesn't exists -> create it
    if ( pzPath == NULL || (pDir = opendir (pzPath)) == NULL)
        mkdir(pzPath, 0777);
    // if directory exists we opened it and we
    // have to close the directory again.
    else if(pDir != NULL)
        (void) closedir (pDir);
}

int incr = 0;
bool saveImg(Mat image, const string DIRECTORY, const string EXTENSION, const char * DIR_FORMAT, const char * FILE_FORMAT)
{

//	vector<int> compression_params;
//	compression_params.push_back(IMWRITE_JPEG_QUALITY);
//	compression_params.push_back(100);

    stringstream ss, copyCommand, rmCommand;
    time_t seconds;
    struct tm * timeinfo;
    char TIME[80];
    time (&seconds);
    // Get the current time
    timeinfo = localtime (&seconds);

    // Create name for the date directory
    strftime (TIME,80,DIR_FORMAT,timeinfo);
    ss.str("");
    ss << DIRECTORY << TIME;
    directoryExistsOrCreate(ss.str().c_str());
    ss << "/cropped";
    directoryExistsOrCreate(ss.str().c_str());

    // Create name for the image
    strftime (TIME,80,FILE_FORMAT,timeinfo);
    ss.str("");
    if(incr < 100) incr++; // quick fix for when delay < 1s && > 10ms, (when delay <= 10ms, images are overwritten)
    else incr = 0;
    ss << DIRECTORY << TIME << static_cast<int>(incr) << EXTENSION;
    cout << "image location=" << ss.str().c_str() << endl;
    bool status = imwrite(ss.str().c_str(), image);

    copyCommand << "/usr/bin/gsutil cp " <<  ss.str().c_str() << "gs://sheviz-home-automation/security-camera/" << ss.str().c_str() ;
    rmCommand << "rm -f "  <<  ss.str().c_str();
    exec(copyCommand.str().c_str());
    //exec(rmCommand.str().c_str());
    cout << "copy=" << copyCommand.str() << "rm=" << rmCommand.str() ;

    return status;
}




