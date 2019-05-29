/*********************************************************************************************
 * compile with:
 * g++ -std=c++11 blobcounter.cpp -o blob_counter `pkg-config --cflags --libs opencv`
 * for static image: ./blob_counter saltandpepper_11.jpg
 * for video: ./blob_counter
*********************************************************************************************/

#include <stdio.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <sys/stat.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace chrono;

const int KERNEL_SIZE = 9;
const int FILTER_SIZE = 256;

const int FRAME_WIDTH  = 640;
const int FRAME_HEIGHT = 480;

const int MAX_SET_SIZE = 1000;

const char* ORIGIN_WND = "Original Image";
const char* DENOISED_WND = "Median Filtered Image";
const char* COUNTING_WND = "Binary Image";
/*********************************************************************************************
 * utilities to access image data
*********************************************************************************************/
#define MpixelB( image, x, y) image.data[(y)*image.step[0]+(x)*image.step[1]]
#define MpixelG( image, x, y) image.data[(y)*image.step[0]+(x)*image.step[1]+1]
#define MpixelR( image, x, y) image.data[(y)*image.step[0]+(x)*image.step[1]+2]
#define Mpixel( image, x, y) image.data[(y)*image.step[0] + x]

void imDetails(Mat& img)
{
    cout << "width = " << img.cols << " height = " << img.rows << endl;
    cout << "depth = " << img.depth() << " channels = " << img.channels() << endl;
    cout << "steps: " << img.step[0] << ", " << img.step[1] << endl;
    cout << "Image type ";
    if (CV_8UC3 == img.type())
        cout << "CV_8UC3";
    else if (CV_8UC2 == img.type())
        cout << "CV_8UC2";
    else if (CV_8UC1 == img.type())
        cout << "CV_8UC1";
    else
        cout << img.type();
    cout << endl;
}

long getTime(system_clock::time_point start) {
    system_clock::time_point end = system_clock::now(); //microseconds
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

/*********************************************************************************************
 * find median of an array
 Example:
   int ary[9] = {23, 1, 3, 4, 0, 12, 45, 11, 3};
   median(ary, 9);
*********************************************************************************************/
int median(int* ary,  int size)
{
    int i = 1;
    for (; i<size; i++) {
        int j = i;
        while (j > 0 && ary[j] < ary[j-1]) {
            int v = ary[j];
            ary[j] = ary[j-1];
            ary[j-1] = v;
            j--;
        }
    }
    return ary[size/2];
}

/*********************************************************************************************
 * apply the median filter
*********************************************************************************************/
void apply_median_filter(Mat& src, Mat& target)
{
    int filter[FILTER_SIZE];
    int pixels[KERNEL_SIZE];
    int rows = src.rows;
    int cols = src.cols;

    for (int y=0; y<rows; y++)
    {
        for (int x=0; x<cols; x++)
        {
            if (x == 0)
            {
                //left
                pixels[0] = 0;//Mpixel(src, 0, y_1);
                pixels[1] = 0;//Mpixel(src, 0, y);
                pixels[2] = 0;//Mpixel(src, 0, y1);
                //middle
                pixels[3] = (y-1 >= 0)?Mpixel(src, 1, y-1):0;
                pixels[4] = Mpixel(src, 1, y);
                pixels[5] = (y+1 < rows)?Mpixel(src, 1, y+1):0;
                //right
                pixels[6] = (y-1 >= 0)?Mpixel(src, 2, y-1):0;
                pixels[7] = Mpixel(src, 2, y);
                pixels[8] = (y+1 < rows)?Mpixel(src, 2, y+1):0;
            }
            else
            {
                memcpy(&pixels[0], &pixels[3], sizeof(int)*3);
                memcpy(&pixels[3], &pixels[6], sizeof(int)*3);
                //right
                if ((x+1 < cols) && (y-1) >=0)
                    pixels[6] = Mpixel(src, x+1, y-1);
                else
                    pixels[6] = 0;

                pixels[7] = (x+1 < cols)?Mpixel(src, x+1, y):0;
                if ((x+1 < cols) && (y+1 < rows))
                    pixels[8] = Mpixel(src, x+1, y+1);
                else
                    pixels[8] = 0;
            }
            //apply median filter
            int v = 0;
            //v = median(filter, FILTER_SIZE);
            memset(filter, 0, sizeof(int)*FILTER_SIZE);
            for (int k=0; k<KERNEL_SIZE; k++)
            {
                filter[pixels[k]] += 1;
            }

            for (int i=0; i<FILTER_SIZE; i++)
            {
                v += filter[i];
                if (v > 4)
                {
                    v = i;
                    break;
                }

            }

            Mpixel(target, x, y) = v;
        }
    }
}

vector<vector<unsigned int>*> blobSet;
int countBlobs(Mat& binImg, const int pixelsThreshold)
{
    short rows = binImg.rows;
    short cols = binImg.cols;

    int A[cols][rows];
    //initialize au
    for (short c=0; c<cols; c++) {
        for (short r=0; r<rows; r++) {
            A[c][r] = -1;
        }
    }

    unsigned int coord = 0;
    int counter, up, left, s1, s2;
    counter = 0;
    for (short y=0; y<rows; y++) {
        for (short x=0; x<cols; x++) {
            int v = MpixelB(binImg, x, y);
            if (v == 0) {
                continue;
            }

            up = (y == 0)?0:MpixelB(binImg, x, y-1);
            left = (x == 0)?0:MpixelB(binImg, x-1, y);
            if (up != 0 || left != 0) {
                s1 = (x==0)?-1:A[x-1][y];
                s2 = (y==0)?-1:A[x][y-1];
                if (s1 != -1) {
                    coord = (x << 16) + y;
                    blobSet[s1]->push_back(coord);
                    A[x][y] = s1;
                }
                if (s2 != -1) {
                    coord = (x << 16) + y;
                    blobSet[s2]->push_back(coord);
                    A[x][y] = s2;
                }
                if ((s1 != s2) && (s1 != -1) && (s2 != -1)) {
                    vector<unsigned int>* nSet1 = blobSet[s1];
                    vector<unsigned int>* nSet2 = blobSet[s2];
                    for (int i=0; i<nSet2->size(); i++) {
                        unsigned int p = (*nSet2)[i];
                        nSet1->push_back(p);
                        A[p>>16][p&0xffff] = s1;
                    }
                    nSet2->clear();
                }
            }
            else {
                if(counter < MAX_SET_SIZE)
                {
                    coord = (x << 16) + y;
                    blobSet[counter]->push_back(coord);
                    A[x][y] = counter++;
                }
            }
        }
    }
    int blobs = 0;
    for (int i=0; i<counter; i++){
        vector<unsigned int>* nSet = blobSet[i];
        if (nSet->size() > pixelsThreshold) {
            blobs++;
        }
        nSet->clear();
    }
    return blobs;
}

int countImageBlob(char* imgFile)
{
    //display the original image
    Mat originImg = imread(imgFile, CV_LOAD_IMAGE_GRAYSCALE);
    //imDetails(originImg);
    namedWindow(ORIGIN_WND, CV_WINDOW_AUTOSIZE);
    imshow(ORIGIN_WND, originImg);

    //apply median-filter to denoise the image
    Mat tmpImg;
    tmpImg.create(originImg.size(), originImg.type());
    apply_median_filter(originImg, tmpImg);

    //if we perform one more time median filter, the image will be perfect.
    Mat clearImg;
    clearImg.create(tmpImg.size(), tmpImg.type());
    //system_clock::time_point s1 = system_clock::now();
    apply_median_filter(tmpImg, clearImg);
    namedWindow(DENOISED_WND, CV_WINDOW_AUTOSIZE);
    imshow(DENOISED_WND, clearImg);

    //conver to a binary image
    Mat binImg;
    threshold(clearImg, binImg, 0, 255, CV_THRESH_BINARY|CV_THRESH_OTSU); //CV_THRESH_BINARY |

    //count blob
    int blobs = countBlobs(binImg, 130);
    char text[20];
    sprintf(text, "%d", blobs);
    putText(binImg, text, cvPoint(10, 30), FONT_HERSHEY_PLAIN, 2, cvScalar(172, 172, 172), 1);
    cout << text << endl;

    namedWindow(COUNTING_WND, CV_WINDOW_AUTOSIZE);
    imshow(COUNTING_WND, binImg);

    waitKey(0);
    return blobs;
}

int countVideoBlob(VideoCapture& cap)
{
    namedWindow(ORIGIN_WND, CV_WINDOW_AUTOSIZE);
    namedWindow(DENOISED_WND, CV_WINDOW_AUTOSIZE);
    namedWindow(COUNTING_WND, CV_WINDOW_AUTOSIZE);

    Mat frame;
    Mat grayFrame;
    Mat tmpFrame;
    Mat binFrame;
    char text[20];

    tmpFrame.create(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1);
    while(true)
    {
        system_clock::time_point s0 = system_clock::now();

        cap >> frame;
        if( frame.empty() ) {
            break;
        }
        int t0 = getTime(s0);
        imshow(ORIGIN_WND, frame);

        // IMPORTANT
        continue;
        //

        cvtColor(frame, grayFrame, CV_BGR2GRAY);

        system_clock::time_point s1 = system_clock::now();
        apply_median_filter(grayFrame, tmpFrame);
        int t1 = getTime(s1);

        imshow(DENOISED_WND, tmpFrame);
        threshold(tmpFrame, binFrame, 0, 255, CV_THRESH_BINARY|CV_THRESH_OTSU);
        cout << getTime(s0) << " " << t0 << " " << t1 << endl;
        //count blob
        int blobs = countBlobs(binFrame, 100);
        sprintf(text, "%d fps: %ld", blobs, 1000/getTime(s0));
        putText(binFrame, text, cvPoint(10, 30), FONT_HERSHEY_PLAIN, 2, cvScalar(172, 172, 172), 1);

        imshow(COUNTING_WND, binFrame);

        int key = waitKey(1);
        if (key == 113 || key == 27) return 0;//either esc or 'q'
    }

    return 0;
}
/* Test with all the images provide on the course stream */
void test()
{
    const int IMG_NUM = 70;
    const char* imgs[IMG_NUM] = {
        "bolts_nuts_1.jpg",
        "bolts_nuts_3.jpg",
        "bolts_nuts_5.jpg",
        "bolts_nuts_7.jpg",
        "bolts_nuts_9.jpg",
        "bolts_nuts_11.jpg",
        "bolts_nuts_13.jpg",
        "bolts_nuts_15.jpg",
        "bolts_nuts_17.jpg",
        "bolts_nuts_19.jpg",
        "bolts_nuts_21.jpg",
        "bolts_nuts_23.jpg",
        "bolts_nuts_25.jpg",
        "bolts_nuts_27.jpg",
        "bolts_nuts_29.jpg",
        "bolts_nuts_31.jpg",
        "bolts_nuts_33.jpg",
        "bolts_nuts_35.jpg",
        "bolts_nuts_37.jpg",
        "bolts_nuts_39.jpg",
        "bolts_nuts_41.jpg",
        "bolts_nuts_43.jpg",
        "bolts_nuts_45.jpg",
        "bolts_nuts_47.jpg",
        "bolts_nuts_49.jpg",
        "saltandpepper_1.jpg",
        "saltandpepper_3.jpg",
        "saltandpepper_5.jpg",
        "saltandpepper_7.jpg",
        "saltandpepper_9.jpg",
        "saltandpepper_11.jpg",
        "saltandpepper_13.jpg",
        "saltandpepper_15.jpg",
        "saltandpepper_17.jpg",
        "saltandpepper_19.jpg",
        "saltandpepper_21.jpg",
        "saltandpepper_23.jpg",
        "saltandpepper_25.jpg",
        "saltandpepper_27.jpg",
        "saltandpepper_29.jpg",
        "saltandpepper_31.jpg",
        "saltandpepper_33.jpg",
        "saltandpepper_35.jpg",
        "saltandpepper_37.jpg",
        "saltandpepper_39.jpg",
        "saltandpepper_41.jpg",
        "saltandpepper_43.jpg",
        "saltandpepper_45.jpg",
        "saltandpepper_47.jpg",
        "saltandpepper_49.jpg",
        "saltandpepper_51.jpg",
        "saltandpepper_53.jpg",
        "saltandpepper_55.jpg",
        "saltandpepper_57.jpg",
        "saltandpepper_59.jpg",
        "saltandpepper_61.jpg",
        "saltandpepper_63.jpg",
        "saltandpepper_65.jpg",
        "saltandpepper_67.jpg",
        "saltandpepper_69.jpg",
        "stationary_1.jpg",
        "stationary_3.jpg",
        "stationary_5.jpg",
        "stationary_7.jpg",
        "stationary_9.jpg",
        "stationary_11.jpg",
        "stationary_13.jpg",
        "stationary_15.jpg",
        "stationary_17.jpg",
        "stationary_19.jpg" };
    int answer[IMG_NUM] = {10, 10, 10, 10, 10,
                    7, 8, 5, 7 ,6,
                    10, 10, 10, 10, 10,
                    10, 7, 6, 7, 8,
                    6, 4, 2, 10, 10,
                    10, 10, 10, 10, 10,
                    7, 8, 5, 7 ,6,
                    10, 10, 10, 10, 10,
                    10, 7, 6, 7, 8,
                    6, 4, 2, 10, 10,
                    3, 3, 5, 1, 3,
                    5, 7, 2, 4, 6,
                    4, 3, 5, 1, 3,
                    5, 7, 2, 4, 6};
    char fileName[100];
    for (int i=0; i<IMG_NUM; i++)
    {
        cout << imgs[i];
        memset(fileName, 0 , 100);
        sprintf(fileName, "img/%s", imgs[i]);
        Mat originImg = imread(fileName);

        Mat binImg;
        cvtColor(originImg, binImg, CV_BGR2GRAY);
        //apply median-filter to denoise the image
        Mat tmpImg;
        tmpImg.create(binImg.size(), binImg.type());
        apply_median_filter(binImg, tmpImg);

        Mat clearImg;
        clearImg.create(tmpImg.size(), tmpImg.type());
        apply_median_filter(tmpImg, clearImg);


        threshold(clearImg, binImg, 0, 255, CV_THRESH_OTSU); //CV_THRESH_BINARY |

        int blobs = countBlobs(binImg, 130);
        cout << ":  expect " << answer[i] << " " << blobs << endl;
    }
}

int main(int argc, char** argv)
{
    for (int i=0; i<MAX_SET_SIZE; i++)
    {
        blobSet.push_back(new vector<unsigned int>());
    }

    //test();    return 0;
    if (argc >= 2)
    {
        struct stat buffer;
        if (stat(argv[1], &buffer) != 0)
        {
            cout << "Error: could not access file " << argv[1] << endl;
            exit(0);
        }
        countImageBlob(argv[1]);
    }
    else
    {
        VideoCapture cap;
        cap.open(0);
        if ( !cap.isOpened())
        {
            cout << "Error: could not open camera" << endl;
            exit(0);
        }
        cap.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
        cout << "Opened camera" << endl;

        countVideoBlob(cap);
    }

    return 0;
}
