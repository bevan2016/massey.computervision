/*********************************************************************************************
 * compile with:
 * g++ -std=c++11 main.cpp -o reader `pkg-config --cflags --libs opencv`
 * test static image: ./reader Darwin_rotated_scaled.jpg
 ********************************************************************************************/

#include <stdio.h>
#include <ctime>
#include <iostream>
#include <sys/stat.h>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define DEBUG_QR_READER 1

using namespace std;
using namespace cv;

const char ENCODINGS[64]={' ',
                          'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','x','y','w','z',
                          'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','X','Y','W','Z',
                          '0','1','2','3','4','5','6','7','8','9',
                          '.'};
const int X_SIZE = 47;
const int Y_SIZE = 47;
const int CORNER_SIZE = 6;
const int TOLERANCE = 5;

const int QR_SIZE = X_SIZE*Y_SIZE - 3*CORNER_SIZE*CORNER_SIZE;
const int MESSAGE_SIZE = QR_SIZE / 2 + 1;

const char* ORIGIN_WND = "Original Image";
const char* UPRIGHT_WND = "Upright Image";

/*********************************************************************************************
 * utilities to access image data
*********************************************************************************************/
#define mpixel( image, x, y, channel) image.data[(y)*image.step[0]+(x)*image.step[1]+channel]

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

float dist(Vec3i& c0, Vec3i& c1)
{
    return sqrt((c0[0]-c1[0])*(c0[0]-c1[0]) + (c0[1]-c1[1])*(c0[1]-c1[1]));
}

/*********************************************************************************************
 * read bar code
*********************************************************************************************/
int is_upright(Vec3i* cc)
{
    /*
      0
      1  2
    */
    if (abs(cc[1][0] - cc[0][0]) < TOLERANCE
        && abs(cc[1][1] - cc[2][1]) < TOLERANCE
        && (cc[1][1] - cc[0][1]) > TOLERANCE
        && (cc[2][0] - cc[1][0]) > TOLERANCE)
        return 1;

    /*
      2
      1  0
    */
    if (abs(cc[1][0] - cc[2][0]) < TOLERANCE
        && abs(cc[1][1] - cc[0][1]) < TOLERANCE
        && (cc[1][1] - cc[2][1]) > TOLERANCE
        && (cc[0][0] - cc[1][0]) > TOLERANCE)
        return 1;

    return 0;
}

int rotate_image(Mat& originImg, Mat& uprightImg, Vec3i* cc)
{
    //imDetails(originImg);
    //detect the positioning circles
    Mat binImg;
    cvtColor(originImg, binImg, CV_BGR2GRAY);
    medianBlur(binImg, binImg, 5) ;

    vector<Vec3f> circles;
    HoughCircles(binImg, circles, CV_HOUGH_GRADIENT, 1, binImg.rows/8, 100, 30, 1, 30);
    if (circles.size() != 3)
    {
        cout << "Error: detected more than 3 positioning circles." << endl;
        return -1;
    }

    for (size_t i=0; i<circles.size(); i++)
    {
        Vec3i c = circles[i];
        circle(originImg, Point(c[0], c[1]), c[2], Scalar(0, 0, 255), 3, LINE_AA);
        circle(originImg, Point(c[0], c[1]), 1, Scalar(255, 0, 0), 3, LINE_AA);
        cc[i] = c;
    }

    //determine the center of the QR code
    float d0, d1, d2;
    float a, b, c;
    d0 = dist(cc[0], cc[1]);
    d1 = dist(cc[0], cc[2]);
    d2 = dist(cc[1], cc[2]);
    //find the circle center that should be in the left-bottom and store it at cc[1]
    if (d0 > d1 && d0 > d2)
    {
        Vec3i temp = cc[2];
        cc[2] = cc[1];
        cc[1] = temp;

        a = d0;
        b = d1;
        c = d2;
    }
    else if (d2 > d0 && d2 > d1)
    {
        Vec3i temp = cc[0];
        cc[0] = cc[1];
        cc[1] = temp;

        a = d2;
        b = d0;
        c = d1;

    }
    else {
        a = d1;
        b = d2;
        c = d0;
    }

    if (abs(a - sqrt(b*b+c*c)) > c/50)
    {
        cout << "Error: can not detect the positioning circles." << endl;
        return -1;
    }

    if (is_upright(cc))
    {
        namedWindow(ORIGIN_WND, CV_WINDOW_AUTOSIZE);
        imshow(ORIGIN_WND, originImg);
        originImg.copyTo(uprightImg);
        return 0;
    }

    //do rotate
    //center of the image
    float ox = (cc[0][0] + cc[2][0]) / 2.0;
    float oy = (cc[0][1] + cc[2][1]) / 2.0;

    float x1 = cc[1][0] - ox;
    float y1 = cc[1][1] - oy;

    float angle = 0;
    if (x1 >= 0 && y1 > 0)
        angle = 270;
    else if (x1 > 0 && y1 <= 0)
        angle = 180;
    else if (x1 <= 0 && y1 < 0)
        angle = 90;
    else
        angle = 0;

    float alpha = angle/180.0*M_PI;
    float x11 = x1*cos(alpha) + y1*sin(alpha);
    float y11 = (-x1)*sin(alpha) + y1*cos(alpha);
    if (abs(y11) < TOLERANCE/2.0) {
        angle += 45;
    }
    else if (abs(x11) < TOLERANCE/2.0) {
        angle -= 45;
    }
    else {
        float beta = atan(abs(y11/x11));
        angle = angle + 45 - beta*180/M_PI;
    }

    Mat rotmatrix = getRotationMatrix2D(Point((int)ox, (int)oy), angle, 1);
    warpAffine(originImg, uprightImg, rotmatrix, originImg.size());

#ifdef DEBUG_QR_READER
    cout << "Rotate " << angle << " degrees to be upright." << endl;
    namedWindow(ORIGIN_WND, CV_WINDOW_AUTOSIZE);
    circle(originImg, Point(ox, oy), 10, Scalar(0, 0, 255), 3, LINE_AA);
    imshow(ORIGIN_WND, originImg);

    namedWindow(UPRIGHT_WND, CV_WINDOW_AUTOSIZE);
    circle(uprightImg, Point(ox, oy), 10, Scalar(0, 0, 255), 3, LINE_AA);
    imshow(UPRIGHT_WND, uprightImg);
#endif // DEBUG_QR_READER

   return 0;
}

char read_code(Mat& qrImg, int x, int y)
{
    char result = 0;
    int bgr;
    int x0 = x - 1;
    int y0 = y - 1;
    for (int i=0; i<3; i++) //read 3 channels
    {
        bgr = 0;
        //read 9 cells and calculate the mean as the value for cell (x, y)
        for (int r=0; r<3; r++)
        {
            for (int c=0; c<3; c++)
            {
                bgr += mpixel(qrImg, x0+r, y0+c, i);
            }
        }

        if (bgr/9 < 128) //128 as the threshold for 0 or 255
            bgr = 0;
        else
            bgr = 1;

        result |= bgr << i;
    }

    return result;
}

void read_message(Mat& qrImg, Vec3i* cc, char* message)
{
    int rows = qrImg.rows;
    int cols = qrImg.cols;

    char buffer[QR_SIZE];
    float xc = (cc[0][0] + cc[2][0]) / 2.0;
    float yc = (cc[0][1] + cc[2][1]) / 2.0;
    float step = dist(cc[0], cc[1]) / (Y_SIZE-CORNER_SIZE);
    cout <<"the image center is (" << xc << ", " << yc << ") step=" << step << endl;
    float x0 = xc - step*(X_SIZE/2);
    float y0 = yc - step*(Y_SIZE/2);
    cout <<"read code from (" << x0 << ", " << y0 << ")"<< endl;

    int index = 0;
    for (int y=0; y<Y_SIZE; y++)
    {
        for (int x=0; x<X_SIZE; x++)
        {
            if ( (y < CORNER_SIZE && x < CORNER_SIZE)
              || (y >= (Y_SIZE-CORNER_SIZE) && x < CORNER_SIZE)
              || (y >= (Y_SIZE-CORNER_SIZE) && x >= (Y_SIZE-CORNER_SIZE)))
              continue;

            buffer[index++] = read_code(qrImg, (int)x0+x*step, (int)y0);
        }
        y0 += step;
    }

    for (int i=0; i<MESSAGE_SIZE-1; i++)
    {
        char code = buffer[i*2]<<3 | buffer[i*2+1];
        message[i] = ENCODINGS[code];
    }
}

void test()
{
    const char* imgs[] = {"img/abcde.jpg",
                          "img/abcde_rotated.jpg",
                          "img/abcde_rotated_scaled.jpg",
                          "img/abcde_scaled.jpg",
                          "img/congratulations.jpg",
                          "img/congratulations_rotated.jpg",
                          "img/congratulations_rotated_scaled.jpg",
                          "img/congratulations_scaled.jpg",
                          "img/Darwin.jpg",
                          "img/Darwin_rotated.jpg",
                          "img/Darwin_rotated_scaled.jpg",
                          "img/Darwin_scaled.jpg",
                          "img/farfaraway.jpg",
                          "img/farfaraway_rotated.jpg",
                          "img/farfaraway_rotated_scaled.jpg",
                          "img/farfaraway_scaled.jpg"};
    char message[MESSAGE_SIZE];
    for (int i=0; i<16; i++)
    {
        memset(message, 0, MESSAGE_SIZE);
        Mat barcodeImg = imread(imgs[i]);
        Vec3i cc[3];
        Mat uprightImg;
        rotate_image(barcodeImg, uprightImg, cc);
        read_message(uprightImg, cc, message);
        cout << imgs[i] << endl << message << endl<< endl;
    }
}

int main(int argc, char** argv)
{
    //uncomment this line to test with the course provided pictures.
    //test();return 0;
    char message[MESSAGE_SIZE];
    memset(message, 0, MESSAGE_SIZE);

    if (argc >= 2)
    {
        struct stat buffer;
        if (stat(argv[1], &buffer) != 0)
        {
            cout << "Error: could not access file " << argv[1] << endl;
            exit(0);
        }
        Mat barcodeImg = imread(argv[1]);
        if (! barcodeImg.data)
        {
            cout <<"open image error"<<endl;
            exit(0);
        }

        Vec3i cc[3];
        Mat uprightImg;
        rotate_image(barcodeImg, uprightImg, cc);
        read_message(uprightImg, cc, message);
    }
    else
    {
        cout << "Usage: ./reader Darwin_rotated_scaled.jpg" << endl;
        exit(0);
    }
    cout<<"The message is: "<<endl;
    cout<<message<<endl;

    waitKey(0);
    return 0;
}
