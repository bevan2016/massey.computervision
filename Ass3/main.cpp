/*********************************************************************************************
 * compile with:
 * g++ -std=c++11 main.cpp -o gr `pkg-config --cflags --libs opencv`
 * for static image: ./gr hand_img.jpg
 * for video: ./gr
*********************************************************************************************/

#include <stdio.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace chrono;

//Macros for colour pixels
#define pixelB( image, x, y) image.data[(y)*image.step[0]+(x)*image.step[1]]
#define pixelG( image, x, y) image.data[(y)*image.step[0]+(x)*image.step[1]+1]
#define pixelR( image, x, y) image.data[(y)*image.step[0]+(x)*image.step[1]+2]

const int FEATURE_NUMBER = 15;
Ptr<ANN_MLP> g_model;

const int FRAME_WIDTH  = 640;
const int FRAME_HEIGHT = 480;

const char* ORIGIN_WND = "Origin";
const char* CONTOUR_WND = "Contour";
const char* CAMERA_WND = "Camera";
/*********************************************************************************************
 * utilities to access image data
*********************************************************************************************/
long getTime(system_clock::time_point start) {
    system_clock::time_point end = system_clock::now(); //microseconds
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

string get_label(string& fileName)
{
    size_t sIndex =  fileName.find('_');
    size_t eIndex =  fileName.find('_', sIndex+1);
    return fileName.substr(sIndex+1, eIndex-sIndex-1);
}

int file_exist (const string& filename)
{
  struct stat   buffer;
  const char * f = filename.c_str();
  return (stat (f, &buffer) == 0);
}
/*********************************************************************************************
 * create train data from the images
*********************************************************************************************/
void elliptic_fd(vector<Point>& contour, vector<float>& CE)
{
    vector<float> ax, ay, bx, by;
    int m = contour.size();
    float t = (2*M_PI)/m;
    for( int k=0; k<FEATURE_NUMBER+1; k++)
    {
        ax.push_back(0.0);
        ay.push_back(0.0);
        bx.push_back(0.0);
        by.push_back(0.0);
        for (int i=0; i<m; i++)
        {
            float rad = (k+1)*t*i;
            ax[k] += contour[i].x * cos(rad);
            bx[k] += contour[i].x * sin(rad);
            ay[k] += contour[i].y * cos(rad);
            by[k] += contour[i].y * sin(rad);
        }
        ax[k] /= m;
        ay[k] /= m;
        bx[k] /= m;
        by[k] /= m;
    }

    float a0 = ax[0]*ax[0] + ay[0]*ay[0];
    float b0 = bx[0]*bx[0] + by[0]*by[0];
    //ignore the first since it is always 2
    for (int k=1; k<FEATURE_NUMBER+1; k++)
    {
        CE.push_back(sqrt((ax[k]*ax[k] + ay[k]*ay[k])/a0) +sqrt((bx[k]*bx[k] + by[k]*by[k])/b0));
    }
}

int fourier_descriptor(Mat& img, vector<float>& features)
{
    Mat grayImg;
    Mat mask(img.rows, img.cols, CV_8UC1);
    cvtColor(img, grayImg, CV_BGR2GRAY); //CV_BGR2GRAY CV_BGR2HSV
    threshold(grayImg, mask, 5, 255, CV_THRESH_BINARY);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3));
    erode(mask, mask, kernel);
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    dilate(mask, mask, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);

    vector<vector<Point> > allContours, contours;
    vector<Vec4i> hierarchy;

    findContours( mask, allContours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());

    int index = 0;
    int largest = 0;
    for(size_t k=0; k<allContours.size(); k++) {
        if (largest < allContours[k].size()) {
            largest = allContours[k].size();
            index = k;
        }
    }
    contours.push_back(allContours[index]);
    approxPolyDP(Mat(allContours[index]), contours[0], 3, true);

#if 0
    //draw contours on a black image
    Mat contourImg = Mat::zeros(img.rows, img.cols, CV_8UC3);
    drawContours(contourImg, contours, 0, Scalar(0, 255, 0), 1, 8);
    namedWindow(CONTOUR_WND, CV_WINDOW_AUTOSIZE);
    imshow(CONTOUR_WND, contourImg);
    waitKey(0);
#endif

    elliptic_fd(allContours[index], features);
    return 0;
}

int create_train_data(const char* folder, const char* dataFile)
{
    char impath[64];
    vector<float> features;

    struct dirent *entry;
    DIR *dir = opendir(folder);
    ofstream ofs;
    ofs.open(dataFile, ios::app);
    while ((entry = readdir(dir)) != NULL) {
        string fname = entry->d_name;
        string label = get_label(fname);
        memset(impath, 0, 64);
        sprintf(impath, "%s/%s", folder, entry->d_name);
        cout << impath << endl;
        Mat img = imread(impath);
        if (! img.data)
            continue;
        features.clear();
        fourier_descriptor(img, features);

        ofs << label;
        for (int i=0; i<features.size(); i++)
            ofs << "," << features[i];
        ofs << endl;
    }
    ofs.close();
    closedir(dir);
    return 0;
}

int load_train_data(const string& filename, Mat& data, Mat& responses)
{
    Mat el_ptr(1, FEATURE_NUMBER, CV_32F);
    vector<int> labels;

    data.release();
    responses.release();
    FILE* f = fopen( filename.c_str(), "rt" );
    if( !f )
    {
        cout << "Could not read the train data " << filename << endl;
        return -1;
    }

    const int BUFFER_SIZE = 512;
    char buf[BUFFER_SIZE+1];
    int i;
    for(;;)
    {
        char* ptr;
        if( !fgets( buf, BUFFER_SIZE, f ) )
            break;

        int resp = buf[0] - 48;
        //cout << "responses " << resp << " " ;

        ptr = buf;
        for( i = 0; i < FEATURE_NUMBER; i++ )
        {
            ptr = strchr( ptr, ',' ) + 1;
            sscanf( ptr, "%f", &el_ptr.at<float>(i));
        }
        //cout << el_ptr << endl;
        if( i < FEATURE_NUMBER )
            break;

        labels.push_back(resp);
        data.push_back(el_ptr);
    }
    fclose(f);
    Mat(labels).copyTo(responses);
    return 0;
}

int load_classifier(const string& filename_to_load)
{
    // load classifier from the specified file
    if (file_exist(filename_to_load)) {
        g_model = StatModel::load<ANN_MLP>( filename_to_load );
        if( ! g_model.empty() )
        {
            cout << "The classifier " << filename_to_load << " is loaded.\n";
            return 0;
        }
        cout << "Error: Could not read the classifier " << filename_to_load << endl;
    }
    else
        cout <<"Error: The classifier " <<filename_to_load <<" is not exist.\n";

    return -1;
}

void test_classifier(const Ptr<StatModel>& model,
                     const Mat& data, const Mat& responses)
{
    int i, nsamples_all = data.rows;
    int training_correct_predict = 0;
    for( i = 0; i < nsamples_all; i++ )
    {
        Mat sample = data.row(i);
        //cout << "Sample: " << responses.at<int>(i)-48 << " row " << data.row(i) << endl;
        float r = model->predict( sample );
        //cout << "Predict:  r = " << r << endl;
        if( (int)r == responses.at<int>(i) )
            training_correct_predict++;
    }
    printf("ntrain_samples %d, training_correct_predict %d \n", nsamples_all, training_correct_predict);
    printf("\nTest Recognition rate: training set = %.1f%% \n\n", training_correct_predict*100.0/nsamples_all);
}

int build_mlp_classifier( const string& train_data_filename, const string& model_filename)
{
    const int class_count = 10;
    Mat data;
    Mat responses;

    int result = load_train_data(train_data_filename, data, responses);
    if( result != 0 )
        return result;

    int nsamples_all = data.rows;
    int ntrain_samples = (int)(nsamples_all*1.0);//SPLIT

    Mat train_data = data.rowRange(0, ntrain_samples);
    Mat train_responses = Mat::zeros( ntrain_samples, class_count, CV_32F );

    // 1. unroll the responses
    cout << "Unrolling the responses...\n";
    for( int i = 0; i < ntrain_samples; i++ )
    {
        int cls_label = responses.at<int>(i);
        train_responses.at<float>(i, cls_label) = 1.f;
    }

    // 2. train classifier: 4 layers [19, 100, 100, 10]
    int layer_sz[] = { data.cols, 100, 100, class_count };
    int nlayers = (int)(sizeof(layer_sz)/sizeof(layer_sz[0]));
    Mat layer_sizes( 1, nlayers, CV_32S, layer_sz );

    Ptr<TrainData> tdata = TrainData::create(train_data, ROW_SAMPLE, train_responses);

    cout << "Training the classifier (may take a few minutes)...\n";
    g_model = ANN_MLP::create();
    g_model->setLayerSizes(layer_sizes);
    g_model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
    g_model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 800, 0));
    g_model->setTrainMethod(ANN_MLP::BACKPROP, 0.001);
    g_model->train(tdata);
    cout << endl;

    g_model->save(model_filename);

    test_classifier(g_model, data, responses);
    return true;
}
/*********************************************************************************************/
void threshold_hsv(Mat& img, Mat& output)
{
    const int UPH = 121;
    const int LOH = 70;
    const int UPS = 255;
    const int LOS = 70;
    const int UPV = 255;
    const int LOV = 98;

    Mat imgHsv;
    cvtColor(img, imgHsv, CV_RGB2HSV);
    output = Mat::zeros(img.rows, img.cols, CV_8UC1);

	for (int x=0; x<imgHsv.cols; x++)
	{
		for (int y=0; y<imgHsv.rows; y++)
		{
            uchar v = pixelR(imgHsv,x,y);
            uchar s = pixelG(imgHsv,x,y);
            uchar h = pixelB(imgHsv,x,y);
			if( v < LOV || v > UPV || s < LOS || s > UPS || h < LOH || h > UPH )
			{
				//pixelR(output, x, y)=0;
				//pixelG(output, x, y)=0;
				pixelB(output, x, y)=0;
			}
			else
			{
				//pixelR(output, x, y)=255;
				//pixelG(output, x, y)=255;
				pixelB(output, x, y)=255;
			}
		}
	}
}

/*********************************************************************************************/
int img_gesture(Mat& img, Mat& contourImg)
{
    Mat mask;
    threshold_hsv(img, mask);
    namedWindow("Threshold", CV_WINDOW_AUTOSIZE);
    imshow("Threshold", mask);

    vector<vector<Point> > allContours, contours;
    vector<Vec4i> hierarchy;
    vector<float> features;

    findContours( mask, allContours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());

    int index = 0;
    int largest = 0;
    for(size_t k=0; k<allContours.size(); k++) {
        if (largest < allContours[k].size()) {
            largest = allContours[k].size();
            index = k;
        }
    }

    contours.push_back(allContours[index]);
    approxPolyDP(Mat(allContours[index]), contours[0], 3, true);
    //draw contours on a black image
    elliptic_fd(allContours[index], features);

    contourImg.release();
    contourImg = Mat::zeros(img.rows, img.cols, CV_8UC3);
    drawContours(contourImg, contours, 0, Scalar(0, 255, 0), 1, 8);  //draw contour

    Mat hand = Mat(Size(features.size(),1),CV_32FC1,(void*)&features[0]).clone();

    float r = g_model->predict( hand );

    return (int)r;
}

int test_img(const char* folder)
{
#if 1
    namedWindow(ORIGIN_WND, CV_WINDOW_AUTOSIZE);
    namedWindow(CONTOUR_WND, CV_WINDOW_AUTOSIZE);
#endif
    Mat contourImg;

    char impath[64];

    struct dirent *entry;
    DIR *dir = opendir(folder);
    ofstream ofs;
    ofs.open(folder, ios::app);

    int correct = 0;
    int total = 0;
    while ((entry = readdir(dir)) != NULL) {
        string fname = entry->d_name;
        int label = entry->d_name[0] - 48;
        memset(impath, 0, 64);
        sprintf(impath, "%s/%s", folder, entry->d_name);
        cout<< impath << endl;
        Mat img = imread(impath);
        if(img.data )
        {
            total++;
            int number = img_gesture(img, contourImg);
            if (number == label)
                correct ++;

            char text[16];
            memset(text, 0, 16);
            sprintf(text,"%d / %d", label, number);
            putText(img, text, cvPoint(10,30), FONT_HERSHEY_PLAIN,2, cvScalar(0,0,255),2,8);
#if 1
            imshow(ORIGIN_WND, img);
            imshow(CONTOUR_WND, contourImg);
            waitKey(0);
#endif
        }
    }
    ofs.close();
    closedir(dir);
    cout << "Testing accuracy: " << correct << " / " << total << endl;
    return 0;
}

void video_gesture(VideoCapture& cap)
{
    namedWindow(CAMERA_WND, CV_WINDOW_AUTOSIZE);
    namedWindow(CONTOUR_WND, CV_WINDOW_AUTOSIZE);
    char text[32];

    double fps=0.0;
    Mat frame;
    Mat contourImg;
    while (1)
    {
        system_clock::time_point stime = system_clock::now();

        frame.release();
        contourImg.release();

        cap >> frame;
        if (frame.empty())
            continue;
        int number = img_gesture(frame, contourImg);
        fps = 1000/getTime(stime);

        memset(text, 0, 32);
        sprintf(text,"Number: %d, fps:%.1f", number, fps);
        putText(frame, text, cvPoint(10,30), FONT_HERSHEY_PLAIN,2, cvScalar(0,0,255),2,8);
        imshow(CAMERA_WND, frame);
        imshow(CONTOUR_WND, contourImg);

        int key = waitKey(1);
        if(key == 113 || key == 27)
            break;
    }
}

int main(int argc, char** argv)
{
    const string model_file = "mlp_model.xml";

#if 0 //create train data and build model
    const string train_data = "train.data";
    create_train_data("train", "train.data");
    build_mlp_classifier(train_data, model_file);
    return 0;
#endif

    //load classifier
    if (load_classifier(model_file) != 0)
    {
        cout << "Could not load model file. Train model first." << endl;
        exit(0);
    }

#if 0 //test with testing image
    test_img("img");
    return 0;
#endif

    if (argc >= 2)
    {
        struct stat buffer;
        if (stat(argv[1], &buffer) != 0)
        {
            cout << "Error: could not access file " << argv[1] << endl;
            exit(0);
        }
        Mat img = imread(argv[1]);
        if (! img.data)
        {
            cout <<"Load image error"<<endl;
            exit(0);
        }
        Mat contourImg;
        int number = img_gesture(img, contourImg);
        char text[16];
        memset(text, 0, 16);
        sprintf(text,"%d", number);
        putText(img, text, cvPoint(10,30), FONT_HERSHEY_PLAIN,2, cvScalar(0,0,255),2,8);

        namedWindow(ORIGIN_WND, CV_WINDOW_AUTOSIZE);
        namedWindow(CONTOUR_WND, CV_WINDOW_AUTOSIZE);
        imshow(ORIGIN_WND, img);
        imshow(CONTOUR_WND, contourImg);
        waitKey(0);
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

        video_gesture(cap);
    }
    return 0;
}
