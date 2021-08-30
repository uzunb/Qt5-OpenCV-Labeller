#include "mainwindow.h"
#include "ui_mainwindow.h"

#include<QFileDialog>
#include<QMessageBox>

#include <iostream>
#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;


Mat static markerMask, img0, img, imgGray, result;
Point static prevPt(-1, -1);
int static c;
QString static fileName;


static void onMouse( int event, int x, int y, int flags, void* )
{
    if( x < 0 || x >= img.cols || y < 0 || y >= img.rows )
        return;
    if( event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON) )
        prevPt = Point(-1,-1);
    else if( event == EVENT_LBUTTONDOWN )
        prevPt = Point(x,y);
    else if( event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON) )
    {
        Point pt(x, y);
        if( prevPt.x < 0 )
            prevPt = pt;
        line( markerMask, prevPt, pt, Scalar::all(255), 5, 8, 0 );
        line( img, prevPt, pt, Scalar::all(255), 5, 8, 0 );
        prevPt = pt;
        imshow("image", img);
    }
}


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    MainWindow::connect(ui->exitPushButton,SIGNAL(clicked()), qApp,SLOT(quit()));
    c = waitKey(0);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_loadPushButton_clicked()
{
    fileName = QFileDialog::getOpenFileName(this, ("Open File"),QDir::homePath(),("Images (*.png *.jpeg *.jpg)"));
    if(!fileName.isNull()) {
        img0 = imread(fileName.toStdString(), 1);
        if( img0.empty() )
        {
            QMessageBox::information(this, tr("Couldn'g open image"),tr("Sorry !"));
        }
        namedWindow( "image", WINDOW_AUTOSIZE );
        img0.copyTo(img);
        cvtColor(img, markerMask, COLOR_BGR2GRAY);
        cvtColor(markerMask, imgGray, COLOR_GRAY2BGR);
        markerMask = Scalar::all(0);
        imshow( "image", img );
        setMouseCallback( "image", onMouse, 0 );

        ui->resetPushButton->setEnabled(true);
        ui->doPushButton->setEnabled(true);
    } else {
        QMessageBox::information(this, tr("Unable to save file"), tr("Sorry !"));
    }
}

void MainWindow::on_resetPushButton_clicked()
{
    ui->savePushButton->setEnabled(false);
    ui->loadPushButton->setEnabled(true);

    destroyWindow("watershed transform");
    result.release();

    markerMask = Scalar::all(0);
    img0.copyTo(img);
    imshow( "image", img );

    ui->doPushButton->setEnabled(true);
}

void MainWindow::on_doPushButton_clicked()
{
    ui->savePushButton->setEnabled(false);
    ui->loadPushButton->setEnabled(false);

    if( c == 27 ) {
        ui->exitPushButton->click();
    }

/* Watershed TODO:
 *  + import image
 * median blurring
 * grayscale
 * binary threshold
 * opening
 * distance between objects
 * resize down image (0.4 down)
    * threshold
 * resize down image for background
    * dilate
    * subtract
 * connectivity
    * connectedComponents
 * watershed
 * draw bounding box
 */


    //import image
    Mat src = img0;
    printf("import image : succesfully\n");

//    //median blurring
//    Mat bluredSrc;
//    medianBlur(src, bluredSrc, 13);
//    printf("median blurring : succesfully\n");


//    //grayscale
//    Mat grayMat;
//    cvtColor(bluredSrc, grayMat, COLOR_BGR2GRAY);
//    printf("grayscale : succesfully\n");


//    //binary threshold
//    Mat binaryImage;
//    threshold( grayMat, binaryImage, 65, 255, THRESH_BINARY);
//    printf("binary threshold : succesfully\n");

//    //opening
//    Mat openedImage;
//    Mat kernel = Mat::ones(3, 3, CV_8UC1);
//    morphologyEx( binaryImage, openedImage, MORPH_OPEN, kernel, Point(-1,-1), 2);
//    printf("opening : succesfully\n");

//    //distance between objects
//    Mat distImage;
//    distanceTransform(openedImage, distImage, DIST_L2, 5);
//    printf("distance between objects : succesfully\n");

//    //resize down image (0.4 down)
//    // * threshold
//    double min, max;
//    cv::minMaxLoc(distImage, &min, &max);
//    Mat sureForeground;
//    threshold( distImage, sureForeground, 0.4*max, 255, THRESH_BINARY);
//    printf("resize down image : succesfully\n");

//    //resize down image for background
//    // * dilate
//    // * subtract
//    Mat sureBackground, unknown;
//    dilate( openedImage, sureBackground, kernel);
//    sureForeground.convertTo(sureForeground, CV_8U);
//    subtract(sureBackground, sureForeground, unknown);
//    printf("resize down image for background : succesfully\n");

//    //connectivity
//    // * connectedComponents
//    Mat marker;
//    int nLabels = connectedComponents(sureForeground, marker);
//    for (int i; i < unknown.cols; i++ ) {
//        for (int j; j < unknown.rows ; j++ ) {
//            // get pixel
//            Vec3b unknownPixel= unknown.at<Vec3b>(j,i);
//            Vec3b & markerPixel = marker.at<Vec3b>(j,i);
//            if(unknownPixel[0] == 255)
//                markerPixel[0] = 0;
//        }
//    }
//    printf("connectivity : succesfully\n");

//    //watershed
//    watershed(src, marker);
//    imshow( "watershed transform", marker );
//    printf("watershed : succesfully\n");

//    //draw bounding box

//    printf("draw bounding box : succesfully\n");


///////////////////////////////////////////////////////////////

    // Change the background from white to black, since that will help later to extract
    // better results during the use of Distance Transform
    Mat mask;
    inRange(src, Scalar(255, 255, 255), Scalar(255, 255, 255), mask);
    src.setTo(Scalar(0, 0, 0), mask);

    // Show output image
    //imshow("Black Background Image", src);

    // Create a kernel that we will use to sharpen our image
    Mat kernel = (Mat_<float>(3,3) <<
                  1,  1, 1,
                  1, -8, 1,
                  1,  1, 1); // an approximation of second derivative, a quite strong kernel

    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
//    Mat imgLaplacian;
//    filter2D(src, imgLaplacian, CV_32F, kernel);
        Mat imgLaplacian;
        medianBlur(src, imgLaplacian, 13);

    Mat sharp;
    src.convertTo(sharp, CV_8UC3);

    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

    imshow( "Laplace Filtered Image", imgLaplacian );
    imshow( "New Sharped Image", imgResult );

    // Create binary image from source image
    Mat bw;
    cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
    imshow("Binary Image", bw);

    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(bw, dist, DIST_L2, 3);

    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    imshow("Distance Transform Image", dist);

    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);

    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1);
    imshow("Peaks", dist);

    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);

    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32S);

    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i)+1), -1);
    }

    // Draw the background marker
    circle(markers, Point(5,5), 3, Scalar(255), -1);
    Mat markers8u;
    markers.convertTo(markers8u, CV_8U, 10);
    imshow("Markers", markers8u);

    //opening
    Mat kernel2 = Mat::ones(7, 7, CV_8UC1);
    morphologyEx( markers8u, markers8u, MORPH_OPEN, kernel2, Point(-1,-1), 3);

    // Perform the watershed algorithm
    watershed(imgResult, markers);
    Mat mark;
    markers.convertTo(mark, CV_8U);
    bitwise_not(mark, mark);
    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
    // image looks like at that point

    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);

    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<Vec3b>(i,j) = colors[index-1];
            }
        }
    }

    // Visualize the final image
    //QMessageBox::information(this, tr("The process has been done"), tr("Show the result"));
    imshow("Final Result", dst);


    /////////////////////////////////////////////////////

//    int i, j, compCount = 0;
//    vector<vector<Point> > contours;
//    vector<Vec4i> hierarchy;

//    findContours(markerMask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

//    if( contours.empty() ){}

//    Mat markers(markerMask.size(), CV_32S);
//    markers = Scalar::all(0);
//    int idx = 0;
//    for( ; idx >= 0; idx = hierarchy[idx][0], compCount++ )
//        drawContours(markers, contours, idx, Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);

//    if( compCount == 0 ){}

//    vector<Vec3b> colorTab;
//    for( i = 0; i < compCount; i++ )
//    {
//        int b = theRNG().uniform(0, 255);
//        int g = theRNG().uniform(0, 255);
//        int r = theRNG().uniform(0, 255);

//        colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
//    }

//    double t = (double)getTickCount();
//    watershed( img0, markers );
//    t = (double)getTickCount() - t;
//    printf( "execution time = %gms\n", t*1000./getTickFrequency() );

//    Mat wshed(markers.size(), CV_8UC3);

//    // paint the watershed image
//    for( i = 0; i < markers.rows; i++ )
//        for( j = 0; j < markers.cols; j++ )
//        {
//            int index = markers.at<int>(i,j);
//            if( index == -1 )
//                wshed.at<Vec3b>(i,j) = Vec3b(255,255,255);
//            else if( index <= 0 || index > compCount )
//                wshed.at<Vec3b>(i,j) = Vec3b(0,0,0);
//            else
//                wshed.at<Vec3b>(i,j) = colorTab[index - 1];
//        }

//    wshed = wshed*0.5 + imgGray*0.5;
//    wshed.copyTo(result);

//    QMessageBox::information(this, tr("The process has been done"), tr("Show the result"));
//    imshow( "watershed transform", wshed );
    ui->savePushButton->setEnabled(true);
}

void MainWindow::on_savePushButton_clicked()
{
    if(!result.empty()) {
        QString s = QFileDialog::getSaveFileName( this,tr("Save Image"),QDir::homePath(),tr("Images (*.png *.xpm *.jpeg *.jpg)") );

        if(!s.isNull()) {
            imwrite(s.toStdString(), result);
            QMessageBox::information(this, tr("Result saved"), tr("Good job"));
            ui->loadPushButton->setEnabled(true);
            ui->savePushButton->setEnabled(false);
            ui->doPushButton->setEnabled(false);
            ui->resetPushButton->setEnabled(true);
        } else {
            QMessageBox::information(this, tr("Unable to save file"), tr("Sorry !"));

        }
    }
}

void MainWindow::on_exitPushButton_clicked(){
    QMessageBox msgBox;

    int ret = msgBox.information(this, tr("Exit"), tr("Are you sure to quit the app?"), QMessageBox::Yes | QMessageBox::No);

    switch (ret) {
      case QMessageBox::No :
        break;
      case QMessageBox::Yes :
            MainWindow::connect(ui->exitPushButton,SIGNAL(clicked()), qApp,SLOT(quit()));
        break;
      default:
          break;
    }
}


void MainWindow::on_actionDescription_triggered()
{
    QMessageBox::information(this, tr("Description"), tr("Qt C++ with OpenCV desktop application (version 1.0.0)\nImages segmentation by implementing the watershed algorithm.\nLicense : GNU GPL v3.0."));
}
