/*

Filename : surfReg.cpp
Author : Varun Santhaseelan
Date: 11/3/2012

This program extracts SURF feature points and registers two images.

*/
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex>
#include <vector>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

void main()
{
	//Give the names of the images to be registered
	const char* imRef_name = "834-r1.png";
	const char* imNxt_name = "835-r1.png";
	int hessianThresh = 100, ransacThresh = 3;;
	Mat mask, H12;
	// Read images
	Mat img1 = imread(imRef_name, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread(imNxt_name, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2Out;	// Registered image2 wrt image1

	// Check to see if images exist
	if(img1.empty() || img2.empty())
	{
		printf("Can’t read one of the images\n");
		exit(0);
	}

	// detecting keypoints
	printf("Finding keypoints ... ");
	SURF ImgSurf(hessianThresh);
	vector<KeyPoint> keypoints1, keypoints2;
	ImgSurf(img1, mask, keypoints1);
	ImgSurf(img2, mask, keypoints2);
	// computing descriptors
	SurfDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor(img1,mask,keypoints1,descriptors1,TRUE);
	extractor(img2, mask, keypoints2, descriptors2, TRUE);

	// Match the points
	printf("\nMatching keypoints ... ");
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors1, descriptors2, matches );

	// Extract indices of matched points
    vector<int> queryIdxs( matches.size() ), trainIdxs( matches.size() );
    for( size_t i = 0; i < matches.size(); i++ )
    {
        queryIdxs[i] = matches[i].queryIdx;
        trainIdxs[i] = matches[i].trainIdx;
    }

	// Extract matched points from indices
    vector<Point2f> points1; KeyPoint::convert(keypoints1, points1, queryIdxs);
    vector<Point2f> points2; KeyPoint::convert(keypoints2, points2, trainIdxs);

	// Use RANSAC to find the homography
	printf("\nComputing homography ... ");
    H12 = findHomography( Mat(points2), Mat(points1), CV_RANSAC, ransacThresh );
	
	// Warp the second image according to the homography
	warpPerspective(img2, img2Out, H12, cvSize(img2.cols, img2.rows), INTER_LINEAR);

	// Write result to file
	imwrite("im2reg.png",img2Out);
	printf("\nDone!!!.... ");
}