#include <iostream>
#include <string>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "config.h"

using namespace std;
using namespace cv;
using namespace std::chrono;

struct Features
{
    Point featureArray[MAX_FEATURES][2];
    int size;

    Features() : size(0)
    {
        memset(&featureArray, 0, sizeof(featureArray));
    }
};

float sqrt_(float n)
{
    float x = n;
    float y = 1;
    float eps = 0.000001;
    while (x - y > eps) {
        x = (x + y) / 2;
        y = n / x;
    }
    return x;
}

Features compute_optical_flow(const Mat& frame1, const Mat& frame2, int window_size, int step)
{
    assert(window_size > 0);
    assert(step > 0);

    int width = frame1.size().width;
    int height = frame1.size().height;

    int magnitudes[height][width];
    memset(&magnitudes, 0, sizeof(magnitudes));

    Mat out = frame1.clone();

    double Ix = 1, Iy = 1, It = 1;
    Scalar_<uint8_t> bgrPixel;

    Features features;

#pragma omp parallel for default(none) shared(height, width, window_size, step, frame1, frame2, out, features) private(Ix, Iy, It) collapse(2) num_threads(omp_get_max_threads())
    for (int l = window_size - 1; l < height - window_size - 1; l += step) {
        for (int k = window_size - 1; k < width - window_size - 1; k += step) {
            // Compute derivatives in x, y and time directions
            Ix = .25 * (frame2.data[l * width + (k + 1)] + frame2.data[(l + 1) * width + (k + 1)] + frame1.data[l * width + (k + 1)] + frame1.data[(l + 1) * width + (k + 1)])
                    - .25 * (frame2.data[l * width + k] + frame2.data[(l + 1) * width + k] + frame1.data[l * width + k] + frame1.data[(l + 1) * width + k]);
            Iy = .25 * (frame2.data[l * width + (k + 1)] + frame2.data[l * width + k] + frame1.data[l * width + (k + 1)] + frame1.data[l* width + k])
                    - .25 * (frame2.data[(l + 1) * width + k + 1] + frame2.data[(l + 1) * width + k] + frame1.data[(l + 1) * width + k] + frame1.data[(l + 1) * width + k]);
            It = .25 * (frame2.data[l * width + k] + frame2.data[l * width + k + 1] + frame2.data[(l + 1) * width + k] + frame2.data[(l + 1) * width + k + 1])
            - .25 * (frame1.data[l * width + k] + frame1.data[l * width + k + 1] + frame1.data[(l + 1) * width + k] + frame1.data[(l + 1) * width + k + 1]);

            double Un_mag = 1.0;
            //float dirX, dirY;
            double Un[2];
            double sqrt_dX_dY = sqrt((double)(Ix * Ix + Iy * Iy));
            if (sqrt_dX_dY != 0)
            {
                Un_mag = abs(It) / sqrt_dX_dY;
                //dirX = Ix / sqrt_dX_dY;
                //dirY = Iy / sqrt_dX_dY;
                Un[0] = (Un_mag * Ix);
                Un[1] = (Un_mag * Iy);

                if ((Un[0] > LOWER_THRESHOLD && Un[1] > LOWER_THRESHOLD) && (Un[0] < UPPER_THRESHOLD && Un[1] < UPPER_THRESHOLD))
                {
                    Point pt1((int)(k + Un[1]), (int)(l + Un[0]));
                    Point pt2(k, l);
                    if (features.size < MAX_FEATURES - 1) {
#pragma omp critical
                        {
                            features.featureArray[features.size][0] = pt1;
                            features.featureArray[features.size][1] = pt2;
                            features.size++;
                        }
                        //arrowedLine(out, pt1, pt2, Scalar_<double>(0.2, 0.4, 0.7));
                    }
                }
            }
        } // for loop cols
    } // for loop rows

    return features;
}

void optical_flow(const string& path)
{
    int frame_width =320;
    int frame_height = 240;
    Size frame_size(frame_width, frame_height);
    int fps = 10;

    VideoWriter output(R"(C:\output.avi)",
                       VideoWriter::fourcc('M', 'J', 'P', 'G'),fps, frame_size, false);

    Mat prev_frame;
    Mat current_frame;

    int count = 0;
    int frames = 1;

    VideoCapture cap(path);
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return;
    }

    microseconds totalTime;
    memset(&totalTime, 0, sizeof(totalTime));

    while(true){
        cap >> current_frame;

        if (current_frame.empty())
            break;

        if (frames >= 3)
        {
            Mat grayscale1;
            Mat grayscale2;

            cvtColor(prev_frame, grayscale1, COLOR_BGR2GRAY);
            cvtColor(current_frame, grayscale2, COLOR_BGR2GRAY);

            Mat prev_frame_P2;
            Mat current_frame_P2;

            pyrDown(grayscale1,prev_frame_P2,Size(prev_frame.size().width/2, prev_frame.size().height/2));
            pyrDown(grayscale2,current_frame_P2,Size(prev_frame.size().width/2, prev_frame.size().height/2));

            Features p0_features, p2_features;

            auto start = high_resolution_clock::now();
#pragma omp parallel default(none) shared(grayscale1, grayscale2, p0_features, p2_features, prev_frame_P2, current_frame_P2) num_threads(2)
            {
#pragma omp sections
            {
#pragma omp section
            {
                p0_features = compute_optical_flow(grayscale1, grayscale2, 2, 1);
            }
#pragma omp section
            {
                p2_features = compute_optical_flow(prev_frame_P2, current_frame_P2, 2, 1);
            }
            }
            }
            auto stop = high_resolution_clock::now();
            totalTime += duration_cast<microseconds>(stop - start);

            for (int i = 0; i < p0_features.size; ++i) {
                arrowedLine(grayscale2, p0_features.featureArray[i][0], p0_features.featureArray[i][1], Scalar_<double>(0.2, 0.4, 0.7));
            }

            for (int i = 0; i < p2_features.size; ++i) {
                arrowedLine(grayscale2, p2_features.featureArray[i][0] * 2, p2_features.featureArray[i][1] * 2, Scalar_<double>(0.2, 0.4, 0.7));
            }

            //output.write(out);

            imwrite(R"(C:\output\img_)" + to_string(count) + ".jpg", grayscale2);
            count++;
        }

        prev_frame = current_frame.clone();
        frames++;

        char c=(char)waitKey(25);
        if(c==27)
            break;
    }

    cout << "Computation time: " << totalTime.count() << " microseconds" << endl;

    cap.release();
    output.release();
    destroyAllWindows();
}

int main() {

    if (omp_get_max_threads() == 0) {
        return -1;
    }

    optical_flow(VIDEO_FILENAME);

    return 0;
}
