#include <iostream>
#include <string>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "config.h"

using namespace std;
using namespace cv;
using namespace std::chrono;

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

Mat compute_optical_flow(const Mat& frame1, const Mat& frame2, int window_size, int step)
{

    assert(window_size > 0);
    assert(step > 0);

    int width = frame1.size().width;
    int height = frame1.size().height;

    int magnitudes[height][width];
    memset(&magnitudes, 0, sizeof(magnitudes));

    Mat out = frame1.clone();

    uint8_t* pixelPtr = (uint8_t*)frame1.data;
    double Ix = 1, Iy = 1, It = 1;
    int cn = frame1.channels();
    Scalar_<uint8_t> bgrPixel;

#pragma omp parallel for default(none) shared(height, width, window_size, step, frame1, frame2, out) private(Ix, Iy, It) collapse(2) num_threads(omp_get_max_threads())
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
                    arrowedLine(out, pt1, pt2, Scalar_<double>(0.2, 0.4, 0.7));
                }
            }
        } // for loop cols
    } // for loop rows

    return out;
}

void optical_flow(const string& path)
{
    Mat prev_frame;
    Mat current_frame;

    int count = 0;
    int frames = 1;

    VideoCapture cap(path);
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return;
    }

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

            Mat out = compute_optical_flow(grayscale1, grayscale2, 2, 1);

            imwrite(R"(C:\output)" + to_string(count) + ".jpg", out);
            count++;
        }

        prev_frame = current_frame.clone();
        frames++;

        char c=(char)waitKey(25);
        if(c==27)
            break;
    }

    cap.release();
    destroyAllWindows();
}

int main() {

    if (omp_get_max_threads() == 0) {
        return -1;
    }

    auto start = high_resolution_clock::now();
    optical_flow(VIDEO_FILENAME);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Computation time: " << duration.count() << " microseconds" << endl;

    return 0;
}
