#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    Mat img = imread(R"(C:\Users\tomoi\CLionProjects\openmp-optical-flow-cpp\Sebes.jpg)", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Image File " << "Not Found" << endl;
        cin.get();
        return -1;
    }

    imshow("Window Name", img);

    waitKey(0);
#pragma omp parallel default(none)
    {
        int pid = omp_get_thread_num();
//#pragma omp critical
        {
            printf("Threads: %d\n", pid);
        }
    }

    return 0;
}
