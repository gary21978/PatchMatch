#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "masked_image.h"
#include "nnf.h"
#include "inpaint.h"

cv::Mat image, mask, visualize;
int cx, cy;
bool drawing;
static int thickness = 20;

void paint(int event, int x, int y, int flags, void* param) {
    cv::Scalar linecolor = cv::Scalar(0, 0, 200);
    const int max_thickness = 100, min_thickness = 1;

    switch (event) {
        case cv::EVENT_LBUTTONDOWN: {  
            // when the left mouse button is pressed down,
            drawing = true;
            cx = x, cy = y;
        }
        break;

        case cv::EVENT_MOUSEMOVE: {
            if (drawing) {
                if (thickness >= min_thickness && thickness <= max_thickness) {
                    cv::line(visualize, cv::Point(cx, cy), cv::Point(x, y), linecolor, thickness);
                    cv::line(mask, cv::Point(cx, cy), cv::Point(x, y), cv::Scalar(255), thickness);
                }
                cx = x, cy = y;
            }
        }
        break;

        case cv::EVENT_LBUTTONUP: {   //when the left mouse button is released,
            drawing = false;
            mask.setTo(cv::Scalar(0));
            visualize = image.clone();
        }
        break;
    }
}

int main(int argc, char** argv) {
    image = cv::imread(argv[1]);

    if (image.empty())
        return -1;

    visualize = image.clone();
    mask = cv::Mat(image.rows, image.cols, CV_8U, cv::Scalar(0));

    cv::namedWindow("Healing Brush Tool");
    cv::setMouseCallback("Healing Brush Tool", paint);

    while(1) {
        cv::imshow("Healing Brush Tool", visualize);

        int key = cv::waitKey(2) & 0xFF;

        if (key == 27)
            break;
        if (key == 61) // +
            thickness += 2;
        if (key == 45) // -
            thickness -= 2;

        cv::Mat image_old = image.clone();
        cv::Mat mask_old = mask.clone();
        Inpainting inpainter;
        inpainter.Fill(image_old, mask_old, &image);
    } 
    cv::destroyAllWindows();
    return 0;
}
