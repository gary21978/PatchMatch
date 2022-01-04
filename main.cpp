#include <opencv2/highgui.hpp>
#include "inpaint.h"

int main(int argc, char** argv) {
   if (argc < 3) {
      printf("Usage: %s input mask output\n", argv[0]);
      return 1;
   }
   cv::Mat input = cv::imread(argv[1]);
   cv::Mat mask = cv::imread(argv[2]);
   cv::Mat result;
   Inpainting app;
   app.Fill(input, mask, &result);
   cv::imwrite(argv[3], result);
   return 0;
}
