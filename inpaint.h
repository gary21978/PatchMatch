#pragma once

#include "nnf.h"

//   #define ENABLE_COMPLETE

class Inpainting {
 public:
  int Fill(const cv::Mat &image, const cv::Mat &mask, cv::Mat *result_image, int max_size = 800);

 private:
  bool _preprocessing(const cv::Mat &image, const cv::Mat &mask, int max_size);
  bool _get_bounding_box(const cv::Mat &mask, cv::Rect *crop_region);
  void _initialize_pyramid(void);
  MaskedImage _expectation_maximization(MaskedImage source, MaskedImage target, int level);

  void _expectation_step(const NearestNeighborField &nnf, bool source2target, cv::Mat *vote,
                         const MaskedImage &source, bool upscaled);
  void _maximization_step(MaskedImage *target, const cv::Mat &vote);

  MaskedImage _initial;
  std::vector<MaskedImage> _pyramid;

  NearestNeighborField _source2target;
  NearestNeighborField _target2source;
  const PatchDistanceMetric *_distance_metric;

  cv::Rect _crop_region;
  cv::Mat _original_image;
  cv::Mat _effective_image, _effective_mask;
  bool _need_scale;
};
