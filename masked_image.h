#pragma once

// #define MULTITHREAD

#ifdef MULTITHREAD
#include <thread>
#define NUM_THREADS 4
#endif  // MULTITHREAD

#include <vector>
#include <opencv2/core.hpp>

class MaskedImage {
 public:
  MaskedImage() : _image(), _mask(), _image_grady(), _image_gradx(), _image_grad_computed(false) {}

  MaskedImage(cv::Mat image, cv::Mat mask)
      : _image(image), _mask(mask), _image_grad_computed(false) {}

  MaskedImage(cv::Mat image, cv::Mat mask, cv::Mat grady, cv::Mat gradx, bool grad_computed)
      : _image(image),
        _mask(mask),
        _image_grady(grady),
        _image_gradx(gradx),
        _image_grad_computed(grad_computed) {}

  MaskedImage(int width, int height) : _image_grady(), _image_gradx() {
    _image = cv::Mat(cv::Size(width, height), CV_8UC3);
    _image = cv::Scalar::all(0);

    _mask = cv::Mat(cv::Size(width, height), CV_8U);
    _mask = cv::Scalar::all(0);
  }

  inline MaskedImage clone() {
    return MaskedImage(_image.clone(), _mask.clone(), _image_grady.clone(), _image_gradx.clone(),
                       _image_grad_computed);
  }

  inline cv::Size size() const { return _image.size(); }

  inline const cv::Mat &image() const { return _image; }

  inline const cv::Mat &mask() const { return _mask; }

  inline const cv::Mat &grady() const {
    assert(_image_grad_computed);
    return _image_grady;
  }

  inline const cv::Mat &gradx() const {
    assert(_image_grad_computed);
    return _image_gradx;
  }

  inline bool is_masked(int y, int x) const {
    return static_cast<bool>(_mask.at<unsigned char>(y, x));
  }

  inline void set_mask(int y, int x, bool value) {
    _mask.at<unsigned char>(y, x) = static_cast<unsigned char>(value);
  }

  inline void clear_mask() { _mask.setTo(cv::Scalar(0)); }

  inline const unsigned char *get_image(int y, int x) const {
    return _image.ptr<unsigned char>(y, x);
  }

  inline unsigned char *get_mutable_image(int y, int x) { return _image.ptr<unsigned char>(y, x); }

  bool contains_mask(int y, int x, int patch_size) const;
  MaskedImage downsample(bool *hole_covered) const;
  MaskedImage upsample(int new_w, int new_h) const;
  void compute_image_gradients();
#ifndef MULTITHREAD
  void compute_image_gradients() const;
#endif

  static const cv::Size kDownsampleKernelSize;
  static const int kDownsampleKernel[6];

 private:
  cv::Mat _image;
  cv::Mat _mask;
  cv::Mat _image_grady;
  cv::Mat _image_gradx;
  bool _image_grad_computed = false;
};
