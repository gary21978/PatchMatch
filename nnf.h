#pragma once

#include "masked_image.h"

// adopt a small PRNG by George Marsaglia, faster than rand(), from:
// http://www.math.uni-bielefeld.de/~sillke/ALGORITHMS/random/marsaglia-c

inline unsigned rand2();
void srand2(unsigned seed);

#define ENABLE_GRAD

class PatchDistanceMetric {
 public:
  explicit PatchDistanceMetric(int patch_size) : _patch_size(patch_size) {}
  virtual ~PatchDistanceMetric() = default;

  inline int patch_size() const { return _patch_size; }
  virtual int operator()(const MaskedImage &source, int source_y, int source_x,
                         const MaskedImage &target, int target_y, int target_x,
                         int baseline) const = 0;
  static const int kDistanceScale;

 protected:
  int _patch_size;
};

class NearestNeighborField {
 public:
  NearestNeighborField() : _source(), _target(), _field(), _distance_metric(nullptr) {}

  NearestNeighborField(const MaskedImage &source, const MaskedImage &target,
                       const PatchDistanceMetric *metric, int max_retry = 20)
      : _source(source), _target(target), _distance_metric(metric) {
    _field = cv::Mat(_source.size(), CV_32SC3);
#ifdef MULTITHREAD
#ifdef ENABLE_GRAD
    std::thread t(&MaskedImage::compute_image_gradients, std::ref(this->_source));
    this->_target.compute_image_gradients();
    t.join();
#endif  // ENABLE_GRAD
#endif  // MULTITHREAD
    _randomize_field(max_retry);
  }

  NearestNeighborField(const MaskedImage &source, const MaskedImage &target,
                       const PatchDistanceMetric *metric, const NearestNeighborField &other,
                       int max_retry = 20)
      : _source(source), _target(target), _distance_metric(metric) {
    _field = cv::Mat(_source.size(), CV_32SC3);
#ifdef MULTITHREAD
#ifdef ENABLE_GRAD
    std::thread t(&MaskedImage::compute_image_gradients, std::ref(this->_source));
    this->_target.compute_image_gradients();
    t.join();
#endif  // ENABLE_GRAD
#endif  // MULTITHREAD
    _initialize_field_from(other, max_retry);
  }

  const MaskedImage &source() const { return _source; }

  const MaskedImage &target() const { return _target; }

  inline cv::Size source_size() const { return _source.size(); }

  inline cv::Size target_size() const { return _target.size(); }

  inline void set_source(const MaskedImage &source) { _source = source; }

  inline void set_target(const MaskedImage &target) { _target = target; }

  inline int *mutable_ptr(int y, int x) { return _field.ptr<int>(y, x); }

  inline const int *ptr(int y, int x) const { return _field.ptr<int>(y, x); }

  inline int at(int y, int x, int c) const { return _field.ptr<int>(y, x)[c]; }

  inline int &at(int y, int x, int c) { return _field.ptr<int>(y, x)[c]; }

  inline void set_identity(int y, int x) {
    auto ptr = mutable_ptr(y, x);
    ptr[0] = y, ptr[1] = x, ptr[2] = 0;
  }

  void minimize(int nr_pass);

 private:
  inline int _distance(int source_y, int source_x, int target_y, int target_x,
                       int baseline = PatchDistanceMetric::kDistanceScale) {
    return (*_distance_metric)(_source, source_y, source_x, _target, target_y, target_x, baseline);
  }

  void _randomize_field(int max_retry = 20, bool reset = true);
  void _initialize_field_from(const NearestNeighborField &other, int max_retry);
  void _minimize_link(int y, int x, int direction);

  MaskedImage _source;
  MaskedImage _target;
  cv::Mat _field;
  const PatchDistanceMetric *_distance_metric;
};

class PatchSSDDistanceMetric : public PatchDistanceMetric {
 public:
  using PatchDistanceMetric::PatchDistanceMetric;
  virtual int operator()(const MaskedImage &source, int source_y, int source_x,
                         const MaskedImage &target, int target_y, int target_x, int baseline) const;
  static const int kSSDScale;
};
