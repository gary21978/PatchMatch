#include "inpaint.h"
#include <opencv2/imgproc.hpp>

PatchSSDDistanceMetric default_metric = PatchSSDDistanceMetric(1);

static std::vector<double> kDistance2Similarity;
// create look-up table using linear interpolation
void init_kDistance2Similarity() {
  const double base[11] = {1.0, 0.99, 0.96, 0.83, 0.38, 0.11, 0.02, 0.005, 0.0006, 0.0001, 0};
  int length = PatchDistanceMetric::kDistanceScale + 1;
  kDistance2Similarity.resize(length);
  for (int i = 0; i < length; ++i) {
    double t = static_cast<double>(i) / length;
    int j = static_cast<int>(100 * t);
    int k = j + 1;
    double vj = (j < 11) ? base[j] : 0;
    double vk = (k < 11) ? base[k] : 0;
    kDistance2Similarity[i] = vj + (100 * t - j) * (vk - vj);
  }
}

inline void _weighted_copy(const MaskedImage &source, int ys, int xs, cv::Mat *target, int yt,
                           int xt, double weight) {
  if (source.is_masked(ys, xs)) return;

  auto source_ptr = source.get_image(ys, xs);
  auto target_ptr = target->ptr<double>(yt, xt);

  target_ptr[0] += source_ptr[0] * weight;
  target_ptr[1] += source_ptr[1] * weight;
  target_ptr[2] += source_ptr[2] * weight;
  target_ptr[3] += weight;
}

bool Inpainting::_preprocessing(const cv::Mat &image, const cv::Mat &mask, int max_size) {
  // determine size capacity
  const int size_capacity =
      fmin(fmax(static_cast<int>(fmin(image.rows, image.cols) * 0.4), max_size / 2), max_size);

  if (_get_bounding_box(mask, &_crop_region)) return true;

  // crop the image and mask
  _effective_image = image(_crop_region);
  _effective_mask = mask(_crop_region);

  // downscale the image if too big
  if (_crop_region.area() > 1.5f * size_capacity * size_capacity) {
    _need_scale = true;
    const float scale_factor = size_capacity / std::sqrt(1.0f * _crop_region.area());

    cv::resize(_effective_image, _effective_image, cv::Size(), scale_factor, scale_factor);
    cv::resize(_effective_mask, _effective_mask, cv::Size(), scale_factor, scale_factor,
               cv::INTER_NEAREST);
    cv::threshold(_effective_mask, _effective_mask, 127, 255, cv::THRESH_BINARY);
  } else {
    _need_scale = false;
  }

  // denoise the mask using closing operation
  cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 13));
  cv::morphologyEx(_effective_mask, _effective_mask, cv::MORPH_CLOSE, close_kernel);

  return false;
}

// initialize the image pyramid
void Inpainting::_initialize_pyramid() {
  // create image pyramid
  MaskedImage source = _initial;
  _pyramid.push_back(source);
  while (source.size().height > _distance_metric->patch_size() &&
         source.size().width > _distance_metric->patch_size()) {
    bool hole_covered(true);
    source = source.downsample(&hole_covered);
    _pyramid.push_back(source);
#ifdef MULTITHREAD
    if (hole_covered) break;
#endif  // MULTITHREAD
  }

  if (kDistance2Similarity.size() == 0) init_kDistance2Similarity();
}

// Main routine: content-aware fill
int Inpainting::Fill(const cv::Mat &image, const cv::Mat &mask, cv::Mat *result_image,
                     int max_size) {
  // clone the original image
  _original_image = image.clone();

  cv::Mat mask_bw;
  if (mask.channels() == 3)
    cv::cvtColor(mask, mask_bw, cv::COLOR_BGR2GRAY);
  else
    mask_bw = mask;

  if (_preprocessing(image, mask_bw, max_size)) {
    *result_image = _original_image;
    return 1;
  }
  const PatchDistanceMetric *metric = &default_metric;
  _distance_metric = metric;
  _initial = MaskedImage(_effective_image, _effective_mask);
  _initialize_pyramid();

  const int max_retry = 15;
  const int nr_levels = _pyramid.size();
  MaskedImage source, target;

  srand2(time(NULL));

  for (int level = nr_levels - 1; level >= 0; --level) {
    // specific level from image pyramid
    source = _pyramid[level];

    // compute nearest neighbor field
    if (level == nr_levels - 1) {
      // The target is initialized to a low-resolution copy of source
      target = source.clone();
      target.clear_mask();
#ifdef ENABLE_COMPLETE
      _source2target = NearestNeighborField(source, target, _distance_metric, max_retry);
#endif  // ENABLE_COMPLETE
      _target2source = NearestNeighborField(target, source, _distance_metric, max_retry);
    } else {
#ifdef ENABLE_COMPLETE
      _source2target =
          NearestNeighborField(source, target, _distance_metric, _source2target, max_retry);
#endif  // ENABLE_COMPLETE
      _target2source =
          NearestNeighborField(target, source, _distance_metric, _target2source, max_retry);
    }
    target = _expectation_maximization(source, target, level);
  }

  // paste inpainted result to the original image
  *result_image = _original_image;
  if (_need_scale) {
    cv::Mat filled(_crop_region.size(), _initial.image().type());
    cv::Mat recovered_mask(_crop_region.size(), _initial.mask().type());
    cv::resize(target.image(), filled, _crop_region.size());
    cv::resize(_effective_mask, recovered_mask, _crop_region.size(), 0, 0, cv::INTER_NEAREST);
    filled.copyTo((*result_image)(_crop_region), recovered_mask);
  } else {
    target.image().copyTo((*result_image)(_crop_region), _effective_mask);
  }
  return 0;
}

// EM-Like algorithm
// Returns a DOUBLE sized target image (unless level = 0).
MaskedImage Inpainting::_expectation_maximization(MaskedImage source, MaskedImage target,
                                                  int level) {
  const int nr_iters_em = 1 + level;
  const int nr_iters_nnf = fmin(2, 1 + level);
  const int patch_size = _distance_metric->patch_size();

  MaskedImage new_source, new_target;

  for (int iter_em = 0; iter_em < nr_iters_em; ++iter_em) {
    if (iter_em != 0) {
#ifdef ENABLE_COMPLETE
      _source2target.set_target(new_target);
#endif  // ENABLE_COMPLETE
      _target2source.set_source(new_target);
      target = new_target;
    }

    cv::Size size = source.size();

#ifndef MULTITHREAD
    for (int i = 0; i < size.height; ++i) {
      for (int j = 0; j < size.width; ++j) {
        // the patch centered at (i,j) does not intersects with mask
        if (!source.contains_mask(i, j, patch_size)) {
#ifdef ENABLE_COMPLETE
          _source2target.set_identity(i, j);
#endif  // ENABLE_COMPLETE
          _target2source.set_identity(i, j);
        }
      }
    }
#else
    auto processor =
        [&source, &patch_size, &size, this ](const int &start, const int &end) -> void {
      for (int index = 0; index < end; ++index) {
        const int &y = index / size.width;
        const int &x = index % size.width;
        if (!source.contains_mask(y, x, patch_size)) {
#ifdef ENABLE_COMPLETE
          this->_source2target.set_identity(y, x);
#endif  // ENABLE_COMPLETE
          this->_target2source.set_identity(y, x);
        }
      }
    };
    const int pts_num = size.height * size.width;
    const int unit_count = static_cast<int>(pts_num / NUM_THREADS);
    std::vector<std::thread> workers;
    int start = 0;
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
      workers.emplace_back(processor, start, start + unit_count);
      start += unit_count;
    }
    processor(start, pts_num);
    for (auto &&iter : workers) {
      iter.join();
    }
#endif  // MULTITHREAD

#ifdef ENABLE_COMPLETE
    _source2target.minimize(nr_iters_nnf);
#endif  // ENABLE_COMPLETE
    _target2source.minimize(nr_iters_nnf);

    // Instead of upsizing the final target, we build the last target from
    // the next level source image. Thus, the final target is less blurry
    // (see "Space-Time Video Completion" - page 5).
    bool upscaled = false;
    if (level >= 1 && iter_em == nr_iters_em - 1) {
      new_source = _pyramid[level - 1];
      new_target = target.upsample(new_source.size().width, new_source.size().height);
      upscaled = true;
    } else {
      new_source = _pyramid[level];
      new_target = target.clone();
    }

    cv::Mat vote = cv::Mat::zeros(new_target.size(), CV_64FC4);
    // Votes for best patch from NNF Source->Target (completeness)
    // and Target->Source (coherence).

#ifdef ENABLE_COMPLETE
    _expectation_step(_source2target, 1, &vote, new_source, upscaled);
#endif  // ENABLE_COMPLETE
    _expectation_step(_target2source, 0, &vote, new_source, upscaled);

    // Compile votes and update pixel values.
    _maximization_step(&new_target, vote);
  }
  return new_target;
}

// Expectation step: vote for best estimations of each pixel.
void Inpainting::_expectation_step(const NearestNeighborField &nnf, bool source2target,
                                   cv::Mat *vote, const MaskedImage &source, bool upscaled) {
  auto source_size = nnf.source_size();
  auto target_size = nnf.target_size();
  const int patch_size = _distance_metric->patch_size();

#ifndef MULTITHREAD
  int xs, ys, xt, yt;
  int _xs, _ys, _xt, _yt;

  for (int i = 0; i < source_size.height; ++i) {
    for (int j = 0; j < source_size.width; ++j) {
      int yp = nnf.at(i, j, 0), xp = nnf.at(i, j, 1), dp = nnf.at(i, j, 2);
      double w = kDistance2Similarity[dp];

      if (source2target) {
        _ys = i, _yt = yp, _xs = j, _xt = xp;
      } else {
        _ys = yp, _yt = i, _xs = xp, _xt = j;
      }

      for (int di = -patch_size; di <= patch_size; ++di) {
        ys = _ys + di, yt = _yt + di;
        if (!(ys >= 0 && ys < source_size.height && yt >= 0 && yt < target_size.height)) continue;

        for (int dj = -patch_size; dj <= patch_size; ++dj) {
          xs = _xs + dj, xt = _xt + dj;
          if (!(xs >= 0 && xs < source_size.width && xt >= 0 && xt < target_size.width)) continue;

          if (upscaled) {
            _weighted_copy(source, 2 * ys, 2 * xs, vote, 2 * yt, 2 * xt, w);
            _weighted_copy(source, 2 * ys, 2 * xs + 1, vote, 2 * yt, 2 * xt + 1, w);
            _weighted_copy(source, 2 * ys + 1, 2 * xs, vote, 2 * yt + 1, 2 * xt, w);
            _weighted_copy(source, 2 * ys + 1, 2 * xs + 1, vote, 2 * yt + 1, 2 * xt + 1, w);
          } else {
            _weighted_copy(source, ys, xs, vote, yt, xt, w);
          }
        }
      }
    }
  }
#else
  auto processor = [
    this, &nnf, &vote, source2target, &source, upscaled, &patch_size, &source_size, &target_size
  ](const int &start, const int &end) -> void {
    for (int index = start; index < end; ++index) {
      const int &i = index / source_size.width;
      const int &j = index % source_size.width;
      for (int di = -patch_size; di <= patch_size; ++di) {
        const int &yt = i + di;
        if (yt < 0 || yt >= target_size.height) {
          continue;
        }
        for (int dj = -patch_size; dj <= patch_size; ++dj) {
          const int &xt = j + dj;
          if (xt < 0 || xt >= target_size.width) {
            continue;
          }
          const int &ys = nnf.at(yt, xt, 0) - di;
          if (ys < 0 || ys >= source_size.height) {
            continue;
          }
          const int &xs = nnf.at(yt, xt, 1) - dj;
          if (xs < 0 || xs >= source_size.width) {
            continue;
          }
          const int &dp = nnf.at(yt, xt, 2);
          const double &w = kDistance2Similarity[dp];
          if (upscaled) {
            _weighted_copy(source, 2 * ys, 2 * xs, vote, 2 * i, 2 * j, w);
            _weighted_copy(source, 2 * ys, 2 * xs + 1, vote, 2 * i, 2 * j + 1, w);
            _weighted_copy(source, 2 * ys + 1, 2 * xs, vote, 2 * i + 1, 2 * j, w);
            _weighted_copy(source, 2 * ys + 1, 2 * xs + 1, vote, 2 * i + 1, 2 * j + 1, w);
          } else {
            _weighted_copy(source, ys, xs, vote, i, j, w);
          }
        }
      }
    }
  };
  const int pts_num = source_size.height * source_size.width;
  const int unit_count = static_cast<int>(pts_num / NUM_THREADS);
  std::vector<std::thread> workers;
  int start = 0;
  for (int i = 0; i < NUM_THREADS - 1; ++i) {
    workers.emplace_back(processor, start, start + unit_count);
    start += unit_count;
  }
  processor(start, pts_num);
  for (auto &&iter : workers) {
    iter.join();
  }
#endif  // MULTITHREAD
}

// Maximization Step: maximum likelihood of target pixel.
void Inpainting::_maximization_step(MaskedImage *target, const cv::Mat &vote) {
#ifndef MULTITHREAD
  auto target_size = target->size();
  for (int i = 0; i < target_size.height; ++i) {
    for (int j = 0; j < target_size.width; ++j) {
      const double *source_ptr = vote.ptr<double>(i, j);
      uchar *target_ptr = target->get_mutable_image(i, j);

      if (source_ptr[3] > 0) {
        // compute average
        target_ptr[0] = cv::saturate_cast<uchar>(source_ptr[0] / source_ptr[3]);
        target_ptr[1] = cv::saturate_cast<uchar>(source_ptr[1] / source_ptr[3]);
        target_ptr[2] = cv::saturate_cast<uchar>(source_ptr[2] / source_ptr[3]);
      } else {
        target->set_mask(i, j, 0);
      }
    }
  }
#else
  auto processor = [&target, &vote ](const int &start, const int &end) -> void {
    for (int index = start; index < end; ++index) {
      const int &i = index / vote.cols;
      const int &j = index % vote.cols;
      const cv::Vec4d *iter = vote.ptr<cv::Vec4d>(i) + j;
      uchar *data = target->get_mutable_image(i, j);
      if (iter->val[3] > 0) {
        data[0] = cv::saturate_cast<uchar>(iter->val[0] / iter->val[3]);
        data[1] = cv::saturate_cast<uchar>(iter->val[1] / iter->val[3]);
        data[2] = cv::saturate_cast<uchar>(iter->val[2] / iter->val[3]);
      } else {
        target->set_mask(i, j, 0);
      }
    }
  };
  const int pts_num = vote.rows * vote.cols;
  const int unit_count = static_cast<int>(pts_num / NUM_THREADS);
  std::vector<std::thread> workers;
  int start = 0;
  for (int i = 0; i < NUM_THREADS - 1; ++i) {
    workers.emplace_back(processor, start, start + unit_count);
    start += unit_count;
  }
  processor(start, pts_num);
  for (auto &&iter : workers) {
    iter.join();
  }
#endif  // MULTITHREAD
}

// determine the bounding box from mask
bool Inpainting::_get_bounding_box(const cv::Mat &mask, cv::Rect *crop_region) {
  if (crop_region == NULL) return true;
  const int min_radius = 15;
  // determine the bounding box according to the given mask
  uint32_t mask_pixel_num = 0;
  int left = mask.cols, right = 0;
  int top = mask.rows, bottom = 0;
  bool mask_valid = false;
  for (int i = 0; i < mask.rows; ++i) {
    const uchar *Mi = mask.ptr<uchar>(i);
    for (int j = 0; j < mask.cols; ++j) {
      if (Mi[j]) {
        mask_valid = true;
        mask_pixel_num++;

        if (i < top) top = i;
        if (j < left) left = j;
        if (i > bottom) bottom = i;
        if (j > right) right = j;
      }
    }
  }

  if (!mask_valid || mask_pixel_num >= 0.99 * mask.rows * mask.cols) return true;

  int padding_width = (right - left + 1) / 2 + min_radius;
  int padding_height = (bottom - top + 1) / 2 + min_radius;
  int startX = fmax(left - padding_width, 0);
  int endX = fmin(right + padding_width, mask.cols - 1);
  int startY = fmax(top - padding_height, 0);
  int endY = fmin(bottom + padding_height, mask.rows - 1);
  //  pad right border if left border reaches the limit
  if (left < padding_width && right + padding_width <= mask.cols - 1)
    endX = fmin(right + 2 * padding_width - left, mask.cols - 1);
  // pad left border if right border reaches the limit
  if (left >= padding_width && right + padding_width > mask.cols - 1)
    startX = fmax(left - 2 * padding_width - right + mask.cols - 1, 0);
  // pad lower border if upper border reaches the limit
  if (top < padding_height && bottom + padding_height <= mask.rows - 1)
    endY = fmin(bottom + 2 * padding_height - top, mask.rows - 1);
  // pad upper border if lower border reaches the limit
  if (top >= padding_height && bottom + padding_height > mask.rows - 1)
    startY = fmax(top - 2 * padding_height - bottom + mask.rows - 1, 0);
  *crop_region = cv::Rect(startX, startY, endX - startX + 1, endY - startY + 1);
  return false;
}
