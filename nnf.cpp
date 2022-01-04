#include "nnf.h"
#include "masked_image.h"

/**
 *   Nearest-Neighbor Field (see PatchMatch algorithm).
 */

static unsigned rand2_u = 2282506733U, rand2_v = 1591164231U;

inline unsigned rand2() {
  rand2_v = 36969 * (rand2_v & 65535) + (rand2_v >> 16);
  rand2_u = 18000 * (rand2_u & 65535) + (rand2_u >> 16);
  return (rand2_v << 16) + (rand2_u & 65535);
}

void srand2(unsigned seed) {
  rand2_u = seed;
  rand2_v = ~seed;
  if (!rand2_u) rand2_u++;
  if (!rand2_v) rand2_v++;
  for (int i = 0; i < 10; i++) rand2();

  rand2_u = rand2();
  rand2_v = rand2() ^ seed;
  if (!rand2_u) rand2_u++;
  if (!rand2_v) rand2_v++;
}

template <typename T>
T clamp(T value, T min_value, T max_value) {
  if (value < min_value) return min_value;
  if (value > max_value) return max_value;
  return value;
}

void NearestNeighborField::_randomize_field(int max_retry, bool reset) {
  cv::Size this_size = source_size();
#ifndef MULTITHREAD
  for (int i = 0; i < this_size.height; ++i) {
    for (int j = 0; j < this_size.width; ++j) {
      int *this_ptr = mutable_ptr(i, j);
      int distance = reset ? PatchDistanceMetric::kDistanceScale : this_ptr[2];
      if (distance < PatchDistanceMetric::kDistanceScale) continue;

      int i_target = 0, j_target = 0;
      for (int t = 0; t < max_retry; ++t) {
        i_target = rand2() % this_size.height;
        j_target = rand2() % this_size.width;

        distance = _distance(i, j, i_target, j_target);
        if (distance < PatchDistanceMetric::kDistanceScale) break;
      }
      this_ptr[0] = i_target, this_ptr[1] = j_target, this_ptr[2] = distance;
    }
  }
#else
  auto processor =
      [&reset, &max_retry, &this_size, this ](const int &start, const int &end) -> void {
    for (int index = start; index < end; ++index) {
      const int &i = index / this_size.width;
      const int &j = index % this_size.width;
      cv::Vec3i *iter = this->_field.ptr<cv::Vec3i>(i) + j;
      int distance = reset ? PatchDistanceMetric::kDistanceScale : iter->val[2];
      if (distance >= PatchDistanceMetric::kDistanceScale) {
        int i_target = 0, j_target = 0;
        for (int t = 0; t < max_retry; ++t) {
          i_target = rand2() % this_size.height;
          j_target = rand2() % this_size.width;
          distance = this->_distance(i, j, i_target, j_target);
          if (distance < PatchDistanceMetric::kDistanceScale) {
            break;
          }
        }
        iter->val[0] = i_target;
        iter->val[1] = j_target;
        iter->val[2] = distance;
      }
    }
    return;
  };
  const int pts_num = this_size.height * this_size.width;
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

void NearestNeighborField::_initialize_field_from(const NearestNeighborField &other,
                                                  int max_retry) {
  const cv::Size &this_size = source_size();
  const cv::Size &other_size = other.source_size();
  float fi = static_cast<float>(this_size.height) / other_size.height;
  float fj = static_cast<float>(this_size.width) / other_size.width;
#ifndef MULTITHREAD
  for (int i = 0; i < this_size.height; ++i) {
    int ilow = static_cast<int>(fmin(i / fi, other_size.height - 1.0f));
    for (int j = 0; j < this_size.width; ++j) {
      int jlow = static_cast<int>(fmin(j / fj, other_size.width - 1.0f));
      int *this_value = mutable_ptr(i, j);
      const int *other_value = other.ptr(ilow, jlow);

      this_value[0] = static_cast<int>(other_value[0] * fi);
      this_value[1] = static_cast<int>(other_value[1] * fj);
      this_value[2] = _distance(i, j, this_value[0], this_value[1]);
    }
  }
#else
  auto processor =
      [ fi, fj, &other, &other_size, this ](const int &start, const int &end) -> void {
    for (int index = start; index < end; ++index) {
      const int &i = index / this->_field.cols;
      const int &j = index % this->_field.cols;
      cv::Vec3i *iter = this->_field.ptr<cv::Vec3i>(i) + j;
      const int &ilow = static_cast<int>(fmin(i / fi, other_size.height - 1.0f));
      const int &jlow = static_cast<int>(fmin(j / fj, other_size.width - 1.0f));
      const int *other_value = other.ptr(ilow, jlow);
      iter->val[0] = static_cast<int>(other_value[0] * fi);
      iter->val[1] = static_cast<int>(other_value[1] * fj);
      iter->val[2] = this->_distance(i, j, iter->val[0], iter->val[1]);
    }
  };
  const int pts_num = this_size.height * this_size.width;
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
  // remove distance-saturated entries
  _randomize_field(max_retry, false);
}

void NearestNeighborField::minimize(int nr_pass) {
  const cv::Size &this_size = source_size();

#ifdef MULTITHREAD
#ifdef ENABLE_GRAD
  std::thread t0(&MaskedImage::compute_image_gradients, std::ref(this->_source));
  this->_target.compute_image_gradients();
  t0.join();
#endif  // ENABLE_GRAD
#endif  // MULTITHREAD

  while (nr_pass--) {
    // propagate downright at odd step
    for (int i = 0; i < this_size.height; ++i)
      for (int j = 0; j < this_size.width; ++j)
        if (at(i, j, 2) > 0)  // matching can be improved
          _minimize_link(i, j, +1);

    // propagate upleft at even step
    for (int i = this_size.height - 1; i >= 0; --i)
      for (int j = this_size.width - 1; j >= 0; --j)
        if (at(i, j, 2) > 0)  // matching can be improved
          _minimize_link(i, j, -1);
  }
}

void NearestNeighborField::_minimize_link(int y, int x, int direction) {
  const cv::Size &this_size = source_size();
  const cv::Size &this_target_size = target_size();
  int *this_ptr = mutable_ptr(y, x);
  int xp, yp, dp;  // correspond to three component of NNF

  // propagation along the y direction.
  if (y - direction >= 0 && y - direction < this_size.height) {
    yp = at(y - direction, x, 0) + direction;
    xp = at(y - direction, x, 1);
    dp = _distance(y, x, yp, xp, at(y, x, 2));  // early termination
    if (dp < at(y, x, 2)) {
      this_ptr[0] = yp, this_ptr[1] = xp, this_ptr[2] = dp;
    }
  }

  // propagation along the x direction.
  if (x - direction >= 0 && x - direction < this_size.width) {
    yp = at(y, x - direction, 0);
    xp = at(y, x - direction, 1) + direction;
    dp = _distance(y, x, yp, xp, at(y, x, 2));  // early termination
    if (dp < at(y, x, 2)) {
      this_ptr[0] = yp, this_ptr[1] = xp, this_ptr[2] = dp;
    }
  }

  // random search with a progressive step size.
  int random_scale = (fmin(this_target_size.height, this_target_size.width) - 1) / 2;
  int ypi = this_ptr[0], xpi = this_ptr[1];
  while (random_scale > 0) {
    yp = ypi + (rand2() % (2 * random_scale + 1)) - random_scale;
    xp = xpi + (rand2() % (2 * random_scale + 1)) - random_scale;
    yp = clamp(yp, 0, target_size().height - 1);
    xp = clamp(xp, 0, target_size().width - 1);
    dp = _distance(y, x, yp, xp, at(y, x, 2));  // early termination
    if (dp < at(y, x, 2)) {
      this_ptr[0] = yp, this_ptr[1] = xp, this_ptr[2] = dp;
    }
    random_scale /= 2;
  }
}

const int PatchDistanceMetric::kDistanceScale = 65535;

#ifdef ENABLE_GRAD
const int PatchSSDDistanceMetric::kSSDScale = 9 * 255 * 255;
#else
const int PatchSSDDistanceMetric::kSSDScale = 3 * 255 * 255;
#endif  // ENABLE_GRAD

// distance between two patches in two images with early termination
int distance_masked_images(const MaskedImage &source, int ys, int xs, const MaskedImage &target,
                           int yt, int xt, int patch_size,
                           int baseline = PatchDistanceMetric::kDistanceScale) {
  const int step_size = 1;
  int sample_number = (2 * patch_size) / step_size + 1;
  uint32_t distance = 0;
  float nf = sample_number * sample_number * static_cast<float>(PatchSSDDistanceMetric::kSSDScale) /
             static_cast<float>(PatchDistanceMetric::kDistanceScale);
  uint32_t distance_bl = static_cast<uint32_t>(baseline * nf);

#ifndef MULTITHREAD
#ifdef ENABLE_GRAD
  source.compute_image_gradients();
  target.compute_image_gradients();
#endif  // ENABLE_GRAD
#endif  // MULTITHREAD

  cv::Size source_size = source.size(), target_size = target.size();
  for (int dy = -patch_size; dy <= patch_size; dy += step_size) {
    const int yys = ys + dy, yyt = yt + dy;

    // Either the given row of source or of target reaches the limit of the image
    if (yys <= 0 || yys >= source_size.height - 1 || yyt <= 0 || yyt >= target_size.height - 1) {
      distance += PatchSSDDistanceMetric::kSSDScale * sample_number;
      if (distance > distance_bl) return PatchDistanceMetric::kDistanceScale;
      continue;
    }

    const auto *p_si = source.image().ptr<unsigned char>(yys, 0);
    const auto *p_ti = target.image().ptr<unsigned char>(yyt, 0);
    const auto *p_sm = source.mask().ptr<unsigned char>(yys, 0);
    const auto *p_tm = target.mask().ptr<unsigned char>(yyt, 0);

    const unsigned char *p_sgm = nullptr;
    const unsigned char *p_tgm = nullptr;

#ifdef ENABLE_GRAD
    const auto *p_sgy = source.grady().ptr<unsigned char>(yys, 0);
    const auto *p_tgy = target.grady().ptr<unsigned char>(yyt, 0);
    const auto *p_sgx = source.gradx().ptr<unsigned char>(yys, 0);
    const auto *p_tgx = target.gradx().ptr<unsigned char>(yyt, 0);
#endif  //  ENABLE_GRAD

    for (int dx = -patch_size; dx <= patch_size; dx += step_size) {
      int xxs = xs + dx, xxt = xt + dx;
      // Either the given location of source or of target reaches the limit of the image
      if (xxs <= 0 || xxs >= source_size.width - 1 || xxt <= 0 || xxt >= target_size.width - 1) {
        distance += PatchSSDDistanceMetric::kSSDScale;
        if (distance > distance_bl) return PatchDistanceMetric::kDistanceScale;
        continue;
      }

      // Either the given location of source or of target falls outside the mask
      if (p_sm[xxs] || p_tm[xxt] || (p_sgm && p_sgm[xxs]) || (p_tgm && p_tgm[xxt])) {
        distance += PatchSSDDistanceMetric::kSSDScale;
        if (distance > distance_bl) return PatchDistanceMetric::kDistanceScale;
        continue;
      }

      int ssd = 0;
      // three channels

#pragma unroll
      for (int c = 0; c < 3; ++c) {
        int s_value = p_si[xxs * 3 + c];
        int t_value = p_ti[xxt * 3 + c];

#ifdef ENABLE_GRAD
        int s_gy = p_sgy[xxs * 3 + c];
        int t_gy = p_tgy[xxt * 3 + c];
        int s_gx = p_sgx[xxs * 3 + c];
        int t_gx = p_tgx[xxt * 3 + c];
#endif  // ENABLE_GRAD

        ssd += (s_value - t_value) * (s_value - t_value);

#ifdef ENABLE_GRAD
        ssd += (s_gx - t_gx) * (s_gx - t_gx);
        ssd += (s_gy - t_gy) * (s_gy - t_gy);
#endif  // ENABLE_GRAD
      }
      distance += ssd;
      if (distance > distance_bl) return PatchDistanceMetric::kDistanceScale;
    }
  }

  int res = static_cast<int>(distance / nf);
  if (res < 0 || res > PatchDistanceMetric::kDistanceScale)
    return PatchDistanceMetric::kDistanceScale;
  return res;
}

int PatchSSDDistanceMetric::operator()(const MaskedImage &source, int source_y, int source_x,
                                       const MaskedImage &target, int target_y, int target_x,
                                       int baseline) const {
  return distance_masked_images(source, source_y, source_x, target, target_y, target_x, _patch_size,
                                baseline);
}
