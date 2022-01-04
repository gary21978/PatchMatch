#include "masked_image.h"

const cv::Size MaskedImage::kDownsampleKernelSize = cv::Size(6, 6);
const int MaskedImage::kDownsampleKernel[6] = {1, 5, 10, 10, 5, 1};

bool MaskedImage::contains_mask(int y, int x, int patch_size) const {
  cv::Size mask_size = size();
  for (int dy = -patch_size; dy <= patch_size; ++dy) {
    int yy = y + dy;
    if (yy < 0 || yy >= mask_size.height) continue;
    for (int dx = -patch_size; dx <= patch_size; ++dx) {
      int xx = x + dx;
      if (xx < 0 || xx >= mask_size.width) continue;

      if (is_masked(yy, xx)) return true;
    }
  }
  return false;
}

MaskedImage MaskedImage::downsample(bool *covered) const {
  const cv::Size &kernel_size = MaskedImage::kDownsampleKernelSize;
  const auto &kernel = MaskedImage::kDownsampleKernel;

  const cv::Size size = this->size();
  MaskedImage ret(size.width / 2, size.height / 2);
#ifndef MULTITHREAD
  for (int y = 0; y < size.height - 1; y += 2) {
    for (int x = 0; x < size.width - 1; x += 2) {
      int r = 0, g = 0, b = 0, ksum = 0;

      for (int dy = -kernel_size.height / 2 + 1; dy <= kernel_size.height / 2; ++dy) {
        int yy = y + dy;
        if (yy < 0 || yy >= size.height) continue;

        for (int dx = -kernel_size.width / 2 + 1; dx <= kernel_size.width / 2; ++dx) {
          int xx = x + dx;
          if (xx < 0 || xx >= size.width) continue;
          if (!is_masked(yy, xx)) {
            auto source_ptr = get_image(yy, xx);
            int k =
                kernel[kernel_size.height / 2 - 1 + dy] * kernel[kernel_size.width / 2 - 1 + dx];
            r += source_ptr[0] * k, g += source_ptr[1] * k, b += source_ptr[2] * k;
            ksum += k;
          }
        }
      }

      if (ksum > 0) {
        r /= ksum, g /= ksum, b /= ksum;
        auto target_ptr = ret.get_mutable_image(y / 2, x / 2);
        target_ptr[0] = r, target_ptr[1] = g, target_ptr[2] = b;
        ret.set_mask(y / 2, x / 2, 0);
      } else {
        ret.set_mask(y / 2, x / 2, 1);
      }
    }
  }

#else
  auto processor = [ this, &ret, &size ](const int &start, const int &end) -> void {
    for (int index = start; index < end; ++index) {
      const int &i = index / size.width;
      const int &j = index % size.width;
      if (i % 2 != 0 || j % 2 != 0 || i == size.height - 1 || j == size.width - 1) {
        continue;
      }
      int r = 0, g = 0, b = 0, ksum = 0;
      for (int dy = -kernel_size.height / 2 + 1; dy <= kernel_size.height / 2; ++dy) {
        const int &yy = i + dy;
        if (yy < 0 || yy >= size.height) {
          continue;
        }
        for (int dx = -kernel_size.width / 2 + 1; dx <= kernel_size.width / 2; ++dx) {
          const int &xx = j + dx;
          if (xx < 0 || xx >= size.width) {
            continue;
          }
          if (!this->is_masked(yy, xx)) {
            auto source_ptr = this->get_image(yy, xx);
            const int k =
                kernel[kernel_size.height / 2 - 1 + dy] * kernel[kernel_size.width / 2 - 1 + dx];
            r += source_ptr[0] * k;
            g += source_ptr[1] * k;
            b += source_ptr[2] * k;
            ksum += k;
          }
        }
      }
      if (ksum > 0) {
        auto target_ptr = ret.get_mutable_image(i / 2, j / 2);
        target_ptr[0] = r / ksum;
        target_ptr[1] = g / ksum;
        target_ptr[2] = b / ksum;
      } else {
        ret.set_mask(i / 2, j / 2, 1);
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
  for (auto &&iter = ret._mask.begin<uchar>(); iter != ret._mask.end<uchar>(); ++iter) {
    if (*iter > 0) {
      *covered = false;
      break;
    }
  }
#endif  // MULTITHREAD

  return ret;
}

MaskedImage MaskedImage::upsample(int new_w, int new_h) const {
  const cv::Size size = this->size();
  MaskedImage ret(new_w, new_h);
#ifndef MULTITHREAD
  for (int y = 0; y < new_h; ++y) {
    int yy = y * size.height / new_h;
    for (int x = 0; x < new_w; ++x) {
      int xx = x * size.width / new_w;
      if (is_masked(yy, xx)) {
        ret.set_mask(y, x, 1);
      } else {
        auto source_ptr = get_image(yy, xx);
        auto target_ptr = ret.get_mutable_image(y, x);
        target_ptr[0] = source_ptr[0];
        target_ptr[1] = source_ptr[1];
        target_ptr[2] = source_ptr[2];
        ret.set_mask(y, x, 0);
      }
    }
  }
#else
  auto processor =
      [ this, size, &ret, &new_w, &new_h ](const int &start, const int &end) -> void {
    for (int index = start; index < end; ++index) {
      const int &i = index / new_w;
      const int &j = index % new_w;
      const int &yy = i * size.height / new_h;
      const int &xx = j * size.width / new_w;
      if (this->is_masked(yy, xx)) {
        ret.set_mask(i, j, 1);
      } else {
        auto source_ptr = this->get_image(yy, xx);
        auto target_ptr = ret.get_mutable_image(i, j);
        target_ptr[0] = source_ptr[0];
        target_ptr[1] = source_ptr[1];
        target_ptr[2] = source_ptr[2];
      }
    }
  };
  const int pts_num = new_h * new_w;
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
  return ret;
}

void MaskedImage::compute_image_gradients() {
  if (_image_grad_computed) {
    return;
  }

  const cv::Size size = _image.size();
  _image_grady = cv::Mat::zeros(size, CV_8UC3);
  _image_gradx = cv::Mat::zeros(size, CV_8UC3);

  for (int i = 1; i < size.height - 1; ++i) {
    const auto *ptr = _image.ptr<unsigned char>(i, 0);
    const auto *ptry1 = _image.ptr<unsigned char>(i + 1, 0);
    const auto *ptry2 = _image.ptr<unsigned char>(i - 1, 0);
    const auto *ptrx1 = _image.ptr<unsigned char>(i, 0) + 3;
    const auto *ptrx2 = _image.ptr<unsigned char>(i, 0) - 3;
    auto *mptry = _image_grady.ptr<unsigned char>(i, 0);
    auto *mptrx = _image_gradx.ptr<unsigned char>(i, 0);
    for (int j = 3; j < size.width * 3 - 3; ++j) {
      mptry[j] = (ptry1[j] / 2 - ptry2[j] / 2) + 128;
      mptrx[j] = (ptrx1[j] / 2 - ptrx2[j] / 2) + 128;
    }
  }

  _image_grad_computed = true;
}

#ifndef MULTITHREAD
void MaskedImage::compute_image_gradients() const {
  const_cast<MaskedImage *>(this)->compute_image_gradients();
}
#endif
