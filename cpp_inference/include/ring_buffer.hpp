#pragma once

#include <array>
#include <cstddef>
#include <optional>

namespace hybrid_arena {

template <typename T, std::size_t Capacity>
class RingBuffer {
 public:
  bool push(const T& item) {
    if (size_ == Capacity) {
      return false;
    }
    data_[(head_ + size_) % Capacity] = item;
    ++size_;
    return true;
  }

  std::optional<T> pop() {
    if (size_ == 0) {
      return std::nullopt;
    }
    T item = data_[head_];
    head_ = (head_ + 1) % Capacity;
    --size_;
    return item;
  }

  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

 private:
  std::array<T, Capacity> data_{};
  std::size_t head_ = 0;
  std::size_t size_ = 0;
};

}  // namespace hybrid_arena
