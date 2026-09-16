#include "mytorch/tensor.h"

#include <stdexcept>
#include <utility>

namespace mytorch {

Tensor::Tensor(std::vector<float> data, std::vector<std::size_t> shape)
    : data_(std::move(data)), shape_(std::move(shape)) {
    std::size_t expected_size = 1;
    for (std::size_t dimension : shape_) {
        expected_size *= dimension;
    }

    if (expected_size != data_.size()) {
        throw std::invalid_argument("tensor data size does not match shape");
    }
}

}  // namespace mytorch
