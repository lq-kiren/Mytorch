#include "mytorch/tensor.h"

#include <vector>
#include <functional>
#include <iostream>
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

void Tensor::info() const noexcept {
    auto print = [](const auto& values) {
        const char* separator = "";
        for (const auto& value : values) {
            std::cout << separator << value;
            separator = ", ";
        }
    };

    std::cout << "Tensor(shape={";
    print(shape_);
    std::cout << "}, data={";
    print(data_);
    std::cout << "})" << std::endl;
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    
    std::function<void(std::size_t, std::size_t&)> print_recursive;
    print_recursive = [&](std::size_t dim, std::size_t& index) {
        if (dim == tensor.shape_.size()) {
            os << tensor.data_[index++];
            return;
        }

        os << "[";
        for (std::size_t i = 0; i < tensor.shape_[dim]; ++i) {
            if (i > 0) {
                os << ", ";
            }
            print_recursive(dim + 1, index);
        }
        os << "]";
    };

    std::size_t index = 0;
    print_recursive(0, index);

    return os;
}

}  // namespace mytorch
