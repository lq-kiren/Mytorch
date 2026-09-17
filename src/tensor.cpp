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

Tensor Tensor::operator+(const Tensor& tensor) const {
    if (shape_ != tensor.shape_) {
        throw std::invalid_argument("tensor shapes do not match for addition");
    }

    std::vector<float> result_data(data_.size());
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result_data[i] = data_[i] + tensor.data_[i];
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::operator-(const Tensor& tensor) const {
    if (shape_ != tensor.shape_) {
        throw std::invalid_argument("tensor shapes do not match for subtraction");
    }

    std::vector<float> result_data(data_.size());
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result_data[i] = data_[i] - tensor.data_[i];
    }

    return Tensor(std::move(result_data), shape_);
}

Tensor Tensor::operator*(const Tensor& tensor) const {
    if (shape_ != tensor.shape_) {
        throw std::invalid_argument("tensor shapes do not match for multiplication");
    }

    std::vector<float> result_data(data_.size());
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result_data[i] = data_[i] * tensor.data_[i];
    }

    return Tensor(std::move(result_data), shape_);

}   

Tensor Tensor::operator/(const Tensor& tensor) const {
    if (shape_ != tensor.shape_) {
        throw std::invalid_argument("tensor shapes do not match for division");
    }

    std::vector<float> result_data(data_.size());
    for (std::size_t i = 0; i < data_.size(); ++i) {
        if (tensor.data_[i] == 0) {
            throw std::invalid_argument("division by zero in tensor division");
        }
        result_data[i] = data_[i] / tensor.data_[i];
    }

    return Tensor(std::move(result_data), shape_);
}

std::vector<std::size_t> Tensor::broadcast_shape(const Tensor tensor) const {
    std::vector<std::size_t> result;

    std::size_t i = shape_.size();
    std::size_t j = tensor.shape_.size();
    
    while (i > 0 || j > 0){
        std::size_t dim1 = (i > 0) ? shape_[i - 1] : 1;
        std::size_t dim2 = (j > 0) ? tensor.shape_[j - 1] : 1;

        if (dim1 == dim2 || dim1 == 1 || dim2 == 1) {
            result.push_back(std::max(dim1, dim2));
        } else {
            throw std::invalid_argument("tensor shapes are not broadcastable");
        }

        if (i > 0) --i;
        if (j > 0) --j;
    }
    std::reverse(result.begin(), result.end());
    return result;
}
}  // namespace mytorch
