#include "mytorch/tensor.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mytorch {

namespace {

std::vector<std::size_t> contiguous_strides(const std::vector<std::size_t>& shape) {
    // 行优先连续布局，例如 shape={2,3,4} 对应 strides={12,4,1}。
    std::vector<std::size_t> strides(shape.size(), 1);
    for (std::size_t i = shape.size(); i > 1; --i) {
        strides[i - 2] = strides[i - 1] * shape[i - 1];
    }
    return strides;
}

template <typename Operation>
Tensor elementwise(const Tensor& lhs, const Tensor& rhs, Operation operation) {
    // 只生成结果数据，不实际复制或展开参与广播的输入。
    std::vector<std::size_t> result_shape = lhs.broadcast_shape(rhs);
    const std::vector<std::size_t> lhs_strides = contiguous_strides(lhs.shape());
    const std::vector<std::size_t> rhs_strides = contiguous_strides(rhs.shape());

    std::size_t result_size = 1;
    for (std::size_t dimension : result_shape) {
        result_size *= dimension;
    }

    std::vector<float> result_data(result_size);
    // rank 较小的输入在左侧补 1，使两个 shape 从末维对齐。
    const std::size_t lhs_leading_dimensions = result_shape.size() - lhs.rank();
    const std::size_t rhs_leading_dimensions = result_shape.size() - rhs.rank();

    for (std::size_t result_index = 0; result_index < result_size; ++result_index) {
        std::size_t remaining = result_index;
        std::size_t lhs_index = 0;
        std::size_t rhs_index = 0;

        // 将结果的一维下标从末维开始拆成多维坐标。
        for (std::size_t result_dimension = result_shape.size(); result_dimension-- > 0;) {
            const std::size_t coordinate = remaining % result_shape[result_dimension];
            remaining /= result_shape[result_dimension];

            if (result_dimension >= lhs_leading_dimensions) {
                const std::size_t lhs_dimension = result_dimension - lhs_leading_dimensions;
                // 广播维长度为 1，该输入始终读取坐标 0。
                if (lhs.shape()[lhs_dimension] != 1) {
                    lhs_index += coordinate * lhs_strides[lhs_dimension];
                }
            }

            if (result_dimension >= rhs_leading_dimensions) {
                const std::size_t rhs_dimension = result_dimension - rhs_leading_dimensions;
                if (rhs.shape()[rhs_dimension] != 1) {
                    rhs_index += coordinate * rhs_strides[rhs_dimension];
                }
            }
        }

        result_data[result_index] = operation(lhs.data()[lhs_index], rhs.data()[rhs_index]);
    }

    return Tensor(std::move(result_data), std::move(result_shape));
}

}  // namespace

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
        // 到达最内层维度时，输出当前扁平存储中的元素。
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
    return elementwise(*this, tensor, std::plus<float>{});
}

Tensor Tensor::operator-(const Tensor& tensor) const {
    return elementwise(*this, tensor, std::minus<float>{});
}

Tensor Tensor::operator*(const Tensor& tensor) const {
    return elementwise(*this, tensor, std::multiplies<float>{});
}   

Tensor Tensor::operator/(const Tensor& tensor) const {
    return elementwise(*this, tensor, [](float lhs, float rhs) {
        if (rhs == 0) {
            throw std::invalid_argument("division by zero in tensor division");
        }
        return lhs / rhs;
    });
}

std::vector<std::size_t> Tensor::broadcast_shape(const Tensor& tensor) const {
    std::vector<std::size_t> result;

    std::size_t i = shape_.size();
    std::size_t j = tensor.shape_.size();
    
    // 从末维向前比较；缺失的前导维按长度 1 处理。
    while (i > 0 || j > 0){
        std::size_t dim1 = (i > 0) ? shape_[i - 1] : 1;
        std::size_t dim2 = (j > 0) ? tensor.shape_[j - 1] : 1;

        if (dim1 == dim2) {
            result.push_back(dim1);
        } else if (dim1 == 1) {
            result.push_back(dim2);
        } else if (dim2 == 1) {
            result.push_back(dim1);
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
