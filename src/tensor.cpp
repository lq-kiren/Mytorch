#include "mytorch/tensor.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mytorch {

namespace {

std::size_t element_count(const std::vector<std::size_t>& shape) {
    std::size_t count = 1;
    for (std::size_t dimension : shape) {
        count *= dimension;
    }
    return count;
}

template <typename Values>
void print_values(std::ostream& os, const Values& values) {
    const char* separator = "";
    for (const auto& value : values) {
        os << separator << value;
        separator = ", ";
    }
}

void print_tensor(std::ostream& os,
                  const std::vector<float>& data,
                  const std::vector<std::size_t>& shape,
                  std::size_t dimension,
                  std::size_t& index) {
    // 到达最内层维度时，输出当前扁平存储中的元素。
    if (dimension == shape.size()) {
        os << data[index++];
        return;
    }

    os << '[';
    for (std::size_t i = 0; i < shape[dimension]; ++i) {
        if (i > 0) {
            os << ", ";
        }
        print_tensor(os, data, shape, dimension + 1, index);
    }
    os << ']';
}

std::vector<std::size_t> broadcast_shape(const std::vector<std::size_t>& lhs,
                                         const std::vector<std::size_t>& rhs) {
    std::vector<std::size_t> result;
    std::size_t i = lhs.size();
    std::size_t j = rhs.size();

    // 从末维向前比较；缺失的前导维按长度 1 处理。
    while (i > 0 || j > 0) {
        const std::size_t lhs_dimension = i > 0 ? lhs[i - 1] : 1;
        const std::size_t rhs_dimension = j > 0 ? rhs[j - 1] : 1;

        if (lhs_dimension == rhs_dimension) {
            result.push_back(lhs_dimension);
        } else if (lhs_dimension == 1) {
            result.push_back(rhs_dimension);
        } else if (rhs_dimension == 1) {
            result.push_back(lhs_dimension);
        } else {
            throw std::invalid_argument("tensor shapes are not broadcastable");
        }

        if (i > 0) --i;
        if (j > 0) --j;
    }

    std::reverse(result.begin(), result.end());
    return result;
}

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
    std::vector<std::size_t> result_shape = broadcast_shape(lhs.shape(), rhs.shape());
    const std::vector<std::size_t> lhs_strides = contiguous_strides(lhs.shape());
    const std::vector<std::size_t> rhs_strides = contiguous_strides(rhs.shape());

    const std::size_t result_size = element_count(result_shape);
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
    if (element_count(shape_) != data_.size()) {
        throw std::invalid_argument("tensor data size does not match shape");
    }
}

void Tensor::info() const {
    info(std::cout);
}

void Tensor::info(std::ostream& os) const {
    os << "Tensor(shape={";
    print_values(os, shape_);
    os << "}, data={";
    print_values(os, data_);
    os << "})" << std::endl;
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    std::size_t index = 0;
    print_tensor(os, tensor.data(), tensor.shape(), 0, index);
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

}  // namespace mytorch
