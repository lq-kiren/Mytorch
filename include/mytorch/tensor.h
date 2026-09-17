#pragma once

#include <ostream>
#include <cstddef>
#include <vector>

namespace mytorch {

class Tensor {
public:
    Tensor(std::vector<float> data, std::vector<std::size_t> shape);

    const std::vector<float>& data() const noexcept { return data_; }
    const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    std::size_t size() const noexcept { return data_.size(); }
    std::size_t rank() const noexcept { return shape_.size(); }

    void info() const noexcept;
    friend std::ostream& operator<<(std::ostream& output, const Tensor& tensor);

    std::vector<std::size_t> broadcast_shape(const Tensor& other) const;

    Tensor operator+(const Tensor& tensor) const;
    Tensor operator-(const Tensor& tensor) const;
    Tensor operator*(const Tensor& tensor) const;
    Tensor operator/(const Tensor& tensor) const;

private:
    // Contiguous row-major storage; views are not supported yet.
    std::vector<float> data_;
    std::vector<std::size_t> shape_;
};

}  // namespace mytorch
