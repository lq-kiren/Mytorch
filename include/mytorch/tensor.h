#pragma once

#include <cstddef>
#include <iosfwd>
#include <vector>

namespace mytorch {

class Tensor {
public:
    Tensor(std::vector<float> data, std::vector<std::size_t> shape);

    const std::vector<float>& data() const noexcept { return data_; }
    const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    std::size_t size() const noexcept { return data_.size(); }
    std::size_t rank() const noexcept { return shape_.size(); }

    void info() const;
    void info(std::ostream& output) const;

    Tensor operator+(const Tensor& tensor) const;
    Tensor operator-(const Tensor& tensor) const;
    Tensor operator*(const Tensor& tensor) const;
    Tensor operator/(const Tensor& tensor) const;

    Tensor matmul(const Tensor& tensor) const;

private:
    // Contiguous row-major storage; views are not supported yet.
    std::vector<float> data_;
    std::vector<std::size_t> shape_;
};

std::ostream& operator<<(std::ostream& output, const Tensor& tensor);

}  // namespace mytorch
