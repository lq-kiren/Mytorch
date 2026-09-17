#include "mytorch/tensor.h"

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

class FlushTrackingBuffer : public std::stringbuf {
public:
    bool flushed = false;

protected:
    int sync() override {
        flushed = true;
        return std::stringbuf::sync();
    }
};

void assert_tensor(const mytorch::Tensor& tensor,
                   const std::vector<float>& expected_data,
                   const std::vector<std::size_t>& expected_shape) {
    assert(tensor.data() == expected_data);
    assert(tensor.shape() == expected_shape);
}

template <typename Operation>
void assert_invalid_argument(Operation operation) {
    bool threw = false;
    try {
        operation();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

}  // namespace

int main() {
    const mytorch::Tensor matrix{{1, 2, 3, 4, 5, 6}, {2, 3}};

    FlushTrackingBuffer info_buffer;
    std::ostream info_output(&info_buffer);
    matrix.info(info_output);
    assert(info_buffer.str() == "Tensor(shape={2, 3}, data={1, 2, 3, 4, 5, 6})\n");
    assert(info_buffer.flushed);

    std::ostringstream tensor_output;
    tensor_output << matrix;
    assert(tensor_output.str() == "[[1, 2, 3], [4, 5, 6]]");

    const mytorch::Tensor matrix2{{6, 5, 4, 3, 2, 1}, {2, 3}};
    assert_tensor(matrix + matrix2, {7, 7, 7, 7, 7, 7}, {2, 3});

    const mytorch::Tensor row{{1, 2, 3}, {3}};
    assert_tensor(matrix + row, {2, 4, 6, 5, 7, 9}, {2, 3});
    assert_tensor(matrix - row, {0, 0, 0, 3, 3, 3}, {2, 3});
    assert_tensor(row - matrix, {0, 0, 0, -3, -3, -3}, {2, 3});
    assert_tensor(matrix * row, {1, 4, 9, 4, 10, 18}, {2, 3});
    assert_tensor(matrix / row, {1, 1, 1, 4, 2.5F, 2}, {2, 3});

    const mytorch::Tensor column{{10, 20}, {2, 1}};
    assert_tensor(column + row, {11, 12, 13, 21, 22, 23}, {2, 3});

    const mytorch::Tensor scalar{{2}, {}};
    assert_tensor(scalar * matrix, {2, 4, 6, 8, 10, 12}, {2, 3});

    const mytorch::Tensor empty{std::vector<float>{}, {0, 3}};
    const mytorch::Tensor singleton_row{{1, 2, 3}, {1, 3}};
    assert_tensor(empty + singleton_row, {}, {0, 3});
    assert_tensor(singleton_row + empty, {}, {0, 3});

    assert_invalid_argument([&] {
        (void)(matrix + mytorch::Tensor{{1, 2, 3, 4}, {2, 2}});
    });
    assert_invalid_argument([&] {
        (void)(matrix / mytorch::Tensor{{1, 0, 1}, {3}});
    });
    assert_invalid_argument([] {
        (void)mytorch::Tensor{{1, 2, 3}, {2, 2}};
    });
}
