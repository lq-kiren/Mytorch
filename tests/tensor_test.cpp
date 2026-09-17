#include "mytorch/tensor.h"

#include <cassert>
#include <limits>
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
    assert_invalid_argument([] {
        const std::size_t huge = std::numeric_limits<std::size_t>::max() / 2 + 1;
        (void)mytorch::Tensor{std::vector<float>{}, {huge, 2, 1}};
    });

    const mytorch::Tensor dot_lhs{{1, 2, 3}, {3}};
    const mytorch::Tensor dot_rhs{{4, 5, 6}, {3}};
    assert_tensor(dot_lhs.matmul(dot_rhs), {32}, {});

    const mytorch::Tensor vector_lhs{{1, 2}, {2}};
    const mytorch::Tensor matrix_rhs{{1, 2, 3, 4, 5, 6}, {2, 3}};
    assert_tensor(vector_lhs.matmul(matrix_rhs), {9, 12, 15}, {3});

    const mytorch::Tensor matrix_lhs{{1, 2, 3, 4, 5, 6}, {2, 3}};
    const mytorch::Tensor vector_rhs{{1, 2, 3}, {3}};
    assert_tensor(matrix_lhs.matmul(vector_rhs), {14, 32}, {2});

    const mytorch::Tensor matrix_product_rhs{{7, 8, 9, 10, 11, 12}, {3, 2}};
    assert_tensor(matrix_lhs.matmul(matrix_product_rhs), {58, 64, 139, 154}, {2, 2});

    const mytorch::Tensor batched_lhs{
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3}};
    const mytorch::Tensor unbatched_rhs{{1, 0, 1}, {3, 1}};
    assert_tensor(batched_lhs.matmul(unbatched_rhs), {4, 10, 16, 22}, {2, 2, 1});
    assert_tensor(batched_lhs.matmul(vector_rhs), {14, 32, 50, 68}, {2, 2});

    const mytorch::Tensor singleton_batch_lhs{{1, 2, 3, 4, 5, 6, 7, 8}, {2, 1, 2, 2}};
    const mytorch::Tensor singleton_batch_rhs{{1, 0, 0, 1, 1, 1}, {3, 2, 1}};
    assert_tensor(singleton_batch_lhs.matmul(singleton_batch_rhs),
                  {1, 3, 2, 4, 3, 7, 5, 7, 6, 8, 11, 15},
                  {2, 3, 2, 1});

    const mytorch::Tensor empty_matrix{std::vector<float>{}, {2, 0}};
    const mytorch::Tensor empty_rhs{std::vector<float>{}, {0, 3}};
    assert_tensor(empty_matrix.matmul(empty_rhs), {0, 0, 0, 0, 0, 0}, {2, 3});

    const mytorch::Tensor zero_batch{std::vector<float>{}, {0, 2, 3}};
    assert_tensor(zero_batch.matmul(matrix_product_rhs), {}, {0, 2, 2});

    const mytorch::Tensor zero_rows{std::vector<float>{}, {0, 3}};
    assert_tensor(zero_rows.matmul(matrix_product_rhs), {}, {0, 2});

    const mytorch::Tensor zero_columns{std::vector<float>{}, {3, 0}};
    assert_tensor(matrix_lhs.matmul(zero_columns), {}, {2, 0});

    const std::size_t huge = std::numeric_limits<std::size_t>::max() / 2 + 1;
    const mytorch::Tensor huge_empty_batch{std::vector<float>{}, {huge, 2, 0, 1}};
    const mytorch::Tensor one_by_one{{1}, {1, 1}};
    assert_tensor(huge_empty_batch.matmul(one_by_one), {}, {huge, 2, 0, 1});

    assert_invalid_argument([&] {
        (void)mytorch::Tensor{{1}, {}}.matmul(dot_rhs);
    });
    assert_invalid_argument([&] {
        (void)mytorch::Tensor{{1, 2, 3, 4}, {2, 2}}.matmul(
            mytorch::Tensor{{1, 2, 3}, {3, 1}});
    });
    assert_invalid_argument([&] {
        (void)mytorch::Tensor{{1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2}}.matmul(
            mytorch::Tensor{{1, 2, 3, 4, 5, 6}, {3, 2, 1}});
    });
}
