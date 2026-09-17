#include "mytorch/tensor.h"

#include <cassert>
#include <stdexcept>
#include <vector>
#include <iostream>

int main() {
    mytorch::Tensor matrix{{1, 2, 3, 4, 5, 6}, {2, 3}};

    std::cout << "Matrix: "<< matrix << std::endl;
    std::cout << "Matrix.info(): ";
    matrix.info();
    std::cout << "Matrix.shape().size(): " << matrix.shape().size() << std::endl;
    std::cout << "Matrix.size(): " << matrix.size() << std::endl;
    std::cout << "Matrix.rank(): " << matrix.rank() << std::endl;



    mytorch::Tensor matrix2{{6, 5, 4, 3, 2, 1}, {2, 3}};

    mytorch::Tensor sum = matrix + matrix2;
    std::cout << "Sum: " << sum << std::endl;
    mytorch::Tensor difference = matrix - matrix2;
    std::cout << "Difference: " << difference << std::endl;
    mytorch::Tensor product = matrix * matrix2;
    std::cout << "Product: " << product << std::endl;
    mytorch::Tensor quotient = matrix / matrix2;
    std::cout << "Quotient: " << quotient << std::endl;

    const mytorch::Tensor row{{1, 2, 3}, {3}};

    const mytorch::Tensor broadcast_sum = matrix + row;
    assert(broadcast_sum.shape() == std::vector<std::size_t>({2, 3}));
    assert(broadcast_sum.data() == std::vector<float>({2, 4, 6, 5, 7, 9}));

    const mytorch::Tensor broadcast_difference = matrix - row;
    assert(broadcast_difference.data() == std::vector<float>({0, 0, 0, 3, 3, 3}));

    const mytorch::Tensor reverse_difference = row - matrix;
    assert(reverse_difference.shape() == std::vector<std::size_t>({2, 3}));
    assert(reverse_difference.data() == std::vector<float>({0, 0, 0, -3, -3, -3}));

    const mytorch::Tensor broadcast_product = matrix * row;
    assert(broadcast_product.data() == std::vector<float>({1, 4, 9, 4, 10, 18}));

    const mytorch::Tensor broadcast_quotient = matrix / row;
    assert(broadcast_quotient.data() == std::vector<float>({1, 1, 1, 4, 2.5F, 2}));

    const mytorch::Tensor column{{10, 20}, {2, 1}};
    const mytorch::Tensor outer_sum = column + row;
    assert(outer_sum.shape() == std::vector<std::size_t>({2, 3}));
    assert(outer_sum.data() == std::vector<float>({11, 12, 13, 21, 22, 23}));

    const mytorch::Tensor scalar{{2}, {}};
    const mytorch::Tensor scaled = scalar * matrix;
    assert(scaled.shape() == std::vector<std::size_t>({2, 3}));
    assert(scaled.data() == std::vector<float>({2, 4, 6, 8, 10, 12}));

    const mytorch::Tensor empty{std::vector<float>{}, {0, 3}};
    const mytorch::Tensor singleton_row{{1, 2, 3}, {1, 3}};
    const mytorch::Tensor empty_sum = empty + singleton_row;
    assert(empty_sum.shape() == std::vector<std::size_t>({0, 3}));
    assert(empty_sum.data().empty());
    const mytorch::Tensor reverse_empty_sum = singleton_row + empty;
    assert(reverse_empty_sum.shape() == std::vector<std::size_t>({0, 3}));
    assert(reverse_empty_sum.data().empty());

    bool incompatible_shapes_threw = false;
    try {
        (void)(matrix + mytorch::Tensor{{1, 2, 3, 4}, {2, 2}});
    } catch (const std::invalid_argument&) {
        incompatible_shapes_threw = true;
    }
    assert(incompatible_shapes_threw);

    bool division_by_zero_threw = false;
    try {
        (void)(matrix / mytorch::Tensor{{1, 0, 1}, {3}});
    } catch (const std::invalid_argument&) {
        division_by_zero_threw = true;
    }
    assert(division_by_zero_threw);
}
