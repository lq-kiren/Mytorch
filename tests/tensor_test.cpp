#include "mytorch/tensor.h"

#include <cassert>
#include <stdexcept>
#include <vector>
#include <iostream>

int main() {
    mytorch::Tensor matrix{{1, 2, 3, 4, 5, 6}, {2, 3}};

    std::cout << "Matrix: "<< matrix << std::endl;
    std::cout << "Matrix.size(): " << matrix.size() << std::endl;
    std::cout << "Matrix.rank(): " << matrix.rank() << std::endl;

    assert(matrix.data() == std::vector<float>({1, 2, 3, 4, 5, 6}));
    assert(matrix.shape() == std::vector<std::size_t>({2, 3}));

    mytorch::Tensor scalar{{42}, {}};
    assert(scalar.size() == 1);
    assert(scalar.rank() == 0);

}
