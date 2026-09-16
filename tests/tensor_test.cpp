#include "mytorch/tensor.h"

#include <cassert>
#include <stdexcept>
#include <vector>

int main() {
    mytorch::Tensor matrix{{1, 2, 3, 4, 5, 6}, {2, 3}};

    assert(matrix.data() == std::vector<float>({1, 2, 3, 4, 5, 6}));
    assert(matrix.shape() == std::vector<std::size_t>({2, 3}));
    assert(matrix.size() == 6);
    assert(matrix.rank() == 2);

    mytorch::Tensor scalar{{42}, {}};
    assert(scalar.size() == 1);
    assert(scalar.rank() == 0);

    bool threw = false;
    try {
        (void)mytorch::Tensor{{1, 2, 3}, {2, 2}};
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}
