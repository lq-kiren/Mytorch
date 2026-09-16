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
}
