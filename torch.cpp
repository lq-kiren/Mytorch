#include <algorithm>
#include <cassert>
#include <cmath>

bool close(double a, double b, double eps = 1e-9){
    return std::abs(a - b) <= eps * std::max({1.0, std::abs(a), std::abs(b)});
}

int main() {
    assert(close(0.1 + 0.2, 0.3));
    assert(close(1e9, 1e9 + 0.5));
}
