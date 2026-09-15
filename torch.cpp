#include <algorithm>
#include <iostream>
#include <cassert>
#include <cmath>


class Dual{
    public:
        double value;
        double tangent;

        Dual(double value, double tangent=0.0) : value(value), tangent(tangent) {}

        Dual operator+(const Dual& other) const {
            return Dual(value + other.value, tangent + other.tangent);
        }

        Dual operator-(const Dual& other) const {
            return Dual(value - other.value, tangent - other.tangent);
        }

        Dual operator*(const Dual& other) const {
            return Dual(value * other.value, value * other.tangent + tangent * other.value);
        }

        Dual operator/(const Dual& other) const {
            return Dual(value / other.value, (tangent * other.value - value * other.tangent) / (other.value * other.value));
        }

        Dual sin() const {
            return Dual(std::sin(value), tangent * std::cos(value));
        }

        Dual cos() const {
            return Dual(std::cos(value), -tangent * std::sin(value));
        }

        Dual exp() const {
            return Dual(std::exp(value), tangent * std::exp(value));
        }

        Dual log() const {
            return Dual(std::log(value), tangent / value);
        }

        Dual tanh() const {
            return Dual(std::tanh(value), tangent / (std::cosh(value) * std::cosh(value)));
        }

        Dual pow(const Dual& other) const {
            return Dual(std::pow(value, other.value), std::pow(value, other.value) * (other.tangent * std::log(value) + tangent * other.value / value));
        }

        Dual operator-() const {
            return Dual(-value, -tangent);
        }

        Dual operator+(double other) const {
            return Dual(value + other, tangent);
        }
};

Dual sin(const Dual& x) {
    return x.sin();
}

Dual cos(const Dual& x) {
    return x.cos();
}

Dual exp(const Dual& x) {
    return x.exp();
}

Dual log(const Dual& x) {
    return x.log();
}

Dual tanh(const Dual& x) {
    return x.tanh();
}

bool close(double a, double b, double eps = 1e-9){
    return std::abs(a - b) <= eps * std::max({1.0, std::abs(a), std::abs(b)});
}

int main() {
    Dual x{2.0, 1.0};
    Dual y{3.0, -2.0};
    Dual z = x * y + sin(x);

    std::cout<<"z.value: " << z.value << ", z.tangent: " << z.tangent << std::endl;
}
