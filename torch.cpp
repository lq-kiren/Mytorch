#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>

class Dual {
public:
    explicit Dual(double value, double tangent = 0.0)
        : value_(value), tangent_(tangent) {}

    double value() const { return value_; }
    double tangent() const { return tangent_; }

private:
    double value_;
    double tangent_;
};

Dual operator+(Dual lhs, Dual rhs) {
    return Dual{
        lhs.value() + rhs.value(),
        lhs.tangent() + rhs.tangent(),
    };
}

Dual operator-(Dual lhs, Dual rhs) {
    return Dual{
        lhs.value() - rhs.value(),
        lhs.tangent() - rhs.tangent(),
    };
}

Dual operator*(Dual lhs, Dual rhs) {
    return Dual{
        lhs.value() * rhs.value(),
        lhs.tangent() * rhs.value() + lhs.value() * rhs.tangent(),
    };
}

Dual operator/(Dual lhs, Dual rhs) {
    if (rhs.value() == 0.0) {
        throw std::domain_error("division by zero");
    }
    return Dual{
        lhs.value() / rhs.value(),
        (lhs.tangent() * rhs.value() - lhs.value() * rhs.tangent()) /
            (rhs.value() * rhs.value()),
    };
}

Dual sin(Dual x) {
    return Dual{std::sin(x.value()), x.tangent() * std::cos(x.value())};
}

Dual exp(Dual x) {
    const double value = std::exp(x.value());
    return Dual{value, x.tangent() * value};
}

Dual log(Dual x) {
    return Dual{std::log(x.value()), x.tangent() / x.value()};
}

Dual tanh(Dual x) {
    const double value = std::tanh(x.value());
    return Dual{value, x.tangent() * (1.0 - value * value)};
}

bool close(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) <= eps * std::max({1.0, std::abs(a), std::abs(b)});
}

void test_step_0() {
    assert(close(0.1 + 0.2, 0.3));
    assert(close(1e9, 1e9 + 0.5));
}

void test_step_1() {
    Dual x{2.0, 1.0};
    Dual y = x + x - Dual{1.0};

    assert(close(y.value(), 3.0));
    assert(close(y.tangent(), 2.0));
}

void test_step_2() {
    Dual x{2.0, 1.0};
    Dual y = x * x + Dual{3.0} * x + Dual{1.0};

    assert(close(y.value(), 11.0));
    assert(close(y.tangent(), 7.0));

    bool threw = false;
    try {
        (void)(x / Dual{0.0});
    } catch (const std::domain_error&) {
        threw = true;
    }
    assert(threw);
}

double step_3_function(double x) {
    return std::tanh(std::log(x) + std::sin(x));
}

void test_step_3() {
    constexpr double point = 1.5;
    constexpr double h = 1e-6;
    Dual x{point, 1.0};
    Dual y = tanh(log(x) + sin(x));
    double numerical =
        (step_3_function(point + h) - step_3_function(point - h)) / (2.0 * h);

    assert(close(y.tangent(), numerical, 1e-6));

    Dual e = exp(Dual{1.0, 1.0});
    assert(close(e.value(), std::exp(1.0)));
    assert(close(e.tangent(), std::exp(1.0)));
}

void test_step_4() {
    Dual x{2.0, 1.0};
    Dual y{3.0, -2.0};
    Dual z = x * y + sin(x);

    assert(close(z.value(), 6.0 + std::sin(2.0)));
    assert(close(z.tangent(), -1.0 + std::cos(2.0)));
}


Dual step_5_function(Dual x, Dual y) {
    return x * y + sin(x);
}

double step_5_function(double x, double y) {
    return x * y + std::sin(x);
}

void test_step_5() {
    Dual x1{2.0, 1.0};
    Dual y1{3.0, 0.0};
    double dzdx = step_5_function(x1, y1).tangent();

    Dual x2{2.0, 0.0};
    Dual y2{3.0, 1.0};
    double dzdy = step_5_function(x2, y2).tangent();

    std::array<double, 2> gradient{dzdx, dzdy};

    constexpr double h = 1e-6;
    double numerical_dfdx =
        (step_5_function(2.0 + h, 3.0) - step_5_function(2.0 - h, 3.0)) /
        (2.0 * h);
    double numerical_dfdy =
        (step_5_function(2.0, 3.0 + h) - step_5_function(2.0, 3.0 - h)) /
        (2.0 * h);

    assert(close(gradient[0], 3.0 + std::cos(2.0)));
    assert(close(gradient[1], 2.0));
    assert(close(gradient[0], numerical_dfdx, 1e-6));
    assert(close(gradient[1], numerical_dfdy, 1e-6));
}

int main() {
    test_step_0();
    test_step_1();
    test_step_2();
    test_step_3();
    test_step_4();
    test_step_5();
}
