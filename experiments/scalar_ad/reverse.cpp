#include <cassert>
#include <cstddef>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

struct Node;

// 一条从当前结果指向操作数的边。
// Step 7 反向传播时会执行：parent.grad += current.grad * local_derivative。
struct Edge {
    std::shared_ptr<Node> parent;
    double local_derivative;
};

struct Node {
    double data;                 // 前向计算得到的数值
    double grad = 0.0;           // 输出对本节点的梯度，初始为 0
    std::vector<Edge> parents;   // 生成本节点的操作数；叶子节点为空
};

// Value 是 Node 的轻量句柄。复制 Value 只会共享节点，不会复制计算图。
class Value {
public:
    // 直接由数值创建的是叶子节点，没有父节点。
    explicit Value(double data)
        : node_(std::make_shared<Node>(Node{data, 0.0, {}})) {}

    double data() const { return node_->data; }
    double grad() const { return node_->grad; }
    std::size_t parent_count() const { return node_->parents.size(); }

    Value parent(std::size_t index) const {
        return Value{node_->parents.at(index).parent};
    }

    double local_derivative(std::size_t index) const {
        return node_->parents.at(index).local_derivative;
    }

    bool same_node_as(const Value& other) const { return node_ == other.node_; }

    void backward();

private:
    // 运算符用这个构造函数把新 Value 包装在已创建的结果节点外面。
    explicit Value(std::shared_ptr<Node> node) : node_(std::move(node)) {}

    // 结果节点只持有父节点，父节点不持有结果节点，所以不会形成引用环。
    std::shared_ptr<Node> node_;

    static void build_topology(
        const std::shared_ptr<Node>& node,
        std::unordered_set<const Node*>& visited,
        std::vector<std::shared_ptr<Node>>& topology);

    // 运算符需要访问 node_，才能把 lhs、rhs 记录为新节点的父节点。
    friend Value operator+(const Value& lhs, const Value& rhs);
    friend Value operator*(const Value& lhs, const Value& rhs);
};

void Value::build_topology(
    const std::shared_ptr<Node>& node,
    std::unordered_set<const Node*>& visited,
    std::vector<std::shared_ptr<Node>>& topology) {
    // 共享子图可能从多条路径到达，同一节点只加入拓扑序一次。
    if (!visited.insert(node.get()).second) {
        return;
    }

    // 后序遍历：先放父节点，再放使用它们计算出的当前节点。
    for (const Edge& edge : node->parents) {
        build_topology(edge.parent, visited, topology);
    }
    topology.push_back(node);
}

void Value::backward() {
    std::unordered_set<const Node*> visited;
    std::vector<std::shared_ptr<Node>> topology;
    build_topology(node_, visited, topology);

    // 标量输出对自身的导数是 1，它是反向传播的起点。
    node_->grad = 1.0;

    // topology 是“叶子到输出”，所以反向迭代才能从输出走向叶子。
    for (auto current = topology.rbegin(); current != topology.rend(); ++current) {
        for (const Edge& edge : (*current)->parents) {
            // 使用 +=，因为同一父节点可能通过多条路径影响输出。
            edge.parent->grad += (*current)->grad * edge.local_derivative;
        }
    }
}

Value operator+(const Value& lhs, const Value& rhs) {
    auto node =
        std::make_shared<Node>(Node{lhs.data() + rhs.data(), 0.0, {}});
    // z = lhs + rhs，所以 dz/dlhs = 1，dz/drhs = 1。
    node->parents = {{lhs.node_, 1.0}, {rhs.node_, 1.0}};
    return Value{std::move(node)};
}

Value operator*(const Value& lhs, const Value& rhs) {
    auto node =
        std::make_shared<Node>(Node{lhs.data() * rhs.data(), 0.0, {}});
    // z = lhs * rhs，所以 dz/dlhs = rhs，dz/drhs = lhs。
    node->parents = {{lhs.node_, rhs.data()}, {rhs.node_, lhs.data()}};
    return Value{std::move(node)};
}

void test_step_6() {
    // 计算图：z(+) 的父节点为 [product, x]，
    // product(*) 的父节点为 [x, y]；两处 x 指向同一个节点。
    Value x{2.0};
    Value y{3.0};
    Value product = x * y;
    Value z = product + x;

    assert(z.data() == 8.0);
    assert(z.grad() == 0.0);
    assert(z.parent_count() == 2);
    assert(z.parent(0).same_node_as(product));
    assert(z.parent(1).same_node_as(x));
    assert(z.local_derivative(0) == 1.0);
    assert(z.local_derivative(1) == 1.0);

    assert(product.parent_count() == 2);
    assert(product.parent(0).same_node_as(x));
    assert(product.parent(1).same_node_as(y));
    assert(product.local_derivative(0) == 3.0);
    assert(product.local_derivative(1) == 2.0);

    assert(x.parent_count() == 0);
    assert(y.parent_count() == 0);
}

void test_step_7() {
    Value x{2.0};
    Value y{3.0};
    Value z = x * y + x;

    z.backward();

    assert(z.data() == 8.0);
    assert(z.grad() == 1.0);
    assert(x.grad() == 4.0);
    assert(y.grad() == 2.0);
}

int main() {
    test_step_6();
    test_step_7();
}
