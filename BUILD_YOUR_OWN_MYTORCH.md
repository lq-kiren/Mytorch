# Build Your Own PyTorch Autograd in C++

本挑战参考 [Build Your Own grep](https://codingchallenges.fyi/challenges/challenge-grep/) 的形式：每一步只增加一个能力，并给出可运行的验收标准。目标不是复刻 PyTorch，而是亲手实现自动微分最核心的两条路径：

- 前向模式自动微分（Forward-mode AD / JVP）
- 反向模式自动微分（Reverse-mode AD / VJP）

## 挑战范围

必做部分从标量开始，最终扩展到只支持 `double`、一维/二维数据的最小 Tensor。

本挑战暂不包含：神经网络层、损失函数、优化器、GPU、并行计算、完整广播、混合精度、序列化和高阶导数。

建议全程使用 C++17 标准库。先写在一个源文件中；只有当文件确实难以阅读时再拆分。

## 难度由你选择

- **基础路线**：完成 Step 0–9，得到一个标量自动微分引擎。
- **完整路线**：继续完成 Step 10–12，得到一个很小但像 PyTorch 的 Tensor 自动微分核心。
- **困难路线**：最后完成 Going Beyond，加入更严格的工程能力。

---

## Step 0：准备最小测试环境

创建一个 C++17 程序，并准备一个浮点比较函数：

```cpp
bool close(double a, double b, double eps = 1e-9);
```

不要引入测试框架。每一步用 `assert` 验收即可。

```bash
g++ -std=c++17 -O2 mytorch.cpp -o mytorch
./mytorch
```

程序正常退出且没有断言失败，就算通过。

<details>
<summary>少量提示</summary>

浮点比较通常需要同时考虑绝对误差和相对误差。

</details>

---

## Step 1：让一个数携带导数

实现 `Dual` 类型，它只保存两个 `double`：

- `value`：函数值
- `tangent`：沿某个方向传播的导数

先实现加法和减法，使下面的检查通过：

```cpp
Dual x{2.0, 1.0};
Dual y = x + x - Dual{1.0, 0.0};

assert(close(y.value, 3.0));
assert(close(y.tangent, 2.0));
```

<details>
<summary>少量提示</summary>

加法的值相加，方向导数也相加。常量的 `tangent` 是 `0`。

</details>

---

## Step 2：实现乘法和除法

为 `Dual` 实现 `*` 和 `/`，然后求：

\[
f(x)=x^2+3x+1
\]

在 `x = 2` 处的值与导数。

```cpp
Dual x{2.0, 1.0};
Dual y = x * x + Dual{3.0, 0.0} * x + Dual{1.0, 0.0};

assert(close(y.value, 11.0));
assert(close(y.tangent, 7.0));
```

再为除数为零选择并固定一种行为，例如抛出 `std::domain_error`。

<details>
<summary>少量提示</summary>

只需要乘积法则与商法则。先把公式写在纸上，再翻译成代码。

</details>

---

## Step 3：加入非线性函数

为 `Dual` 实现以下自由函数：

```cpp
Dual sin(Dual x);
Dual exp(Dual x);
Dual log(Dual x);
Dual tanh(Dual x);
```

用它们计算：

\[
f(x)=\tanh(\log(x)+\sin(x))
\]

并把自动微分结果与中心差分比较：

\[
f'(x) \approx \frac{f(x+h)-f(x-h)}{2h}
\]

在 `x = 1.5`、`h = 1e-6` 时，两者误差应小于 `1e-6`。

<details>
<summary>少量提示</summary>

每个函数都只做两件事：计算新值，再把局部导数乘以上游 `tangent`。

</details>

---

## Step 4：计算多变量函数的 JVP

计算：

\[
f(x,y)=xy+\sin(x)
\]

在 `(x, y) = (2, 3)` 处，沿方向 `v = (1, -2)` 的方向导数。

```cpp
Dual x{2.0, 1.0};
Dual y{3.0, -2.0};
Dual z = x * y + sin(x);
```

验证 `z.tangent` 等于：

\[
J_f(x,y)v=(y+\cos x)\cdot1+x\cdot(-2)
\]

完成这一步后，你已经实现了 Jacobian-vector product（JVP）。

<details>
<summary>少量提示</summary>

不要显式构造 Jacobian。输入的 `tangent` 就是方向向量的分量。

</details>

---

## Step 5：用前向模式得到完整梯度

仍使用上一步的 `f(x, y)`。分别以 `(1, 0)` 和 `(0, 1)` 作为输入的 `tangent`，计算：

```text
df/dx = 3 + cos(2)
df/dy = 2
```

把两次结果组成梯度，并与中心差分逐项比较。

验收要求：每个分量误差小于 `1e-6`。

<details>
<summary>少量提示</summary>

前向模式一次传播只得到一个方向导数。对每个输入方向各运行一次即可。

</details>

---

## Step 6：记录反向计算图

现在开始实现反向模式。创建一个标量 `Value`，至少保存：

- 当前数值
- 当前梯度
- 产生它的父节点
- 把本节点梯度传播给父节点的规则

让 `+` 和 `*` 返回新的 `Value`，并记录动态计算图。此时先不实现 `backward()`。

构造：

```cpp
auto z = x * y + x;
```

验收要求：从 `z` 能找到两个直接输入；继续沿图向前能找到叶子 `x` 和 `y`。

<details>
<summary>少量提示</summary>

最省事的所有权方案是让一个 `Tape` 统一拥有节点，`Value` 只引用节点。先不要处理跨 Tape 运算。

</details>

---

## Step 7：实现第一次反向传播

实现：

```cpp
z.backward();
```

只需支持标量输出。调用时令 `z.grad = 1`，再按计算图的逆拓扑顺序传播梯度。

使用：

\[
z=xy+x
\]

在 `x = 2`、`y = 3` 处检查：

```text
z      = 8
dz/dx  = 4
dz/dy  = 2
```

<details>
<summary>少量提示</summary>

先深度优先遍历生成拓扑序，再反向遍历。不要在创建节点时立即求导。

</details>

---

## Step 8：正确处理分叉与梯度累加

反向模式最容易漏掉的是：同一个值可能沿多条路径影响输出。

验证：

\[
y=x^2+x
\]

在 `x = 3` 处应有：

```text
y     = 12
dy/dx = 7
```

再验证共享子表达式：

```cpp
auto a = x * x;
auto y = a + a;
```

在 `x = 3` 处，`dy/dx` 应为 `12`。

<details>
<summary>少量提示</summary>

传播给父节点时使用 `+=`，不是赋值。遍历图时，同一节点只加入拓扑序一次。

</details>

---

## Step 9：补齐算子并对拍两种 AD

为反向模式实现与前向模式相同的算子：

- `+`、`-`、`*`、`/`
- `sin`、`exp`、`log`、`tanh`

选择至少三个包含多层复合与重复使用变量的表达式。对每个表达式，同时计算：

1. 前向模式导数
2. 反向模式导数
3. 中心差分近似

三者应在 `1e-6` 误差内一致。

还要明确重复调用 `backward()` 的语义：要么自动清零图中梯度，要么要求调用者先执行 `zero_grad()`；二选一并写测试固定下来。

<details>
<summary>少量提示</summary>

前向 AD、反向 AD 和数值差分是三份独立实现，互相对拍比手写大量期望值更容易发现错误。

</details>

完成这里，你已经拥有一个类似 micrograd 的标量自动微分引擎。

---

## Step 10：支持 JVP 与 VJP 接口

前向模式已经隐式实现了 JVP。现在给两种模式各提供一个清晰接口：

```text
jvp(f, inputs, direction) -> (output, Jv)
vjp(f, inputs, cotangent) -> (output, vJ)
```

先用向量函数验收：

\[
f(x,y)=(xy,\ x+y)
\]

在 `(x, y) = (2, 3)` 处：

- 用前向模式计算方向 `(1, -1)` 的 JVP
- 用反向模式计算输出种子 `(2, -1)` 的 VJP

不允许在实现中显式构造完整 Jacobian。

<details>
<summary>少量提示</summary>

多个标量输出可以共享同一张图。VJP 可把各输出乘对应种子后求和，再做一次反向传播。

</details>

---

## Step 11：加入最小 Tensor

实现只支持 `double` 的 `Tensor`：

```cpp
Tensor({2, 3}, {1, 2, 3, 4, 5, 6});
```

只要求：

- 一维与二维 shape
- 连续的行优先存储
- 同 shape 的逐元素 `+` 与 `*`
- `sum()`，把所有元素归约为标量
- shape 不兼容时抛出异常

让前向模式通过：

```cpp
auto y = (x * x).sum();
```

其 JVP 应满足：

\[
d\sum_i x_i^2=\sum_i 2x_i\,dx_i
\]

再让反向模式对同一表达式得到逐元素梯度 `2x`。

<details>
<summary>少量提示</summary>

Tensor 的 `value`、`tangent` 和 `grad` 可以使用同一种小型数据容器。此阶段不要实现广播。

</details>

---

## Step 12：最终挑战——矩阵乘法

为二维 Tensor 实现矩阵乘法：

```cpp
Tensor matmul(const Tensor& a, const Tensor& b);
```

并使它同时支持前向与反向自动微分。

用以下标量输出做验收：

\[
L=\mathrm{sum}(AB)
\]

其中 `A` 为 `2×3`，`B` 为 `3×2`。

检查：

1. 前向值与手算结果一致
2. 随机方向上的 JVP 与中心差分一致
3. `A.grad`、`B.grad` 与逐元素中心差分一致
4. shape 不兼容时抛出异常

<details>
<summary>少量提示</summary>

先写三个普通循环得到前向结果。反向传播可从矩阵乘法的微分式推导，不需要任何线性代数库。

</details>

完成这一步后，你就拥有了一个最小的、支持动态计算图、JVP 与 VJP 的 C++ 自动微分核心。

---

## Going Beyond

基础实现稳定后，可以任选一项继续；不要一次全部加入。

- 用随机表达式做 property-based 对拍测试
- 检测 `log` 非法输入、除零和 shape 溢出
- 支持保留非叶子节点梯度
- 支持从指定种子开始的非标量 `backward(seed)`
- 支持有限广播，并为广播维度正确归约梯度
- 用模板把标量类型从 `double` 推广到 `float`
- 实现前向套前向或反向套反向，探索高阶导数

## 完成标准

在宣布挑战完成前，确认以下条件全部满足：

- 同一表达式的前向 AD、反向 AD 与中心差分结果一致
- 分叉图与共享子表达式能够正确累加梯度
- 反向传播严格按逆拓扑顺序执行
- JVP 与 VJP 不显式构造完整 Jacobian
- Tensor 的错误 shape 会被拒绝
- 所有验收都能通过一次命令运行

如果某个功能不在上述标准中，就先不要实现它。
