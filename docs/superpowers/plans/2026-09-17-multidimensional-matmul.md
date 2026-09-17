# Multidimensional Tensor Matmul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement PyTorch-style `Tensor::matmul` semantics for vectors, matrices, and broadcast batches.

**Architecture:** Virtually promote one-dimensional operands to matrices, broadcast only their leading batch dimensions, and compute directly against flat row-major storage. Do not materialize expanded inputs or add persistent stride/view state.

**Tech Stack:** C++17, CMake, CTest, standard library only

**Spec:** `docs/superpowers/specs/2026-09-17-multidimensional-matmul-design.md`

## Global Constraints

- Keep `Tensor matmul(const Tensor& tensor) const` as the public interface.
- Scalar operands are invalid.
- Preserve existing elementwise broadcasting, formatting, and constructor behavior.
- Add no dependencies, new classes, persistent strides, views, dtype promotion, BLAS, or autograd.
- Use the existing flat contiguous row-major storage.

---

### Task 1: Implement multidimensional matmul

**Files:**
- Modify: `include/mytorch/tensor.h`
- Modify: `src/tensor.cpp`
- Test: `tests/tensor_test.cpp`

**Interfaces:**
- Consumes: `Tensor::data()`, `Tensor::shape()`, `Tensor::rank()`, `element_count`, `broadcast_shape`, and `contiguous_strides`
- Produces: `Tensor Tensor::matmul(const Tensor& tensor) const`

- [ ] **Step 1: Write failing behavior tests**

Append these cases to `main()` in `tests/tensor_test.cpp`:

```cpp
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

const mytorch::Tensor singleton_batch_lhs{{1, 2, 3, 4, 5, 6, 7, 8}, {2, 1, 2, 2}};
const mytorch::Tensor singleton_batch_rhs{{1, 0, 0, 1, 1, 1}, {1, 3, 2, 1}};
assert_tensor(singleton_batch_lhs.matmul(singleton_batch_rhs),
              {1, 3, 2, 4, 3, 7, 5, 7, 6, 8, 11, 15},
              {2, 3, 2, 1});

const mytorch::Tensor empty_matrix{std::vector<float>{}, {2, 0}};
const mytorch::Tensor empty_rhs{std::vector<float>{}, {0, 3}};
assert_tensor(empty_matrix.matmul(empty_rhs), {0, 0, 0, 0, 0, 0}, {2, 3});

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
```

These literals independently verify every output value and shape. They catch
wrong vector squeezing, wrong matrix indexing, missing batch broadcasting,
incorrect singleton handling, and missing validation.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
cmake --build build
```

Expected: link failure reporting an undefined reference to
`mytorch::Tensor::matmul(const mytorch::Tensor&) const`.

- [ ] **Step 3: Add the public declaration if absent**

Ensure the public section of `include/mytorch/tensor.h` contains:

```cpp
Tensor matmul(const Tensor& tensor) const;
```

- [ ] **Step 4: Add a batch-offset helper**

Add this file-local helper after `contiguous_strides` in `src/tensor.cpp`:

```cpp
std::size_t broadcast_batch_offset(
    std::size_t batch_index,
    const std::vector<std::size_t>& batch_shape,
    const std::vector<std::size_t>& operand_shape,
    const std::vector<std::size_t>& operand_strides) {
    std::size_t offset = 0;
    const std::size_t operand_batch_rank = operand_shape.size() - 2;
    const std::size_t leading_dimensions = batch_shape.size() - operand_batch_rank;

    for (std::size_t dimension = batch_shape.size(); dimension-- > 0;) {
        const std::size_t coordinate = batch_index % batch_shape[dimension];
        batch_index /= batch_shape[dimension];

        if (dimension >= leading_dimensions) {
            const std::size_t operand_dimension = dimension - leading_dimensions;
            if (operand_shape[operand_dimension] != 1) {
                offset += coordinate * operand_strides[operand_dimension];
            }
        }
    }

    return offset;
}
```

The helper receives promoted operand shapes, so each operand always has two
matrix dimensions after its batch dimensions.

- [ ] **Step 5: Implement `Tensor::matmul`**

Add this implementation after the arithmetic operators in `src/tensor.cpp`:

```cpp
Tensor Tensor::matmul(const Tensor& tensor) const {
    if (rank() == 0 || tensor.rank() == 0) {
        throw std::invalid_argument("matmul requires tensors with rank at least 1");
    }

    const bool lhs_was_vector = rank() == 1;
    const bool rhs_was_vector = tensor.rank() == 1;

    std::vector<std::size_t> lhs_shape = shape_;
    std::vector<std::size_t> rhs_shape = tensor.shape_;
    if (lhs_was_vector) lhs_shape.insert(lhs_shape.begin(), 1);
    if (rhs_was_vector) rhs_shape.push_back(1);

    const std::size_t rows = lhs_shape[lhs_shape.size() - 2];
    const std::size_t inner = lhs_shape.back();
    const std::size_t rhs_inner = rhs_shape[rhs_shape.size() - 2];
    const std::size_t columns = rhs_shape.back();
    if (inner != rhs_inner) {
        throw std::invalid_argument("matmul inner dimensions do not match");
    }

    const std::vector<std::size_t> lhs_batch_shape(lhs_shape.begin(), lhs_shape.end() - 2);
    const std::vector<std::size_t> rhs_batch_shape(rhs_shape.begin(), rhs_shape.end() - 2);
    const std::vector<std::size_t> batch_shape =
        broadcast_shape(lhs_batch_shape, rhs_batch_shape);

    std::vector<std::size_t> result_shape = batch_shape;
    result_shape.push_back(rows);
    result_shape.push_back(columns);

    std::vector<float> result_data(element_count(result_shape), 0.0F);
    const std::vector<std::size_t> lhs_strides = contiguous_strides(lhs_shape);
    const std::vector<std::size_t> rhs_strides = contiguous_strides(rhs_shape);
    const std::size_t batch_count = element_count(batch_shape);

    for (std::size_t batch = 0; batch < batch_count; ++batch) {
        const std::size_t lhs_batch_offset =
            broadcast_batch_offset(batch, batch_shape, lhs_shape, lhs_strides);
        const std::size_t rhs_batch_offset =
            broadcast_batch_offset(batch, batch_shape, rhs_shape, rhs_strides);
        const std::size_t result_batch_offset = batch * rows * columns;

        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                float sum = 0.0F;
                for (std::size_t k = 0; k < inner; ++k) {
                    const std::size_t lhs_index =
                        lhs_batch_offset + row * lhs_strides[lhs_shape.size() - 2]
                        + k * lhs_strides.back();
                    const std::size_t rhs_index =
                        rhs_batch_offset + k * rhs_strides[rhs_shape.size() - 2]
                        + column * rhs_strides.back();
                    sum += data_[lhs_index] * tensor.data_[rhs_index];
                }
                result_data[result_batch_offset + row * columns + column] = sum;
            }
        }
    }

    if (lhs_was_vector) {
        result_shape.erase(result_shape.begin() + batch_shape.size());
    }
    if (rhs_was_vector) {
        result_shape.pop_back();
    }

    return Tensor(std::move(result_data), std::move(result_shape));
}
```

- [ ] **Step 6: Build and verify GREEN**

Run:

```powershell
cmake --build build
ctest --test-dir build --output-on-failure
```

Expected: build exits `0`; CTest reports `100% tests passed, 0 tests failed`.

- [ ] **Step 7: Verify warnings and diff cleanliness**

Run:

```powershell
g++ -std=c++17 -Wall -Wextra -Wpedantic -Iinclude src/tensor.cpp tests/tensor_test.cpp -o build/tensor_test_warnings.exe
build\tensor_test_warnings.exe
git diff --check
```

Expected: compilation and the test executable exit `0`, no compiler warnings
are printed, and `git diff --check` exits `0`.

- [ ] **Step 8: Commit the implementation**

```powershell
git add include/mytorch/tensor.h src/tensor.cpp tests/tensor_test.cpp
git commit -m "feat: add multidimensional matmul"
```
