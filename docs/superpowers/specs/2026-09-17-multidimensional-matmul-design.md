# Multidimensional Tensor Matmul Design

## Goal

Extend `Tensor::matmul(const Tensor&)` to match PyTorch-style `matmul`
shape semantics for one-dimensional vectors, two-dimensional matrices, and
higher-dimensional batched matrices. Tensor storage remains contiguous and
row-major.

## Public API

Keep the existing declaration:

```cpp
Tensor matmul(const Tensor& tensor) const;
```

The operation returns a new Tensor and never mutates either operand.

## Shape Semantics

Scalar tensors (rank zero) are invalid operands.

One-dimensional operands are virtually promoted for the calculation without
allocating temporary Tensor data:

- A left shape `[K]` is treated as `[1, K]`.
- A right shape `[K]` is treated as `[K, 1]`.

The promoted shapes follow the general rule:

```text
lhs[..., M, K] @ rhs[..., K, N] -> result[..., M, N]
```

The leading batch dimensions are broadcast using the existing trailing-axis
rule: corresponding dimensions must be equal or one of them must be `1`.
After multiplication, dimensions introduced by one-dimensional promotion are
removed:

- `[K] @ [K] -> {}`
- `[K] @ [K, N] -> [N]`
- `[M, K] @ [K] -> [M]`
- `[M, K] @ [K, N] -> [M, N]`

Zero-length dimensions remain valid. In particular, an inner dimension of
zero produces zero-valued sums when the result itself is non-empty.

## Computation

Reuse the existing `element_count`, `broadcast_shape`, and
`contiguous_strides` helpers.

1. Build virtual promoted shapes for one-dimensional operands.
2. Validate that the left inner dimension equals the right inner dimension.
3. Broadcast the leading batch shapes.
4. Form the unsqueezed result shape as `batch_shape + [M, N]`.
5. Iterate each broadcast batch and each output `(row, column)` pair.
6. Map the batch coordinate into each operand, using coordinate zero for an
   operand batch dimension whose length is `1`.
7. Accumulate the dot product over `K` directly from the operands' flat data.
8. Remove the temporary vector dimensions from the final result shape.

Inputs are not expanded or copied. Runtime is
`O(batch_count * M * N * K)` and extra memory is limited to result storage and
small shape/stride vectors.

## Errors

Throw `std::invalid_argument` when:

- either operand is rank zero;
- the matrix inner dimensions differ;
- the leading batch dimensions cannot broadcast.

## Tests

Add direct result-data and result-shape assertions for:

- vector dot vector;
- vector times matrix;
- matrix times vector;
- matrix times matrix;
- batched matrix multiplication;
- broadcasting an unbatched matrix across batches;
- broadcasting singleton batch dimensions on both operands;
- a zero-length inner dimension;
- scalar rejection;
- inner-dimension mismatch;
- incompatible batch dimensions.

All existing elementwise broadcasting and formatting tests must continue to
pass.

## Non-goals

- BLAS integration or SIMD optimization;
- persistent strides or non-contiguous views;
- dtype promotion;
- autograd support;
- an explicit expanded Tensor representation.
