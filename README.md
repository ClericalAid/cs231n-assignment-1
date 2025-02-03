Details about this assignment can be found [on the course webpage](http://cs231n.github.io/), under Assignment #1 of Spring 2024.

# Course Resources
In order to complete this assignment, there are relevant readings provided by the course. They are listed below, alongside the relevant content they provide:

- [loss functions of SVM and softmax classifier described](https://web.archive.org/web/20250203035247/https://cs231n.github.io/linear-classify/)
- [gradient of SVM](https://web.archive.org/web/20250125103925/https://cs231n.github.io/optimization-1/)
- [gradient of softmax](https://web.archive.org/web/20250103075854/https://cs231n.github.io/neural-networks-case-study/)

# The math behind the gradient
## Context
Since we are dealing with vectors and matrices, some derivatives may feel a bit unintuitive. Provided below is an example of how to work through a derivative by hand. The trick is to consider how each component of the vector/matrix affects the output, and individually calculate it for each value within the vector/matrix.

For example, let's look at the loss of the softmax classifier.

```math
L_i = -\log(\frac{e^{w_k x_i}}{\sum_j e^{w_j x_i}})
```

In this case, $w_k$ refers to the $k^{th}$ row of the matrix $w$. When multiplied by the vector $x$, it produces the score which matches our prediction. We will refer to this as $f_k$ ($f$ is the result of the matrix multiplication between $w$ and $x$).

In the notes, they provide the gradient of the loss as follows:
```math
p_k = \frac{e^{f_k}}{\sum_j e^{f_j}}
```
```math
\frac{\partial L_i}{\partial f_k} = p_k - 1(y_i = k)
```

The result of $wx$ is a vector $f$, which can be written out as:
```math
\begin{bmatrix} f_1 \\ f_2 \\ ... \\ f_n \end{bmatrix}
```

The gradient of this vector would be:
```math
\frac{\partial L}{\partial f} = \begin{bmatrix} \frac{\partial L}{\partial f_1} \\ \frac{\partial L}{\partial f_2} \\ ... \\ \frac{\partial L}{\partial f_n} \end{bmatrix}
```

The gradient provided by the note is only valid for where $y_i = k$. However, for this assignment we need a gradient for the entire vector.

## Calculating gradient analytically
As an example, we will solve the gradient for the first position of the vector.

```math
\frac{\partial L_i}{\partial f_1} = \frac{\partial}{\partial f_1}(-\log(\frac{e^{f_k}}{\sum_j e^{f_j}}))
```
The numerator $e^{f_k}$ is seen as a constant here. This is because we are taking the derivative with respect to $f_1$ and not $f_k$. We will write $e^{f_k}$ as a constant $A$.

Also, the denominator $\sum_j e^{f_j}$ can be rewritten as: $B + e^{f_1}$. This is because we only care about $f_1$ here, and every other variable is treated as a constant. So it applies to the rest of the numbers in the summation. We have a simpler representation for our equation now
```math
\frac{\partial L_i}{\partial f_1} = \frac{\partial}{\partial f_1}(-\log(\frac{A}{B + e^{f_1}}))
```
```math
= -1(\frac{B+e^{f_1}}{A})\frac{\partial}{\partial f_1}(\frac{A}{B + e^{f_1}})
```
```math
= -1(\frac{B+e^{f_1}}{A})(\frac{-A}{(B + e^{f_1})^2})\frac{\partial}{\partial f_1}(B + e^{f_1})
```
```math
= (\frac{1}{B + e^{f_1}})\frac{\partial}{\partial f_1}(B + e^{f_1})
```
```math
= \frac{e^{f_1}}{B + e^{f_1}}
```
```math
= \frac{e^{f_1}}{\sum_j e^{f_j}}
```
```math
= p_1
```