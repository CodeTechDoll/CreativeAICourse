# Unit 1: Introduction to Linear Algebra

## Core Concepts

- Scalars, vectors, and matrices
- Vector spaces and subspaces
- Linear independence and basis
- Linear combinations
- Spanning sets

## Goals

- Understand the basic concepts and terminology of linear algebra.
- Learn the difference between scalars, vectors, and matrices.
- Understand vector spaces, subspaces, linear independence, and basis.
- Learn about linear combinations and spanning sets.

## Scalars, Vectors, and Matrices

- **Scalars:** A scalar is a single numerical value, usually denoted by lowercase letters, such as $a$ or $b$.

- **Vectors:** A vector is an ordered list of numbers, typically represented as a column or row of numbers. Vectors can be denoted by bold lowercase letters, such as $\mathbf{v}$ or $\mathbf{u}$. For example, a 3-dimensional column vector can be represented as:

  $\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}$

- **Matrices:** A matrix is a 2-dimensional array of numbers, arranged in rows and columns. Matrices can be denoted by uppercase letters, such as $A$ or $B$. For example, a 2x3 matrix can be represented as:

  $A = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{bmatrix}$

## Vector Spaces and Subspaces

- **Vector Spaces:** A vector space is a set of vectors that is closed under vector addition and scalar multiplication. In other words, for any vectors $\mathbf{u}$ and $\mathbf{v}$ in the vector space, and any scalar $c$, the following conditions hold:

  1. $\mathbf{u} + \mathbf{v}$ is in the vector space.
  2. $c\mathbf{u}$ is in the vector space.

- **Subspaces:** A subspace is a subset of a vector space that is also a vector space. In other words, it satisfies the same conditions as a vector space but is a smaller subset.

## Linear Independence and Basis

- **Linear Independence:** A set of vectors is linearly independent if no vector in the set can be expressed as a linear combination of the others.

- **Basis:** A basis is a set of linearly independent vectors that span a vector space. In other words, any vector in the vector space can be expressed as a linear combination of the basis vectors.

## Linear Combinations and Spanning Sets

- **Linear Combination:** A linear combination of a set of vectors is formed by multiplying each vector by a scalar and then adding the results. For example, a linear combination of the vectors $\mathbf{v}$ and $\mathbf{u}$ can be represented as:

  $c_1\mathbf{v} + c_2\mathbf{u}$, where $c_1$ and $c_2$ are scalars.

- **Spanning Set:** A spanning set is a set of vectors whose linear combinations can generate all vectors in a given vector space or subspace.

# Examples of Linear Algebra Concepts

## Scalars, Vectors, and Matrices

- **Scalars:** Let $a = 2$ and $b = -3$. These are examples of scalar values.

- **Vectors:** Let $\mathbf{v} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ and $\mathbf{u} = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}$. These are examples of 3-dimensional column vectors.

- **Matrices:** Let $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$. These are examples of 2x2 matrices.

## Vector Spaces and Subspaces

- **Vector Spaces:** Consider the set of all 2-dimensional vectors. This set forms a vector space because it satisfies the conditions for vector addition and scalar multiplication:

  1. If $\mathbf{u} = \begin{bmatrix} u_1 \\ u_2 \end{bmatrix}$ and $\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \end{bmatrix}$, then $\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \end{bmatrix}$, which is still a 2-dimensional vector.
  2. If $c$ is a scalar, then $c\mathbf{u} = \begin{bmatrix} cu_1 \\ cu_2 \end{bmatrix}$, which is also a 2-dimensional vector.

- **Subspaces:** Consider the set of all 2-dimensional vectors where the second component is zero, such as $\begin{bmatrix} x \\ 0 \end{bmatrix}$. This set is a subspace of the 2-dimensional vector space because it satisfies the conditions for vector addition and scalar multiplication:

  1. If $\mathbf{u} = \begin{bmatrix} u_1 \\ 0 \end{bmatrix}$ and $\mathbf{v} = \begin{bmatrix} v_1 \\ 0 \end{bmatrix}$, then $\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ 0 \end{bmatrix}$, which still has a zero in the second component.
  2. If $c$ is a scalar, then $c\mathbf{u} = \begin{bmatrix} cu_1 \\ 0 \end{bmatrix}$, which also has a zero in the second component.

## Linear Independence and Basis

- **Linear Independence:** Consider the vectors $\mathbf{v} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $\mathbf{u} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$. These vectors are linearly independent because neither can be expressed as a linear combination of the other.

- **Basis:** The vectors $\mathbf{v}$ and $\mathbf{u}$ from the previous example also form a basis for the 2-dimensional vector space, because they are linearly independent and can be used to generate any 2-dimensional vector through linear combinations.

## Linear Combinations and Spanning Sets

- **Linear Combination:** Let $\mathbf{v} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ and $\mathbf{u} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$. We can create a linear combination of these vectors with scalars $c_1 = 2$ and $c_2 = -1$:

  $c_1\mathbf{v} + c_2\mathbf{u} = 2\begin{bmatrix} 1 \\ 2 \end{bmatrix} - 1\begin{bmatrix} 3 \\ 1 \end{bmatrix} = \begin{bmatrix} -1 \\ 3 \end{bmatrix}$

- **Spanning Set:** Let $\mathbf{v} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $\mathbf{u} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$. These vectors form a spanning set for the 2-dimensional vector space, because any 2-dimensional vector $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$ can be expressed as a linear combination of $\mathbf{v}$ and $\mathbf{u}$:

  $\mathbf{x} = x_1\mathbf{v} + x_2\mathbf{u} = x_1\begin{bmatrix} 1 \\ 0 \end{bmatrix} + x_2\begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} x_1 \ x_2 \end{bmatrix}$
