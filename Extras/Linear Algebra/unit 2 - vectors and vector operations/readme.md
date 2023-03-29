# Unit 2: Vectors and Vector Operations

## Core Concepts

- Vector addition and subtraction
- Scalar multiplication
- Dot product
- Cross product
- Vector magnitude and normalization
- Projection and orthogonal components

## Goals

- Learn and understand the basic vector operations.
- Understand the concept of dot product and its properties.
- Understand the concept of cross product and its properties.
- Learn how to calculate vector magnitude and normalization.
- Understand the concepts of projection and orthogonal components.

## Vector Addition and Subtraction

- **Vector Addition:** To add two vectors, simply add their corresponding components. Given vectors $\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}$ and $\mathbf{u} = \begin{bmatrix} u_1 \\ u_2 \\ u_3 \end{bmatrix}$, their sum is:

  $\mathbf{v} + \mathbf{u} = \begin{bmatrix} v_1 + u_1 \\ v_2 + u_2 \\ v_3 + u_3 \end{bmatrix}$

- **Vector Subtraction:** To subtract two vectors, simply subtract their corresponding components. Given vectors $\mathbf{v}$ and $\mathbf{u}$ as before, their difference is:

  $\mathbf{v} - \mathbf{u} = \begin{bmatrix} v_1 - u_1 \\ v_2 - u_2 \\ v_3 - u_3 \end{bmatrix}$

## Scalar Multiplication

- **Scalar Multiplication:** To multiply a vector by a scalar, simply multiply each component of the vector by the scalar. Given a vector $\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}$ and a scalar $c$, their product is:

  $c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ cv_3 \end{bmatrix}$

## Dot Product

- **Dot Product:** The dot product (also known as the scalar or inner product) is a binary operation that takes two vectors and returns a scalar. Given vectors $\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}$ and $\mathbf{u} = \begin{bmatrix} u_1 \\ u_2 \\ u_3 \end{bmatrix}$, their dot product is: $\mathbf{v} \cdot \mathbf{u} = v_1u_1 + v_2u_2 + v_3u_3$

The dot product has several important properties:

- **Commutative:** $\mathbf{v} \cdot \mathbf{u} = \mathbf{u} \cdot \mathbf{v}$
- **Distributive:** $\mathbf{v} \cdot (\mathbf{u} + \mathbf{w}) = \mathbf{v} \cdot \mathbf{u} + \mathbf{v} \cdot \mathbf{w}$
- **Associative with scalar multiplication:** $(c\mathbf{v}) \cdot \mathbf{u} = c(\mathbf{v} \cdot \mathbf{u})$

The dot product can also be used to find the angle between two vectors:

  $\cos{\theta} = \frac{\mathbf{v} \cdot \mathbf{u}}{|\mathbf{v}||\mathbf{u}|}$, where $\theta$ is the angle between $\mathbf{v}$ and $\mathbf{u}$, and $|\mathbf{v}|$ and $|\mathbf{u}|$ are the magnitudes of $\mathbf{v}$ and $\mathbf{u}$, respectively.

## Cross Product

- **Cross Product:** The cross product (also known as the vector product) is a binary operation that takes two vectors and returns a vector. It is only defined for 3-dimensional vectors. Given vectors $\mathbf{v} = \begin{bmatrix} v_1 \ v_2 \ v_3 \end{bmatrix}$ and $\mathbf{u} = \begin{bmatrix} u_1 \\ u_2 \\ u_3 \end{bmatrix}$, their cross product is:

  $\mathbf{v} \times \mathbf{u} = \begin{bmatrix} v_2u_3 - v_3u_2 \\ v_3u_1 - v_1u_3 \\ v_1u_2 - v_2u_1 \end{bmatrix}$

The cross product has several important properties:

1. Anti-commutative: $\mathbf{v} \times \mathbf{u} = -(\mathbf{u} \times \mathbf{v})$
2. Distributive: $\mathbf{v} \times (\mathbf{u} + \mathbf{w}) = \mathbf{v} \times \mathbf{u} + \mathbf{v} \times \mathbf{w}$
3. Associative with scalar multiplication: $(c\mathbf{v}) \times \mathbf{u} = c(\mathbf{v} \times \mathbf{u})$

The cross product is orthogonal to both input vectors:

- $\mathbf{v} \cdot (\mathbf{v} \times \mathbf{u}) = 0$
- $\mathbf{u} \cdot (\mathbf{v} \times \mathbf{u}) = 0$

## Vector Magnitude and Normalization

- **Vector Magnitude:** The magnitude (or length) of a vector $\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}$ is denoted by $\|\mathbf{v}\|$ and is calculated using the Pythagorean theorem:

  $\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + v_3^2}$

- **Vector Normalization:** To normalize a vector, divide each component by its magnitude. The normalized vector (also known as a unit vector) has a magnitude of 1:

  $\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|} = \begin{bmatrix} \frac{v_1}{\|\mathbf{v}\|} \\ \frac{v_2}{\|\mathbf{v}\|} \\ \frac{v_3}{\|\mathbf{v}\|} \end{bmatrix}$

## Projection and Orthogonal Components

- **Projection:** The projection of a vector $\mathbf{v}$ onto another vector $\mathbf{u}$, denoted by $\text{proj}_{\mathbf{u}}(\mathbf{v})$, is the vector in the direction of $\mathbf{u}$ that represents the "shadow" of $\mathbf{v}$:

  $\text{proj}_{\mathbf{u}}(\mathbf{v}) = \frac{\mathbf{v} \cdot \mathbf{u}}{\|\mathbf{u}\|^2}\mathbf{u}$

- **Orthogonal Component:** The orthogonal component of a vector $\mathbf{v}$ with respect to another vector $\mathbf{u}$, denoted by $\text{orth}_{\mathbf{u}}(\mathbf{v})$, is the vector that is orthogonal to $\mathbf{u}$ and whose sum with $\text{proj}_{\mathbf{u}}
(\mathbf{v})$ results in $\mathbf{v}$:

  $\text{orth}_{\mathbf{u}}(\mathbf{v}) = \mathbf{v} - \text{proj}_{\mathbf{u}}(\mathbf{v})$

# Examples of Vector Operations

## Vector Addition and Subtraction

- **Vector Addition:** Let $\mathbf{v} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ and $\mathbf{u} = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}$. To add these vectors, simply add their corresponding components:

  $\mathbf{v} + \mathbf{u} = \begin{bmatrix} 1 + 4 \\ 2 + 5 \\ 3 + 6 \end{bmatrix} = \begin{bmatrix} 5 \\ 7 \\ 9 \end{bmatrix}$

- **Vector Subtraction:** Using the same vectors $\mathbf{v}$ and $\mathbf{u}$, to subtract them, subtract their corresponding components:

  $\mathbf{v} - \mathbf{u} = \begin{bmatrix} 1 - 4 \\ 2 - 5 \\ 3 - 6 \end{bmatrix} = \begin{bmatrix} -3 \\ -3 \\ -3 \end{bmatrix}$

## Scalar Multiplication

- **Scalar Multiplication:** Let $\mathbf{v} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ and a scalar $c = 2$. To multiply the vector by the scalar, multiply each component of the vector by the scalar:

  $c\mathbf{v} = 2\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} = \begin{bmatrix} 2 \\ 4 \\ 6 \end{bmatrix}$

## Dot Product

- **Dot Product:** Let $\mathbf{v} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ and $\mathbf{u} = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}$. To find their dot product, multiply their corresponding components and sum the results:

  $\mathbf{v} \cdot \mathbf{u} = (1)(4) + (2)(5) + (3)(6) = 4 + 10 + 18 = 32$

## Cross Product

- **Cross Product:** Let $\mathbf{v} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ and $\mathbf{u} = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}$. To find their cross product, use the formula:

  $\mathbf{v} \times \mathbf{u} = \begin{bmatrix} (2)(6) - (3)(5) \\ (3)(4) - (1)(6) \\ (1)(5) - (2)(4) \end{bmatrix} = \begin{bmatrix} -3 \\ 6 \\ -3 \end{bmatrix}$

## Vector Magnitude and Normalization

- **Vector Magnitude:** Let $\mathbf{v} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$. To find its magnitude, use the formula:

  $\|\mathbf{v}\| = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{1 + 4 + 9} = \sqrt{14}$

- **Vector Normalization:** To normalize the vector $\mathbf{v}$, divide each component by its magnitude:

  $\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|} = \frac{1}{\sqrt{14}}\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} = \begin{bmatrix} \frac{1}{\sqrt{14}} \\ \frac{2}{\sqrt{14}} \\ \frac{3}{\sqrt{14}} \end{bmatrix}$

## Projection and Orthogonal Components

- **Projection:** Let $\mathbf{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$ and $\mathbf{u} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$. To find the projection of $\mathbf{v}$ onto $\mathbf{u}$, use the formula:

  $\text{proj}_{\mathbf{u}}(\mathbf{v}) = \frac{\mathbf{v} \cdot \mathbf{u}}{\|\mathbf{u}\|^2}\mathbf{u} = \frac{(3)(1) + (4)(2)}{1^2 + 2^2}\begin{bmatrix} 1 \\ 2 \end{bmatrix} = \frac{11}{5}\begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} \frac{11}{5} \\ \frac{22}{5} \end{bmatrix}$

- **Orthogonal Component:** To find the orthogonal component of $\mathbf{v}$ with respect to $\mathbf{u}$, subtract the projection from $\mathbf{v}$:

  $\text{orth}_{\mathbf{u}}(\mathbf{v}) = \mathbf{v} - \text{proj}_{\mathbf{u}}(\mathbf{v}) = \begin{bmatrix} 3 \\ 4 \end{bmatrix} - \begin{bmatrix} \frac{11}{5} \\ \frac{22}{5} \end{bmatrix} = \begin{bmatrix} \frac{4}{5} \\ \frac{-2}{5} \end{bmatrix}$
