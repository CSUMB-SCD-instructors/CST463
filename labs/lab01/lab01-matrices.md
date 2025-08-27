# Lab: Linear Algebra - Matrices


## Q1

Suppose a matrics $A$ has size $2 \times 3$, $B$ has size $1 \times 3$ and $C$ has size $3 \times 3$.
Please answer the below questions

### Q1a

Is the produce $AB$ possible?  If so, what is its size?

<details>
<summary>Answer</summary>

No, they are the wrong sizes since the number of columns in A doesn't match the number of rows in B.

</details>

### Q1b

Is the product $BC$ possible?  If so what is its size?

<details>
<summary>Answer</summary>

Yes, the resulting matrix would be $1 \times 3$

</details>

### Q1c

If $C$ has an inverse, what is its size?

<details>
<summary>Answer</summary>

It would be the same size as $C$, so $3 \times 3$

</details>

### Q1d 

What is the size of $A^{-1}$?

<details>
<summary>Answer</summary>

Since $A$ is not a square matrix, no inverse is possible.

</details>

### Q1e

What is the size of $(BC)^{T}$?

<details>
<summary>Answer</summary>
As we know from <a href="#q1b">Q1B</a>, the size of $BC$ is $1 \times 3$, so it's transpose would be $3 \times 1$.

</details>

### Q1f

Does it make sense to write $C^3$?  What does it mean?

<details>
<summary>Answer</summary>

Extrapolating from what we usually use the notation for, this probably means to multiple C by itself multiple times.
Since $C$ is a square matrix multiplication by itself is possible and result in a matrix of the same size -- and still square!
Therefore, this should be possible!

</details>

## Q2

What entries replace the question marks in the below matrices?  Note, there may be more than one entry per question mark.

### Q2a

```math
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
%
\begin {bmatrix}
?
\end{bmatrix}
=
\begin{bmatrix}
0.3a + 0.7b \\
0.3c + 0.7d
\end{bmatrix}

```

<details><summary>Answer</summary>

```math
\begin{bmatrix}
0.3 \\
0.7
\end{bmatrix}

```

</details>

### Q2b

```math
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
%
\begin {bmatrix}
?
\end{bmatrix}
=
\begin{bmatrix}
b & a \\
d & c
\end{bmatrix}

```

<details><summary>Answer</summary>

```math
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}

```

</details>

### Q2b

```math
\begin {bmatrix}
?
\end{bmatrix}
%
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
=
\begin{bmatrix}
b & a \\
d & c
\end{bmatrix}

```

<details><summary>Answer</summary>

```math
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}

```

Notice the solution is the same as the previous problem.  When it is used after another matrix it's a "column swapper", but when it's used before another matrix it's a "row swapper".

</details>

## Q3

Complete the Jupyter notebook [lab1-matrices.ipynb](lab1-matrices.ipynb).

## Q4

With matrix addition, do the two matrices involves always have to be the same size?

<details><summary>Answer</summary>
Yes.

</details>

## Q5

Prove that matrix addition is commutative.

<details><summary>Hint</summary>
Try _showing_ this for a specific size matrix, such as $A = \begin{bmatrix}a & b \\ c & d\end{matrix}$ and think about why it would be.
</details>

## Q6

Is it true that, for every matrix $A$ there exists another matrix $B$ such that $A+B = \begin{bmatrix}0\end{bmatrix}_{u \times v}$ , where \begin{bmatrix}0\end{bmatrix}_{u \times v}$ is the zero matrix the size of $A$ and $B$?


<details><summary>Answer</summary>
Yes!  
Since addition is element-wise, if we just set every value in our matrix $B$ to the negative of the value in $A$ we can get the zero matric.

</details>


## Extra Reading

- http://linear.ups.edu/html/section-VO.html