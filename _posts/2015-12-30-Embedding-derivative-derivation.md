---
layout: post
title: Embedding derivative derivation
---

Sometimes you can't use automatic differentiation to train a
model and then you have to do the derivation yourself. This derivation
is for
a simple two-layer model where the input layer is followed by an embedding 
layer which is followed by a fully connected softmax layer. It is based on
some [old and new matrix algebra (and calculus) useful for statistics](http://research.microsoft.com/en-us/um/people/minka/papers/matrix/).

## Softmax model

A single layer softmax model is defined like

\\[
p(\mathbf{y}|\mathbf{x}, \Theta) = 
    \frac{1}{Z} \prod_{c=1}^C \exp\\{\mathbf{x}^T \Theta_c \\}^{y_c} ,\\\
Z=\sum_{c=1}^C \exp\\{\mathbf{x}^T \Theta_c \\} ,
\\]

where 

- \\( C \\) is the number of classes,
- \\( \mathbf{y} \\) is the vector of one-hot encoded labels,
- \\( \mathbf{x} \\) is the vector of features,
- \\( \Theta \\) is the matrix of parameters (\\( \Theta_c \\) being the cth column), and
- \\( Z \\) is the normalising constant.

## Softmax with embeddings

Now we want \\( \mathbf{x} \\) to represent a sequence of words/characters.
One way to do that is to concatenate a bunch of one-hot encoded vectors

\\[
\mathbf{x}\_{MV\times1} = \[ \mathbf{x}\_1^T\  \mathbf{x}\_2^T \dots \mathbf{x}\_M^M \]^T 
\\]

where each \\(\mathbf{x}_i \\) is a \\( V \times 1 \\) vector that chooses an
element out of the vocabulary.

We want each 1 in the input vector to select a row from the embedding
matrix \\( E_{V\times H} \\), where \\( H \\) is the dimension of each embedding vector.
So,
\\(\mathbf{x}^T R \\), where \\( R \\) is our mystery matrix, should equal

\\[
\[\mathbf{x}\_1^T E \ \ \mathbf{x}\_2^T E \dots \mathbf{x}\_M^T E \].
\\]

This happens when
\\[
R = \begin{bmatrix}
    E & 0 & \ldots & 0 \\\
    0 & E & \ldots & 0 \\\
    \vdots & \vdots & \ddots
    & \vdots \\\
    0 & 0 & \ldots & E
\end{bmatrix}\_{MV\times MV} ,
\\]

which is just the Kronecker product \\( I \otimes E \\).

Now if we define 

\\[
\Theta_c = (I\otimes E)W_c,
\\]

and substitute into the softmax definition above we get

\\[
p(\mathbf{y}|\mathbf{x}, W, E) = 
    \frac{1}{Z} \prod_{c=1}^C \exp\\{\mathbf{x}^T (I\otimes E)W_c \\}^{y_c} ,
\\]

where 

- \\(V \\) is the number of embeddings (the vocabulary size),
- \\(M \\) is the number of embeddings per data point,
- \\(H \\) is the dimension of each embedding,
- \\( \mathbf{y}_{C\times 1} \\) is the vector of one-hot encoded labels,
- \\( \mathbf{x}_{VM\times 1} \\) is the vector of one-hot encoded features,
- \\(I_{M\times M} \\) is an identity matrix,
- \\(E_{V\times H} \\) is the latent embedding matrix, and
- \\({W_c}\_{MH\times 1}, \quad c \in 1 .. C \quad \\) are vectors of weights.


## Log likelihood
The log likelihood - our objective function - is 

\begin{align}
\mathcal{L} &= \sum_{n=1}^N \log p(\mathbf{y}\_n|\mathbf{x}\_n W E) \\\
            &= \sum_{n=1}^N \sum_{c=1}^C \{y_{nc} \mathbf{x}\_n^T (I\otimes E)W_c \} - \log Z_n ,
\end{align}

## Derivative with respect to the weights

The differential in terms of \\( W_c \\) is then

\begin{align}
d\mathcal{L} &= \sum_{n=1}^N  \{y_{nc} \mathbf{x}\_n^T (I\otimes E)dW_c \} 
    -  \frac{1}{Z_n} d Z_n \\\
&= \sum_{n=1}^N \{y_{nc} \mathbf{x}\_n^T (I\otimes E)dW_c \} 
    -  \frac{1}{Z_n}  d\exp\\{\mathbf{x}\_n^T (I\otimes E)W_c \\}  \\\
&= \sum_{n=1}^N \{y_{nc} \mathbf{x}\_n^T (I\otimes E)dW_c \} 
    -  \frac{1}{Z_n}  \exp\\{\mathbf{x}\_n^T (I\otimes E)W_c \\}
    \mathbf{x}\_n^T (I\otimes E)dW_c
\end{align}

which implies

\\[
\frac{d\mathcal{L}}{dW_c} = \sum_{n=1}^N \left( y_{nc} 
    -  \frac{1}{Z_n}  \exp\\{\mathbf{x}\_n^T (I\otimes E)W_c \\} \right)
    \mathbf{x}\_n^T (I\otimes E)
\\]

## Derivative with respect to the embeddings

Here we use the trace operator to rearrange elements so that the differential
is last. To rearrange the Kronecker product we also need the 
[commutation matrix](https://en.wikipedia.org/wiki/Commutation_matrixu)
\\( K_{p,q} \\) and the vector transposition operator (vec-transpose) 
\\( (\cdot)^{(q)} \\).

\begin{align}
d\mathcal{L} =& d\mathrm{tr} \( \sum_{n=1}^N \sum_{c=1}^C \{y_{nc} \mathbf{x}\_n^T (I\otimes E)W_c \} \)
    -  d\mathrm{tr} \( \frac{1}{Z_n} d Z_n \) \\\
=&  \sum_{n=1}^N \sum_{c=1}^C y_{nc}  d \mathrm{tr} \left( W_c \mathbf{x}\_n^T K_{M,V} (E\otimes I) K_{H, M} I \right)
    -   \frac{1}{Z_n} \mathrm{tr} 
    \left( d \sum_{c=1}^C \exp\\{\mathbf{x}\_n^T  (I\otimes E)  W_c  \\} \right) \\\
=&  \sum_{n=1}^N \sum_{c=1}^C y_{nc} d \mathrm{tr} \left( \left( W_c \mathbf{x}\_n^T \right)^{(M)}
    \left( K_{M,V} (E\otimes I) K_{H, M} I \right)^{(M)} \right) \\\
&-   \frac{1}{Z_n}  \exp\\{\mathbf{x}\_n^T  d(I\otimes E)  W_c  \\}
    d \mathrm{tr} \left( \mathbf{x}\_n^T K_{M,V} (E\otimes I) K_{H, M} W_c \right) \\\
=&  \sum_{n=1}^N \sum_{c=1}^C d \mathrm{tr} 
    \left( 
        K_{H, M}^{(M)T} (I\otimes I) 
        \left( 
            W_c \mathbf{x}\_n^T K_{M,V} 
        \right)^{(M)T} E 
    \right) 
\left(y_{nc}  - \frac{1}{Z_n}  \exp\\{\mathbf{x}\_n^T (I\otimes E) W_c  \\} \right) \\\
=&  \sum_{n=1}^N \sum_{c=1}^C 
    \left(y_{nc}  - \frac{1}{Z_n}  \exp\\{\mathbf{x}\_n^T (I\otimes E) W_c \\} \right) 
    \mathrm{tr} 
    \left( 
        K_{H, M}^{(M)T} (I\otimes I) 
        \left( 
            W_c \mathbf{x}\_n^T K_{M,V} 
        \right)^{(M)T} dE 
    \right) .
\end{align}

The derivative is then

\\[
\frac{d \mathcal{L}}{d E} = 
\sum_{n=1}^N \sum_{c=1}^C 
    \left(y_{nc} - \frac{1}{Z_n}  \exp\\{\mathbf{x}\_n^T (I\otimes E) W_c \\} \right) 
    K_{H, M}^{(M)T} 
    \left( 
        W_c \mathbf{x}\_n^T K_{M,V} 
    \right)^{(M)T} .
\\]

Well, that seems to be the answer! The \\( K_{H, M}^{(M)T} \left( W_c \mathbf{x}\_n^T K_{M,V} \right)^{(M)T} \\)
terms are basically just the outer product of the features and weights 
rearranged into the shape of \\( E^T \\).

Next I'll show a naive implementation in `numpy` and some of the character embeddings it learned.

