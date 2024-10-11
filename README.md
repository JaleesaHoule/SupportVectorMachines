# SupportVectorMachine
Implementation of SVM for pattern recognition course assignment.


## Linear SVM

SVM is a method of classification which seeks to minimize structural risk for given data. The idea is that a decision boundary or hyperplane will generalize better when there is a wider margin between the boundary and the nearest data points. The data points which are closest to the boundary are called Support Vectors (SVs), as they are the most influential when determining the decision boundary. Training an SVM involves solving a quadratic problem with linear constraints. 

There are three general cases in which SVMs can be used. The most simple case is when the data is linearly separable. In this instance, we can consider the general form of the linear discriminant,

```math
\begin{equation}
    g(x)=w^Tx + w_0
\end{equation}
```

where we decide class 1 if $g(x)>0$ and class 2 if $g(x)<0$. If we consider the dual classification problem 

```math
\begin{equation}
    z_k (w^Tx_k + w_0) > 0 
\end{equation}
```
for $k=1,2,...,n$ data points, then we can see that the distance $r$ between $x_k$ and the decision boundary can be constrained such that

```math
\begin{equation}
    r = \frac{z_k g(x_k)}{||w||} > b
\end{equation}
```
for $b>0$ and $b||w||>1$. This then indicates that $b = 1/ ||w||$, where $2b$ is the total width of the margin separating the closest $x_k$ data points on either side of the decision boundary. The goal of SVM is then to maximize $2/||w||$, which can also be accomplished by minimizing the quadratic function

```math
\begin{equation}
    \frac{1}{2}||w||^2
\end{equation}
```
which is subject to $z_k g(x_k) > 1$ for $k=1,2,...n$. In practice, this is done using Langrange optimization such that

```math
\begin{equation}
    L(w,w_0,\lambda) =\frac{1}{2}||w||^2 - \sum^n_{k+1} \lambda_k [(w^Tx_k + w_0)-1]
\end{equation}
```
where $\lambda_k \geq 0$ are referred to as Langrange multipliers. Using these Langrange multipliers, we are able to reframe the problem so that we are maximizing 

```math
\begin{equation}
    \sum^n_{k+1} \lambda_k - \frac{1}{2}\sum_{k,j}^n \lambda_k \lambda_j z_k z_j x^T_j x_k.
\end{equation}
```
When maximizing this expression, we assume that $\sum^n_{k=1} z_k \lambda_k = 0$ and $\lambda_k >0$. Solving this maximization problem leads to a linear discriminant in the new form

```math
\begin{equation}
\begin{aligned}
        g(x)&= \sum^n_{k+1}z_k \lambda_k (x^T_kx) + w_0 \\
        & = \sum^n_{k+1}z_k \lambda_k (x \cdot x_k) + w_0. \\
\end{aligned}
\end{equation}
```
This linear discriminant now only depends on the support vectors, as the value of $\lambda_k$ is zero for any $x_k$ which is not a support vector. \\\\

## Non-Linear SVM and Kernels

The theory for SVM can be expanded for data that is not linearly separable by mapping the data points $x_k$ using some function $\Phi (x_k)$. The non-linear SVM then is mapped to dimension $h$ and can be expressed as 

```math
\begin{equation}
    g(x) = \sum^n_{k+1}z_k \lambda_x (\Phi(x) \cdot \Phi(x_k)) + w_0.
\end{equation}
```

In practice, mapping to $h$ dimensions is computationally expensive, and finding the function $\Phi(x)$ can be challenging. As a solution to this problem, kernels can be employed to simplify the computations. A kernel is a positive-definite, symmetric matrix defined as 

```math
\begin{equation}
    K(x,x_k) = \Phi(x) \cdot \Phi(x_k).
\end{equation}
```
Replacing this into the above equation, the discriminant becomes

```math
\begin{equation}
    g(x) = \sum^n_{k+1}z_k \lambda_x K(x,x_k) + w_0.
\end{equation}
```

This manipulation allows us to compute these dot products without explicitly mapping to the higher feature space. The challenge then becomes finding the kernel and its corresponding parameters which yield the best results for a given problem. 

For this experiment, we will be exploring different parameters for the polynomial and the radial basis function (RBF) kernel. The polynomial kernel is defined as 

```math
\begin{equation}
    K(x,x_k) = (\gamma x^T x_k + c_0)^d.
\end{equation}
```
We are asked to explore the results of a three-fold cross validation experiment by varying the degree so that $d=[1,2,3]$. To simplify parameter options, we will set $\gamma=1$ and $c_0=0$. 

The RBF kernel is defined as 

```math
\begin{equation}
    K(x,x_k) = e^{-\gamma |x-x_k|^2}.
\end{equation}
```
For this kernel, we are asked to explore the results for $\gamma = [0.1, 1, 10, 100]$. In addition to varying $\gamma$ and $d$ for these two kernels, we are told to explore the result of choosing different cost values $C$, which are associated with an added cost function. For data which may contain outliers (i.e., the data is "almost" linearly separable), a cost function can be added to Equation \ref{linear_svm} so that we can account for these misclassifications by introducing positive error variables $\Psi_k$ so that 

```math
\begin{equation}
    z_k (w^Tx_k+w_0) \geq 1- \Psi_k
\end{equation}
```
for $k=1,2,...n$. These variables are referred to as slack variables, and they provide a framework for a modified linear discriminant function by seeking to minimize 

```math
\begin{equation}
    \frac{1}{2}||w||^2 + C\sum^n_{k+1} \Psi_k 
\end{equation}
```
where $C$ is a constant associated with the cost of allowing misclassifications. By allowing these slack variables, the Langrange multipliers, $\lambda_k$, from the above equation become constrained such that $0<\lambda_k<C$. By observation, we can see that smaller values of $C$ will result in a smaller overall value of $C\sum^n_{k+1} \Psi_k$, so the inclusion of misclassifications is treated less severely as we try to minimize the cost function. Conversely, large values of $C$ will make that expression much larger, associating a higher cost with misclassifications, which will lead to a decision boundary with a smaller margin of separation as we seek to minimize those misclassifications. One challenge of using SVM is determining the best cost so that these misclassifications are minimized while the margin of separation is maximized. For this experiment, we will be exploring the effect of setting $C=[0.1,1,10,100]$ for both the polynomial and RBF kernels.\\\\
