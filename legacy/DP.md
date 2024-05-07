# Differentiable Physics for Simulations

## Overview

The two most common applications of DP are:
- Error correction of numerical simulations: A differentiable version of the forward model, is levergaed to propagate the error of the forward model and train an in the loop corrector model.
- Paremeter Estimation: A differentiable forward model is used to solve inverse problems, such as parameter estimation or initial condition estimatio.

The construction of a differentiable version of the forward model, depends on weather its a implicit or explicit model. In the case of an implicit model, the adjoint method is used to compute the gradient of the loss function with respect to the input parameters. In the case of an explicit model, the gradient can be computed directly using automatic differentiation. 

Generally when we use AD, the gradient is derived from the numerical integration schema, since for AD the model needs to be implemented as a program. Another option would be symbolic differentiation. If we choose the adjoint method, there is the option to differentiate and then integrate the model, or to integrate and then differentiate the model.


## Optimization

First we summarize optimization methods and explain gradient based methods. Given a scalar loss function $\mathcal{L} \in \mathbb{R^n} \rightarrow \mathbb{R}$ and acollection of parameters $\mathbb{y} \in \mathbb{R^n}$ we wish optimize. If we take the derivate of the Loss with respect to the parameters $y$, we obtain the jacobian

$$ \frac{\partial \mathcal{L}}{\partial y} = J(x).$$ 

$J$ is a row vector and the gradient $\nabla \mathcal{L}$ is given by $J^T$. Taking the derivate with respect to $y$ again yields the Hessian

$$ \frac{\partial^2 \mathcal{L}}{\partial^2 y } = H(x).$$

Furthermore using the Taylor series, we can derive a relation between $J$ and $H$

$$ \mathcal{L}(x + \Delta) = \mathcal{L}(x) + \Delta J + \frac{1}{2} \Delta^2 H(x),$$

where $\Delta$ is a step in the parameter space.

### Newton's Method

We approximate the Loss function as a Parabola 

$$ \mathcal{L} = \frac{1}{2} H(y - y^*)^2 + c.$$

Analytically we can directly derive a formula to compute the minimum

$$ y^* = y - \frac{J^T}{H},$$

hence an update step is given by 

$$ \Delta = \frac{J^T}{H}.$$ 

It can be proven that Newtons Method always converges, even if the loss function $\mathcal{L}$ is not a parabola. However obtaining $H$ is very difficult to obtain in practive, making the method unfeasible in many application. There are a number of methods, which approximate $H$ such as Broyden's, BFGS and Gauss Newton. The Gauss Newton method essentially approximates the Hessian with the Squared Jacobian, thus the update step becomes

$$ \Delta = \frac{J^T}{J^TJ} = \frac{1}{J}.$$

The only disadvante is that the Jacobian needs to be inverted, which is costly and not always possible. 

### Adam 

Adam goes a step further and leverages a diagonal approximation of the Hessian

$$ H = \sqrt{\text{diag}(J^TJ)} \approx \sqrt{\text{diag}(J^2)} \approx \text{diag}(J).$$

The update step is then given by

$$ \Delta = \frac{J^T}{\text{diag}(J)} = \frac{1}{\sqrt{\text{diag}(J)}}.$$

Note that 

## Adjoint Method

Let's consider an ODE with $\mathbf{u}(t)$, the state of the system at time $t$, $\theta$ the parameters of the system and $u_0$ the initial state. The ODE is given by

$$ \frac{d\mathbf{u}}{dt} = f(\mathbf{u}, \theta).$$

and 

$$ \mathbf{u}(t_0) = \mathbf{u}_0.$$

Furthermore we are given an objective function $\mathcal{L}$, which is a scalar function of the state at the final time $t_f$ or potentially the parameters $\theta$. Thus the components are

- $\mathbf{u} \in \mathbb{R}^n$ the state of the system
- $\theta \in \mathbb{R}^p$ collection of parameters
- $f: \mathbb{R}^n \times \mathbb{R}^p \rightarrow \mathbb{R}^n$ the right hand side of the ODE, can be non-linear

The loss function is given by

$$ \mathcal{L} = \int_{0}^{T} g(\mathbf{u}, \theta) dt$$




### Gradient Descent

For gradient descent an update step is 

$$ \Delta = - \lambda J^T $$

which actually lives in the wrong space. Namely in $\frac{1}{y}$ instead of $y$ space. Neverthless the method is good enough for most application, however in physics based optimization problems, the inverted space can become a problem. 


## Numerical Simulation

Our goal is to make numerical simulations differentiable. This means that we want to be able to compute gradients of the simulation output with respect to the input parameters, which we can then use for gradient-based optimization.

We describe a physical model:

- The state of the physical system $\mathbf{x} \in \mathbb{R}^n$ is a vector of $n$ variables. We denote the state at time $t$ as $\mathbf{x}(t)$. Hence $x$ is a function of time.
- $\theta \in \mathbb{R}^p$ the parameters of the physical model
- $\mathcal{P}^*$ the continous physical model, for example a PDE. Usually first and second derivatives exist.
- $\mathcal{P \in \mathbb{R}^n \times \mathbb{R}^p \rightarrow \mathbb{R}^n}$ the discretized approximation of $\mathcal{P}^*$ It updates the state into a new state at evolved time $\mathbf{x}(t + \Delta t) = \mathcal{P}(\mathbf{x}(t), \theta)$, given the parameters $\theta$.
- The function $\mathcal{P}$ can be written as a sequence of operations $\mathcal{P} = \mathcal{P}_1 \circ \mathcal{P}_2 \circ \ldots \circ \mathcal{P}_m$.
- Finally we have an objective function $\mathcal{L} \in \mathbb{R}^n \rightarrow \mathbb{R}$ mapping from a state to a scalar loss value.

The derivate of the map $\mathcal{P(\mathbf{x})}$ with respect to the state $\mathbf{x}$ is the Jacobian 

$$ J = \frac{\partial \mathcal{P}}{\partial \mathbf{x}} \in \mathbb{R}^{n \times n}.$$

The derivative of the loss function with respect to the state is the gradient

$$ \nabla_{\mathbf{x}} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}} \frac{\partial \mathcal{P}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}}.$$

## The adjoint method

Historically, the adjoint method has been used to compute gradients of the loss function with respect to the initial state. The adjoint method is based on the idea of backpropagating the gradient of the loss function through the physical model. The adjoint method is based on the following steps:

1. We define the adjoint state $\mathbf{p}(t)$ which is a function of time and space.
2. We define the adjoint equation which is a PDE that describes the evolution of the adjoint state.
3. We define the adjoint initial condition $\mathbf{p}(t^*) = \nabla_{\mathbf{s}(t^*)} \mathcal{L}(\mathbf{s}(t^0), \mathbf{s}^*)$.
4. We solve the adjoint equation backwards in time from $t^*$ to $t^0$.




## Sources

- [Physics-based Deep Learning](https://physicsbaseddeeplearning.org)
