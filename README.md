
# Yötä

Reverse-mode automatic differentiation for static and dynamic graphs.

## Usage

```julia
mutable struct Linear{T}
    W::AbstractArray{T,2}
    b::AbstractArray{T}
end

forward(m::Linear, X) = m.W * X

loss(m::Linear, X) = sum(forward(m, X))

m = Linear(rand(3,4), rand(3))
X = rand(4,5)

val, g = grad(loss, m, X)
```

`g` is an object of type `GradientResult` holding gradients w.r.t. input variables. For scalars
and tensors it returns gradient value, for structs it returns dictionary of
(field path → gradient) pairs:

```julia
julia> g[1]
Dict{Tuple{Symbol},Array{Float64,2}} with 1 entry:
  (:W,) => [3.38128 2.97142 2.39706 1.55525; 3.38128 2.97142 2.39706 1.55525; 3.38128 2.97142 2.39706 1.55525]   # gradient w.r.t. m.W

julia> g[2]  # gradient w.r.t. X
4×5 Array{Float64,2}:
 0.910691  0.910691  0.910691  0.910691  0.910691
 1.64994   1.64994   1.64994   1.64994   1.64994 
 1.81215   1.81215   1.81215   1.81215   1.81215 
 2.31594   2.31594   2.31594   2.31594   2.31594
```

## Primitives and their gradients

Internally, Yota uses operator overloading and a "tape" to record function calls during first function execution (forward pass) and propagate derivatives back to function arguments (backward pass). Some functions such as `*`, `sum()` or `reshape()` are recorded to the tape as is. We call them primitives and define gradients for each of them. Others, say, `my_own_function()` are recursively broken down into simpler ones until only primitives are left.

We can mark a function as primitive using `@primitive` macro. Consider the following example:

```julia
relu(x::AbstractArray) = max.(x, 0)
@primitive relu(x::AbstractArray)
```
Now `relu()` will be written to the tape as is instead of being decomposed into simpler elements. To make differentiation possible, we also must define gradient function for it:


```julia
relu_grad(x::AbstractArray) = float(x .> 0)
@primitive relu_grad(x::AbstractArray)
```

and map one to the other:

```julia
@grad relu(x::AbstractArray) 1 relu_grad(x)
```

Now we can calculate gradient of a function involving `relu`:

```julia
relu_test(x) = sum(relu(x))

grad(relu_test, rand(5))
```