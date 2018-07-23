
# Yötä

Reverse-mode automatic differentiation for static and dynamic graphs.

## Usage

```
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

```
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
