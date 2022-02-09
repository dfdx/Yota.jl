# Yota

## Basic usage

The most important function is [`grad()`](@ref), which has the form
`grad(f, args...) -> (output, gradients)`, e.g.:

```@example
using Yota

f(x) = 5x + 3

val, g = grad(f, 10)
```

Here `val` is the result of `f(10)` and `g` is a tuple of gradients w.r.t. to the inputs
including the function itself (which is [`ZeroTangent()`](https://juliadiff.org/ChainRulesCore.jl/dev/api.html#ChainRulesCore.ZeroTangent) in this case).


A bit more complex example from the ML domain:

```@example
using Yota

mutable struct Linear{T}
    W::AbstractArray{T,2}
    b::AbstractArray{T}
end

(m::Linear)(X) = m.W * X .+ m.b

# not very useful, but simple example of a loss function
loss(m::Linear, X) = sum(m(X))

m = Linear(rand(3,4), rand(3))
X = rand(4,5)

val, g = grad(loss, m, X)

@show g[2].W
@show g[2].b
```

The computed gradients can then be used in the `update!()` function to modify tensors and fields of (mutable) structs:

```julia
for i=1:100
    val, g = grad(loss, m, X)
    println("Loss value in $(i)th epoch: $val")
    update!(m, g[2], (x, gx) -> x .- 0.01gx)
end
```