# Yötä

[![Build Status](https://travis-ci.org/dfdx/Yota.jl.svg?branch=master)](https://travis-ci.org/dfdx/Yota.jl)

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

`GradientResult` can be used in conjunction with `update!()` function to modify tensors and fields of (mutable) structs. To continue out previous example:

```julia
for i=1:100
    val, g = grad(loss, m, X)
    println("Loss value in $(i)th epoch: $val")
    update!(m, g[1], (x, gx) -> x .- 0.01gx)
end
```

(Note that our simplified loss function doesn't actually represent an error to be minimized, so loss value quickly goes below zero. For more realistic and much more complex examples see [vae.jl](https://github.com/dfdx/Yota.jl/blob/master/examples/vae.jl).)

## Custom derivatives

You can add custom derivatives using `@diffrule` macro (see list of allowed variable names below).

```julia
logistic(x) = 1 / (1 + exp(-x))
# for an expression like `y = logistic(x)` where x is a Number
# gradient w.r.t. x
# is `(logistic(x) * (1 - logistic(x)) * dy)` where "dy" stands for derivative "dL/dy"
@diffrule logistic(x::Number) x (logistic(x) * (1 - logistic(x)) * dy)

L(x) = sum(logistic.(x))
val, g = grad(L, rand(5))
```

For functions accepting keyword arguments use `@diffrule_kw` instead:

```julia
import NNlib: conv, ∇conv_data, ∇conv_filter

@diffrule_kw conv(x, w) x ∇conv_data(dy, w)
@diffrule_kw conv(x, w) w ∇conv_filter(dy, x)
```

During reverse pass Yota will generate call to derivative function with the same keyword arguments that were
passed to the original one. For example, if you have:

```julia
conv(A, W; pad=1)
```

corresponding derivative will be:

```julia
∇conv_data(dy, w; pad=1)
```

There's also `@nodiff(call_pattern, variable)` macro which stops Yota from backpropagating through that variable.

### Allowed variable names

To distinguish between variable names that can be matched to (a.k.a. placeholders) and fixed symbols (e.g. function names), `@diffrule` uses several rules:

* `y` means return value of a primal expression, e.g. `y = f(x)`
* `dy` means derivative of a loss function w.r.t. to `y`
* `t`, `u`, `v`, `w`, `x`, as well as `i`, `j`, `k` (all listed in `Yota.DIFF_PHS`) are "placeholders" and can be used as names of variables, e.g. `@diffrule foo(u, v) u ∇foo(dy, u, v)`
* anything starting with `_` is also considered a placeholder, e.g. `@diffrule bar(u, _state) _state ∇bar(dy, u, _state)`

## Tracer and the Tape

Being a reverse-mode AD package, Yota works in 2 steps:

1. Record all primitive operations onto a "tape".
2. Go back trough the tape, recording derivatives for each operation.

"Tape" here is simply a list of operations. You can get the tape as a `.tape` field of `GradientResult` or construct it directly using `trace` function:

```julia
import Yota: trace

val, tape = trace(L, rand(5))
print(tape)

# Tape
#   inp %1::Array{Float64,1}
#   const %2 = logistic::typeof(logistic)
#   %3 = broadcast(%2, %1)::Array{Float64,1}
#   %4 = sum(%3)::Float64
```
`trace` uses [IRTools.jl](https://github.com/FluxML/IRTools.jl) to collect function calls during execution. Functions are divided into 2 groups:

 * primitive, which are recorded to the tape;
 * non-primitive, which are traced-through down to primitive ones.

By default, set of primitive functions is defined in `Yota.PRIMITIVES` and includes such beasts as `*`, `broadcast`, `getproperty` as well as all functions for which `@diffrule` is defined. You can also specify custom primitives using `primitive=Set([...])` keyword to `trace()`.

Also note that `broadcast`'s first argument is always considered a primitive and recorded to the tape as is, so backpropagation will only work if there's a `@diffrule` for it.

Tape can also be executed and compiled:

```julia
using BenchmarkTools
import Yota: play!, compile!

x = rand(5)

@btime play!(tape, x)
# 3.526 μs (13 allocations: 640 bytes)

compile!(tape)
@btime play!(tape, x)
# 492.063 ns (2 allocations: 144 bytes)
```

Note that `trace()` is an alias to `irtrace()` - IRTools-based tracer. As of Yota 0.4.0, two other tracers are available:

 * `ctrace()`, based on [Cassette.jl](https://github.com/jrevels/Cassette.jl)
 * `itrace()`, based on [JuliaInterpreter.jl](https://github.com/JuliaDebug/JuliaInterpreter.jl)

These tracers can be used for experimental purposes, but **their reliability or even existence is not guaranteed in future**. For any long-term code please use alias `trace()` which always points to the most recent and well-tested implementation.

## CUDA support

`CuArray` is fully supported. If you encounter an issue with CUDA arrays which you don't have with ordinary arrays, please file a bug.

## Static vs. dynamic (experimental)

Tracer records operations as they are executed the first time with given arguments. For example, for a loop like this:

```julia
function iterative(x, n)
    for i=1:n
        x = 2 .* x
    end
    return sum(x)
end
```
exactly `n` iterations will be recorded to the tape and replaying tape with any other values of `n` will make no effect. This also applies to a standard `grad()`:

```julia
x = rand(4)
_, g = grad(iterative, x, 1)   # g[1] == [2.0, 2.0, 2.0, 2.0]
_, g = grad(iterative, x, 2)   # g[1] == [2.0, 2.0, 2.0, 2.0]
_, g = grad(iterative, x, 3)   # g[1] == [2.0, 2.0, 2.0, 2.0]
```

Nevertheless, Yota provides pseudo-dynamic capabilities by caching gradient results for all ever generated tapes. This doesn't eliminate cost of re-tracing, but avoids repeated backpropagation and tape optimization. You can tell `grad()` to use dynamic caching using `dynamic=true` keyword argument:


```julia
x = rand(4)
_, g = grad(iterative, x, 1; dynamic=true)   # g[1] == [2.0, 2.0, 2.0, 2.0]
_, g = grad(iterative, x, 2; dynamic=true)   # g[1] == [4.0, 4.0, 4.0, 4.0]
_, g = grad(iterative, x, 3; dynamic=true)   # g[1] == [8.0, 8.0, 8.0, 8.0]
```

Note that this feature is experimental and may be removed in future versions.
