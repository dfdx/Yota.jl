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

You can add custom derivatives using `@diffrule` macro.

```julia
logistic(x) = 1 / (1 + exp(-x))
# for an expression like `logistic(x)` where x is a Number
# gradient w.r.t. x
# is `(logistic(x) * (1 - logistic(x)) * ds)` where "ds" stands for derivative "dL/dy"
@diffrule logistic(x::Number) x (logistic(x) * (1 - logistic(x)) * ds)

L(x) = sum(logistic.(x))
val, g = grad(L, rand(5))
```

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
`trace` uses [Cassette.jl](https://github.com/jrevels/Cassette.jl/) to collect function calls during execution. Functions are divided into 2 groups:

 * primitive, which are recorded to the tape;
 * non-primitive, which are traced-through down to primitive ones.

By default, set of primitive functions is defined in `Yota.PRIMITIVES` and includes such beasts as `*`, `broadcast`, `getproperty` as well as all functions for which `@diffrule` is defined. You can also specify custom primitives using `primitive=Set([...])` keyword to `trace()`.


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


## Loops, conditions, etc.

Tracer records operations as they are executed the first time with given arguments. For example, for a loop like this:

```julia
function iterative(x, n)
    for i=1:n
        x = 2x
    end
    return x
end
```
exactly `n` iterations will be recorded to the tape and all future values of `n` will make no effect.

## CuArrays support (experimental)

Yota should work with CuArrays out of the box, although integration is not well tested yet.

In addition, you can use function `to_cuda()` to transform arrays and structs into CUDA-compatible, see [cu_vae.jl](https://github.com/dfdx/Yota.jl/blob/master/examples/cu_vae.jl) for an example. Note that this API is highly experimental and will most likely change to something more device-agnostic.