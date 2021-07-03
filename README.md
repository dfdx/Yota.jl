# Yötä

[![Build Status](https://travis-ci.org/dfdx/Yota.jl.svg?branch=master)](https://travis-ci.org/dfdx/Yota.jl)

Reverse-mode automatic differentiation for static and dynamic graphs.

## Migration to Yota v0.5

If you have previously used Yota < v0.5, pay attention to the following changes:

* `grad()` now returns `(value, (fn-grad, arg-grads...))`, where `fn-grad` is the gradient w.r.t. the function object fields (if any). Previous versions of Yota only returned gradients w.r.t. function arguments, which are now shifted by one. That is, must use `g[i + 1]` to refer to the gradient w.r.t. to the `ith` argument.
* Struct gradients are now represented by [`ChainRulesCore.Tangent`](https://juliadiff.org/ChainRulesCore.jl/stable/api.html#ChainRulesCore.Tangent) type.
* Function tracing has been reworked and moved to [Ghost.jl](https://github.com/dfdx/Ghost.jl).


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

`g` is a tuple of gradients of the loss function w.r.t. to the function object itself and its 2 arguments.

These gradients can then be used in the `update!()` function to modify tensors and fields of (mutable) structs:

```julia
for i=1:100
    val, g = grad(loss, m, X)
    println("Loss value in $(i)th epoch: $val")
    update!(m, g[2], (x, gx) -> x .- 0.01gx)
end
```

Note that Yota caches gradients and may not see changes to functions
if you redefine them (e.g. in REPL). To reset the cache, invoke:

```julia
Yota.reset!()
```


## ChainRules

The primary method for extending the set of supported derivatives is by adding methods to `rrule()` function from [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl). Note that Yota reads the list of available `rrule`s during initialization, if you define new `rrules` _after_
Yota is loaded, you need to explicitely call `Yota.update_chainrules_primitives!()`.

Some functions are handled by Yota's own rules ("d-rules") instead, but at the moment this mechanism is purely internal and should not be used outside of the package.


## How it works

Yota is built on top of the code tracer in [**Ghost.jl**](https://github.com/dfdx/Ghost.jl). Essentially, differentiation boils down to the following steps:

1. Trace function execution using [`Ghost.trace()`](https://dfdx.github.io/Ghost.jl/dev/reference/#Ghost.trace) producing a computational graph as a [`Tape`](https://dfdx.github.io/Ghost.jl/dev/reference/#Ghost.Tape).
2. Run `Yota.gradtape!()` to add derivative operations to that tape.
3. Compile the tape back to a Julia function.

One function useful for debugging is `Yota.gradtape(f, args...)` (without exclamation sign) which skips the compilation and instead returns the computed tape.

