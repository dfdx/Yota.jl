# Yota.jl

Umlaut.jl is a code tracer for the Julia programming language. It lets you trace the function execution, recording all primitive operations onto a linearized tape. Here's a quick example:


```@example
using Umlaut     # hide
inc(x) = x + 1
mul(x, y) = x * y
inc_double(x) = mul(inc(x), inc(x))

val, tape = trace(inc_double, 2.0)
```
The tape can then be analyzed, modified and even compiled back to a normal function. See the following sections for details.

!!! note

    Umlaut.jl was started as a fork of Ghost.jl trying to overcome some of its
    limitations, but eventually the codebase has diverged so much that the new package was born. Although the two have pretty similar API, there are several notable differences.
    See [Migration from Ghost](@ref) for details.