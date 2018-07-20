import Base: *, /, +, -, sin, cos, exp, log, abs, abs2, sign, tanh
import Base: sum, transpose, minimum, maximum, getindex, reshape
import LinearAlgebra: mul!
import Statistics: mean

include("fwddecl.jl")
include("utils.jl")
include("tape.jl")
include("tracked.jl")
include("ops.jl")
include("scalar.jl")
include("tensor.jl")
include("scalargrad.jl")
include("tensorgrad.jl")
include("back.jl")
include("helpers.jl")


## ways to create tracked variables:
## 1. tracked(tape, val) - create var bound, but not written to tape
## 2. record!(tape, Op, ...) - create var by executin Op, put resulting op onto the tape
