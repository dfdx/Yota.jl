include("fwddecl.jl")
include("utils.jl")
include("tape.jl")
include("tracked.jl")
include("ops.jl")
include("alg.jl")
include("grad.jl")


## ways to create tracked variables:
## 1. tracked(tape, val) - create var bound, but not written to tape
## 2. record!(tape, Op, ...) - create var by executin Op, put resulting op onto the tape
