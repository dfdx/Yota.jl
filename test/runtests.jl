using Test
using Yota
using Yota: Input, Call, Constant, trace, play!
using Yota: mean_grad, setfield_nested!

include("test_tracer.jl")
include("gradcheck.jl")
include("test_simple.jl")
include("test_grad.jl")
include("test_update.jl")
include("test_examples.jl")
