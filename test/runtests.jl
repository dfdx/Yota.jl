using Test
using Yota
using Yota: Tape, Input, Call, Constant, trace, play!, transform
using Yota: mean_grad, setfield_nested!, copy_with

include("test_tracer.jl")
include("gradcheck.jl")
include("test_simple.jl")
include("test_grad.jl")
include("test_update.jl")
include("test_transform.jl")
include("test_examples.jl")
