using Test
using Yota
using Yota: Tape, Input, Call, Constant, trace, play!, transform, binarize_ops
using Yota: mean_grad, setfield_nested!, copy_with, simplegrad, remove_unused

include("test_tracer.jl")
include("gradcheck.jl")
include("test_simple.jl")
include("test_grad.jl")
include("test_dynamic.jl")
include("test_simplegrad.jl")
include("test_update.jl")
include("test_transform.jl")
include("test_examples.jl")
