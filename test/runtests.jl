using Test
using Yota
using Yota: Tape, Input, Call, Constant, trace, itrace, play!, transform, binarize_ops
using Yota: mean_grad, setfield_nested!, copy_with, simplegrad, remove_unused
using Yota: find_field_source_var, unwind_iterate

include("test_trace_cassette.jl")
include("test_trace_interp.jl")
include("gradcheck.jl")
include("test_simple.jl")
include("test_grad.jl")
include("test_dynamic.jl")
include("test_simplegrad.jl")
include("test_update.jl")
include("test_transform.jl")
include("test_examples.jl")
