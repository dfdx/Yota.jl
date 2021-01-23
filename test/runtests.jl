using Test
using Yota
using Yota: Tape, Input, Call, Constant, trace, play!, transform, binarize_ops
using Yota: ∇mean, setfield_nested!, copy_with, simplegrad, remove_unused
using Yota: find_field_source_var, unwind_iterate, eliminate_common
using Yota: unvcat, unhcat, uncat, ungetindex!, ungetindex
using CUDA


CUDA.allowscalar(false)


include("test_trace.jl")
include("gradcheck.jl")
include("test_helpers.jl")
include("test_simple.jl")
include("test_grad.jl")
include("test_distributions.jl")
include("test_dynamic.jl")
include("test_simplegrad.jl")
include("test_update.jl")
include("test_transform.jl")
include("test_examples.jl")
