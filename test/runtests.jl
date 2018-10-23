using Test
using Yota
using Yota: Tape, tracked, record!, exec!, play!, grad, grad!, update!, setfield_nested!
using Yota: gettape, getid, getvalue, setvalue!
using Yota: Input, Constant, Call, Bcast, Assign

include("test_tracked.jl")
include("test_tape.jl")
include("test_macro.jl")
include("test_grad.jl")
include("test_update.jl")
include("test_examples.jl")
