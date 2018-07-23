using Test
using Yota: Tape, tracked, record!, exec!, play!, grad, grad!
using Yota: gettape, getid, getvalue, setvalue!
using Yota: Input, Constant, Call, Bcast, Assign

include("test_tracked.jl")
include("test_tape.jl")
include("test_grad.jl")
