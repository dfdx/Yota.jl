import Statistics
using OrderedCollections
using ChainRules
using CUDA
using Ghost
using Ghost: Tape, Variable, V, Call, mkcall, Constant, inputs
using Ghost: bound, _getfield, compile, play!, isstruct, ungetfield, ungetindex, uncat
using Ghost: unbroadcast, unbroadcast_prod_x, unbroadcast_prod_y
using Ghost: remove_first_parameter, kwfunc_signature, call_signature


include("drules.jl")
include("chainrules.jl")
include("grad.jl")
include("update.jl")
include("gradcheck.jl")


function __init__()
    update_chainrules_primitives!()
end