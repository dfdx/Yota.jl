import Statistics
using LinearAlgebra
using OrderedCollections
using ChainRulesCore
using ChainRules
using NNlib
using Ghost
using Ghost: Tape, Variable, V, Call, mkcall, Constant, inputs
using Ghost: bound, compile, play!, isstruct
using Ghost: remove_first_parameter, kwfunc_signature, call_signature


include("helpers.jl")
include("drules.jl")
include("chainrules.jl")
include("grad.jl")
include("update.jl")
include("gradcheck.jl")


function __init__()
    update_chainrules_primitives!()
end