using Test
using Yota
using Yota: gradtape, gradcheck, update_chainrules_primitives!
using Yota: trace, compile, play!
import ChainRulesCore: Tangent, ZeroTangent


include("test_grad.jl")
include("test_update.jl")
include("test_examples.jl")
