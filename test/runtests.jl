using Test
using Yota
using Yota: gradtape, gradcheck, update_chainrules_primitives!
using Yota: trace, compile, play!
using CUDA
import ChainRulesCore: Tangent, ZeroTangent


CUDA.allowscalar(false)

include("test_grad.jl")
include("test_update.jl")
include("test_examples.jl")
