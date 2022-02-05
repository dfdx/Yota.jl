using Test
using Yota
using Yota: gradtape, gradcheck, update_chainrules_primitives!
using Yota: trace, compile, play!
using Yota: make_rrule, YotaRuleConfig
import ChainRulesCore: Tangent, ZeroTangent, rrule_via_ad

# test-only dependencies
using CUDA


include("test_helpers.jl")
include("test_chainrules.jl")
include("test_rulesets.jl")
include("test_grad.jl")
include("test_update.jl")
include("test_examples.jl")
