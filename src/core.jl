import Statistics
using LinearAlgebra
using Setfield
using Espresso
using Distributions
using ChainRules
using CUDA


include("utils.jl")
include("scatter/scatter.jl")
include("helpers.jl")
include("devices.jl")
include("tape.jl")
include("tapeutils.jl")
include("trace.jl")
# include("diffrules/diffrules.jl")
include("chainrules.jl")
include("grad.jl")
include("compile.jl")
include("update.jl")
include("transform.jl")
# include("cuda.jl")
include("gradcheck.jl")


const BEST_AVAILABLE_DEVICE = Ref{AbstractDevice}(CPU())
best_available_device() = BEST_AVAILABLE_DEVICE[]

function __init__()
    if CUDA.functional()
        BEST_AVAILABLE_DEVICE[] = GPU(1)
    end
    update_chain_rules!()
end



# step 1: update tape to support Variable & constants in Calls, compilation, derivatives
# step 1.1: update tape to support arguments instead of custom Input type
# step 2: update primitives to support methods signatures, not just functions
# step 3: update gradient calculation to resolve function signatures
# step 3.1: make it work with broadcast
# step 4: add ChainRules support
# step 5: get rid of Espresso, unbound tape by default (performance tests?)


# maybe remove:
# 1. struct field derivs (is there anything for this in the ChainRuleCore?)
# 2. @nodiff -> replace with DoesNotExist