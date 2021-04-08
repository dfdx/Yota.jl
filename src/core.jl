import Statistics
using LinearAlgebra
using Setfield
using Espresso
using Distributions
using ChainRulesCore
using CUDA


include("utils.jl")
include("scatter/scatter.jl")
include("helpers.jl")
include("devices.jl")
include("tape.jl")
include("tapeutils.jl")
include("trace.jl")
include("diffrules/diffrules.jl")
include("grad.jl")
include("compile.jl")
include("update.jl")
include("transform.jl")
include("cuda.jl")
include("gradcheck.jl")


const BEST_AVAILABLE_DEVICE = Ref{AbstractDevice}(CPU())

if CUDA.functional()
    BEST_AVAILABLE_DEVICE[] = GPU(1)
end

best_available_device() = BEST_AVAILABLE_DEVICE[]


# step 1: update tape to support Variable & constants in Calls, compilation, derivatives
# step 1.1: update tape to support arguments instead of custom Input type
# step 2: update primitives to support methods signatures, not just functions
# step 3: update gradient calculation to resolve function signatures
# step 4: add ChainRules support