import Statistics
using LinearAlgebra
using Setfield
using OrderedCollections
using IRTools
using ChainRules
using CUDA


include("funres.jl")
include("utils.jl")
include("scatter/scatter.jl")
include("helpers.jl")
include("drules.jl")
include("chainrules.jl")
include("devices.jl")
include("tape.jl")
include("trace.jl")
include("grad.jl")
include("compile.jl")
include("update.jl")
include("gradcheck.jl")


const BEST_AVAILABLE_DEVICE = Ref{AbstractDevice}(CPU())
best_available_device() = BEST_AVAILABLE_DEVICE[]

function __init__()
    if CUDA.functional()
        BEST_AVAILABLE_DEVICE[] = GPU(1)
    end
    update_chainrules_primitives!()
end