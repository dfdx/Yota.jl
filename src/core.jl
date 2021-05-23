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

<<<<<<< HEAD
function __init__()
    if CUDA.functional()
        BEST_AVAILABLE_DEVICE[] = GPU(1)
=======
if CUDA.functional()
    try
        BEST_AVAILABLE_DEVICE[] = GPU(1)        
    catch exc
        # something is wrong with the user's set-up (or there's a bug in CuArrays)
        @warn "CUDA is installed, but not working properly" exception=(exc, catch_backtrace())

>>>>>>> e44bdb1 (Add loop outisder variables as its inputs)
    end
    update_chainrules_primitives!()
end