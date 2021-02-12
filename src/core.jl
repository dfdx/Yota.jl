import Statistics
using LinearAlgebra
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


const BEST_AVAILABLE_DEVICE = Ref{AbstractDevice}(CPU())

if CUDA.functional()
    try
        BEST_AVAILABLE_DEVICE[] = GPU(1)        
    catch exc
        # something is wrong with the user's set-up (or there's a bug in CuArrays)
        @warn "CUDA is installed, but not working properly" exception=(exc, catch_backtrace())

    end
end


best_available_device() = BEST_AVAILABLE_DEVICE[]
