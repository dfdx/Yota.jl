import Statistics
using LinearAlgebra
using Cassette
using Cassette: Tagged, tag, untag, istagged, metadata, hasmetadata,
    enabletagging, @overdub, overdub, canrecurse, similarcontext, fallback
using JuliaInterpreter
using Espresso
using CUDAapi


include("utils.jl")
include("helpers.jl")
include("devices.jl")
include("tape.jl")
include("tapeutils.jl")
include("trace/trace.jl")
include("diffrules/diffrules.jl")
include("grad.jl")
include("compile.jl")
include("update.jl")
include("transform.jl")


# function __init__()
#     @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("cuda.jl")
# end

const BEST_AVAILABLE_DEVICE = Ref{AbstractDevice}(CPU())

if has_cuda()
    try
        using CuArrays
        using CUDAnative

        BEST_AVAILABLE_DEVICE[] = GPU(0)

        include("cuda.jl")
    catch ex
        # something is wrong with the user's set-up (or there's a bug in CuArrays)
        @warn "CUDA is installed, but CuArrays.jl fails to load" exception=(ex,catch_backtrace())

    end
end


best_available_device() = BEST_AVAILABLE_DEVICE[]
