import Statistics
using LinearAlgebra
using Cassette
using Cassette: Tagged, tag, untag, istagged, metadata, hasmetadata,
    enabletagging, @overdub, overdub, canrecurse, similarcontext, fallback
using Espresso


include("helpers.jl")
include("devices.jl")
include("tape.jl")
include("trace.jl")
include("diffrules.jl")
include("grad.jl")
