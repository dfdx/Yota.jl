using Test
using Yota
using Yota: gradtape, gradcheck, update_chainrules_primitives!, compile
using CUDA
import ChainRulesCore: Composite, Zero


CUDA.allowscalar(false)

include("test_funres.jl")
include("test_tape.jl")
include("test_trace.jl")
include("test_grad.jl")
include("test_helpers.jl")
include("test_devices.jl")
# include("test_distributions.jl") -- we may come back to it later
include("test_update.jl")
include("test_examples.jl")
