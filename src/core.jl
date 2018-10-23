import Base: *, /, +, -, ^, sin, cos, exp, log, abs, abs2, sign, tanh, sqrt
import Base: sum, dropdims, transpose, adjoint, minimum, maximum, getindex, setindex!, reshape
import LinearAlgebra: mul!
import Statistics: mean
import Espresso: ExGraph, ExNode, matchingex, rewrite
import Espresso
using Requires

include("fwddecl.jl")
include("utils.jl")
include("tape.jl")
include("tracked.jl")
include("ops.jl")
include("forward/scalar.jl")
include("forward/tensor.jl")
include("backward/scalar.jl")
include("backward/tensor.jl")
include("macro.jl")
include("compile.jl")
include("devices.jl")
include("grad.jl")
include("update.jl")
include("helpers.jl")


function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("cuda.jl")
end
