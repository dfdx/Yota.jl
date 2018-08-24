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
include("compile.jl")
include("grad.jl")
include("update.jl")
include("helpers.jl")
include("cuda.jl")
