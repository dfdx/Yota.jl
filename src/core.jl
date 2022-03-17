import Statistics
using LinearAlgebra
using ChainRulesCore
using ChainRules
using NNlib
using Umlaut
import Umlaut: record_primitive!, isprimitive, BaseCtx
import Umlaut: record_or_recurse!, Tracer, trace!, code_info_of


const V = Umlaut.Variable
const broadcasted = Broadcast.broadcasted


include("helpers.jl")
include("utils.jl")
include("deprecated.jl")
include("cr_api.jl")
include("rulesets.jl")
include("grad.jl")
include("update.jl")
include("gradcheck.jl")


Base.show(io::IO, tape::Tape{GradCtx}) = Umlaut.show_compact(io, tape)