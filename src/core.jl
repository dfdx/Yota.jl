import Statistics
using LinearAlgebra
using ChainRulesCore
using ChainRules
using Compat
using Compat: @constprop
using NNlib
using Umlaut
import Umlaut: record_primitive!, isprimitive, BaseCtx
import Umlaut: Tracer, trace!, getcode


const V = Umlaut.Variable
const broadcasted = Broadcast.broadcasted


include("helpers.jl")
include("utils.jl")
include("deprecated.jl")
include("chainrules.jl")
include("rulesets.jl")
include("grad.jl")
include("update.jl")      # TODO: remove this, update docs to use Optimisers.jl instead
include("gradcheck.jl")


Umlaut.SHOW_CONFIG.compact = true
