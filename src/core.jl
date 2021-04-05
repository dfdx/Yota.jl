using IRTools
using ChainRules


include("trie.jl")
include("trace.jl")
include("chainrules.jl")



inc(x) = x + 1
mul(x, y) = x * y
double(x) = mul(x, 2)

foo(x, y) = cos(x) + inc(y)

inc_mul(A::AbstractArray, B::AbstractArray) = inc_mul.(A, B)



const PRIMITIVES = TypeTrie()


function __init__()
    for Ts in rrule_primitives()
        push!(PRIMITIVES, Ts)
    end
    push!(PRIMITIVES, (typeof(Base.broadcast), Vararg))
    push!(PRIMITIVES, (typeof(Base.broadcasted), Vararg))
    push!(PRIMITIVES, (typeof(Base.materialize), Vararg))
    # remove primitive for rrule(::Any, ...)
    delete!(PRIMITIVES.children, Any)
end


function main()
    f = inc_mul
    args = (rand(3), rand(3))
    fargs = (f, args...)

    __init__()

    ir = IRTools.@code_ir f(args...)

    # t = IRTracer(f, args, PRIMITIVES)
    # tape = t(fargs...)

    _, tape = trace(fargs...)

end