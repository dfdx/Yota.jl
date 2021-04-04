using IRTools
using ChainRules


include("trie.jl")
include("trace.jl")
include("chainrules.jl")



inc(x) = x + 1
mul(x, y) = x * y
double(x) = mul(x, 2)

foo(x, y) = cos(x) + inc(y)


function grad(f, args...)

end


const PRIMITIVES = TypeTrie()


function __init__()
    for Ts in rrule_primitives()
        push!(PRIMITIVES, Ts)
    end
end


function main()
    f = foo
    args = (2.0, 3.0)
    fargs = (f, args...)

    __init__()

    ir = IRTools.@code_ir f(args...)

    t = IRTracer(f, args, PRIMITIVES)
    tape = t(fargs...)

    _, tape = trace(fargs...)

end