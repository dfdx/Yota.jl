using IRTools
using ChainRules


include("trie.jl")
include("trace.jl")
include("chainrules.jl")



inc(x) = x + 1
mul(x, y) = x * y
double(x) = mul(x, 2)

foo(x) = cos(exp(x))


function grad(f, args...)

end


const PRIMITIVES = TypeTrie()


function main()
    # TODO: move to __init__()
    for Ts in rrule_primitives()
        push!(PRIMITIVES, Ts)
    end




    f = foo
    args = (2.0, 3.0)
end