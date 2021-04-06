using IRTools
using ChainRules


include("trie.jl")
include("trace.jl")
include("chainrules.jl")



inc(x) = x + 1
mul(x, y) = x * y
double(x) = mul(x, 2)

foo(x, y) = cos(x) + inc(y)

inc_mul(a::Real, b::Real) = a * (b + 1.0)
inc_mul(A::AbstractArray, B::AbstractArray) = inc_mul.(A, B)
inc_mul2(A::AbstractArray, B::AbstractArray) = A .* (B .+ 1)


function loop1(a, n)
    a = 2a
    for i in 1:n
        a = mul(a, n)
        n = n + 1
    end
    a = a + n
    return a
end

function loop2(a, b)
    while b > 0
        a = mul(a, b)
        b = b - 1
    end
    return a
end


function loop3(a, b)
    while b > 1
        @show a, b
        b = b - 1
        a = b
        while a < 100
            @show a, b
            a = a * b + 1
        end
    end
    return a
end


function cond1(a, b)
    if b > 0
        a = 2a
    end
    return a
end



const PRIMITIVES = TypeTrie()


function __init__()
    for Ts in rrule_primitives()
        push!(PRIMITIVES, Ts)
    end
    push!(PRIMITIVES, (typeof(Base.broadcast), Vararg))
    push!(PRIMITIVES, (typeof(Base.broadcasted), Vararg))
    push!(PRIMITIVES, (typeof(Base.materialize), Vararg))
    push!(PRIMITIVES, (typeof(getfield), Vararg))
    push!(PRIMITIVES, (typeof(iterate), Vararg))
    push!(PRIMITIVES, (typeof(Base.not_int), Vararg))
    # push!(PRIMITIVES, (UnitRange{Int}, Int, Int))
    push!(PRIMITIVES, (DataType, Int, Int))
    push!(PRIMITIVES, (Colon, Int64, Int64))
    # remove primitive for rrule(::Any, ...)
    delete!(PRIMITIVES.children, Any)
end


function main()
    f = loop1
    args = (2.0, 3)
    fargs = (f, args...)

    __init__()

    ir = IRTools.@code_ir f(args...)
    iro = IRTools.@code_ir f(args...)

    _, tape = trace(fargs...)


    t = IRTracer(f, args, PRIMITIVES)
    tape = t(fargs...)

end