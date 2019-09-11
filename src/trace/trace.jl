function __new__(T, args...)
    # @show T
    # @show args
    # note: we also add __new__() to the list of primitives so it's not overdubbed recursively
    if T <: NamedTuple
        return T(args)
    else
        return T(args...)
    end
end


__tuple__(args...) = tuple(args...)
__getfield__(args...) = getfield(args...)


function module_functions(modl)
    res = Vector{Function}()
    for s in Base.names(modl; all=true)
        isdefined(modl, s) || continue
        fn = getfield(modl, s)
        if fn isa Function && match(r"^[a-z#]+$", string(s)) != nothing
            push!(res, fn)
        end
    end
    return res
end

const PRIMITIVES = Set{Any}(vcat(
    module_functions(Base),
    module_functions(Core),
    [Broadcast.materialize, Broadcast.broadcasted, Colon(),
     # our own special functions
     __new__, __tuple__, __getfield__, namedtuple]))


include("cassette.jl")
include("interp.jl")


trace = ctrace
