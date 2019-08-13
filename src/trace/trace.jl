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

const PRIMITIVES = Set([
    # *, /, +, -, sin, cos, sum, Base._sum,
    print, println,
    Base.getproperty, Base.getfield, Base.indexed_iterate,
    # broadcasting
    broadcast, Broadcast.materialize, Broadcast.broadcasted,
    # functions with kw arguments
    Core.apply_type, Core.kwfunc,
    tuple,
    # for loop primitives
    Colon(), Base.iterate, Base.not_int, ===,
    # our own special functions
    __new__, __tuple__, __getfield__, namedtuple])


include("cassette.jl")
include("interp.jl")


trace = itrace
