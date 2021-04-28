"""
Dict-like data structure which maps function signature to a value.
Unlike real dict, getindex(rsv, sig) returns either exact match, or
closest matching function signature. Example:

    rsv = FunctionResolver{Symbol}()
    rsv[Tuple{typeof(sin), Float64}] = :Float64
    rsv[Tuple{typeof(sin), Real}] = :Real
    rsv[Tuple{typeof(sin), Number}] = :Number
    order!(rsv)                      # Important: sort methods before usage

    rsv[Tuple{typeof(sin), Float64}]   # ==> :Float64
    rsv[Tuple{typeof(sin), Float32}]   # ==> :Real
"""
struct FunctionResolver{T}
    signatures::Dict{Symbol, Vector{Pair{Any, T}}}
    FunctionResolver{T}() where T = new{T}(Dict())
end

# function FunctionResolver{T}(pairs::Vector{Pair{S, T}}) where {S, T}
# function FunctionResolver{T}(pairs::Vector{Pair{S, T} where S}) where T
function FunctionResolver{T}(pairs::Vector) where T
    rsv = FunctionResolver{T}()
    for (sig, val) in pairs
        rsv[sig] = val
    end
    order!(rsv)
    return rsv
end

Base.show(io::IO, rsv::FunctionResolver) = print(io, "FunctionResolver($(length(rsv.signatures)))")


function_type(sig) = sig isa UnionAll ? function_type(sig.body) : sig.parameters[1]

function Base.setindex!(rsv::FunctionResolver{T}, val::T, sig::Type{<:Tuple}) where T
    fn_typ = Symbol(function_type(sig))
    if !haskey(rsv.signatures, fn_typ)
        rsv.signatures[fn_typ] = Pair{Type, T}[]
    end
    push!(rsv.signatures[fn_typ], sig => val)
    return val
end

function Base.getindex(rsv::FunctionResolver{T}, sig::Type{<:Tuple}) where T
    fn_typ = Symbol(function_type(sig))
    if haskey(rsv.signatures, fn_typ)
        for (TT, val) in rsv.signatures[fn_typ]
            if sig <: TT
                return val
            end
        end
    end
    return nothing
end

function order!(rsv::FunctionResolver)
    for (fn_typ, sigs) in rsv.signatures
        sort!(sigs, lt=(p1, p2) -> p1[1] <: p2[1])
    end
end

Base.haskey(rsv::FunctionResolver, sig::Type{<:Tuple}) = (rsv[sig] !== nothing)
Base.in(sig::Type{<:Tuple}, rsv::FunctionResolver{Bool}) = haskey(rsv, sig)
Base.empty!(rsv::FunctionResolver) = empty!(rsv.signatures)
