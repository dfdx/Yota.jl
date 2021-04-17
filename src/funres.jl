"""
Dict-like data structure which maps function signature to a value.
Unlike real dict, getindex(rsv, sig) returns either exact match, or
closest matching function signature. Example:

    rsv = FunctionResolver{Symbol}()
    rsv[(typeof(sin), Float64)] = :Float64
    rsv[(typeof(sin), Real)] = :Real
    rsv[(typeof(sin), Number)] = :Number
    order!(rsv)                      # Important: sort methods before usage

    rsv[(typeof(sin), Float64)]   # ==> :Float64
    rsv[(typeof(sin), Float32)]   # ==> :Real
"""
struct FunctionResolver{T}
    signatures::Dict{Symbol, Vector{Pair{DataType, T}}}
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


function Base.setindex!(rsv::FunctionResolver{T}, val::T, sig::Tuple) where T
    fn_typ = Symbol(sig[1])
    tuple_sig = Tuple{sig...}
    if !haskey(rsv.signatures, fn_typ)
        rsv.signatures[fn_typ] = Pair{Type, T}[]
    end
    push!(rsv.signatures[fn_typ], tuple_sig => val)
    return val
end

function Base.getindex(rsv::FunctionResolver{T}, sig::Tuple) where T
    fn_typ = Symbol(sig[1])
    if haskey(rsv.signatures, fn_typ)
        tuple_sig = Tuple{sig...}
        for (TT, val) in rsv.signatures[fn_typ]
            if tuple_sig <: TT
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

Base.haskey(rsv::FunctionResolver, sig::Tuple) = (rsv[sig] !== nothing)
Base.in(sig::Tuple, rsv::FunctionResolver{Bool}) = haskey(rsv, sig)
Base.empty!(rsv::FunctionResolver) = empty!(rsv.signatures)
