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
mutable struct FunctionResolver{T}
    signatures::Dict{Symbol, Vector{Pair{Any, T}}}
    ordered::Bool
    FunctionResolver{T}() where T = new{T}(Dict(), false)
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


function_type(@nospecialize sig) = sig isa UnionAll ? function_type(sig.body) : sig.parameters[1]
function_type_key(fn_typ) = Symbol("$(Base.parentmodule(fn_typ)).$(Base.nameof(fn_typ))")


function Base.setindex!(rsv::FunctionResolver{T}, val::T, @nospecialize sig::Type{<:Tuple}) where T
    fn_typ = function_type(sig)
    key = function_type_key(fn_typ)
    if !haskey(rsv.signatures, key)
        rsv.signatures[key] = Pair{Type, T}[]
    end
    push!(rsv.signatures[key], sig => val)
    rsv.ordered = false
    return val
end

function Base.getindex(rsv::FunctionResolver{T}, @nospecialize sig::Type{<:Tuple}) where T
    rsv.ordered || order!(rsv)
    fn_typ = function_type(sig)
    key = function_type_key(fn_typ)
    if haskey(rsv.signatures, key)
        for (TT, val) in rsv.signatures[key]
            if sig <: TT
                return val
            end
        end
    end
    return nothing
end


function isless_signature(sig1, sig2)
    # signatures with Varargs should go last
    if any([p isa Type && p <: Vararg for p in get_type_parameters(sig2)])
        return true
    else
        return sig1 <: sig2
    end
end

function order!(rsv::FunctionResolver)
    for (fn_typ, sigs) in rsv.signatures
        sort!(sigs, lt=(p1, p2) -> isless_signature(p1[1], p2[1]))
    end
    rsv.ordered = true
end

Base.haskey(rsv::FunctionResolver, sig::Type{<:Tuple}) = (rsv[sig] !== nothing)
Base.in(sig::Type{<:Tuple}, rsv::FunctionResolver) = haskey(rsv, sig)
Base.empty!(rsv::FunctionResolver) = empty!(rsv.signatures)


function find_signatures_for(rsv::FunctionResolver, f::Union{Function, DataType})
    return rsv.signatures[function_type_key(typeof(f))]
end