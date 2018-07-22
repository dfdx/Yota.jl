## data types to track execution and write operations to a tape

## TRACKED REAL

mutable struct TReal <: Real
    tape::Tape
    id::Int        # ID of cooresponding operation on the tape
    val::Real      # value
end

Base.show(io::IO, x::TReal) = print(io, "%$(x.id) = $(x.val)")
tracked(tape::Tape, x::Real) =
    TReal(tape, -1, x)
# Base.zero(x::TReal) = tracked(x.tape, zero(x.val))
getvalue(x::TReal) = x.val



## TRACKED ARRAY

mutable struct TArray{T,N} <: AbstractArray{T,N}
    # see definition of TReal above for field meaning
    tape::Tape
    id::Int
    val::AbstractArray{T,N}
end

Base.show(io::IO, x::TArray{T,N}) where {T,N} =
    print(io, "%$(x.id) = TArray{$T,$N}(size=$(size(x.val)))")
Base.show(io::IO, ::MIME{Symbol("text/plain")}, x::TArray{T,N}) where {T,N} =
    print(io, "%$(x.id) = TArray{$T,$N}(size=$(size(x.val)))")
tracked(tape::Tape, x::AbstractArray) =
    TArray(tape, -1, x)
# Base.zero(x::TArray) = tracked(x.tape, zero(x.val))
getvalue(x::TArray) = x.val


## PROMOTION

# Base.promote_rule(::Type{TReal}, ::Type{<:Real}) = TReal
# function Base.convert(::Type{TReal}, x::R) where {R <: Real}
#     return record!()
#     var = genname()
#     g = get_default_graph()
#     push!(g, ExNode{:constant}(var, x; val=x))
#     return TReal(g, var, x)
# end
# Base.convert(::Type{TReal}, x::TReal) = x
# Base.convert(::Type{R}, x::TReal) where {R <: Real} = x.val


# Base.promote_rule(::Type{TArray}, ::Type{<:AbstractArray}) = TArray
# function Base.convert(::Type{TArray}, x::A) where {A <: AbstractArray}
#     var = genname()
#     g = get_default_graph()
#     push!(g, ExNode{:constant}(var, x; val=x))
#     return TArray(g, var, x)
# end
# Base.convert(::Type{TArray}, x::TArray) = x


## OTHER TYPES AND UTILS

const TAny = Union{TReal, TArray}
tracked(x::TAny) = x

getvalue(x) = x


# # TODO: finish
# function tracked(tape::Tape, s; cont=nothing, field=nothing)
#     if isstruct(s)
#         tr_fields = Array{Any}(undef, length(fieldnames(typeof(s))))
#         for (i, f) in enumerate(fieldnames(typeof(s)))
#             tr_fields[i] = tracked(tape, getfield(s, f); cont=s, field=f)
#         end
#         return typeof(s)(tr_fields...)
#     else
#         return s
#     end
# end
