## data types to track execution and write operations to a tape

## TRACKED REAL

mutable struct TReal <: Real
    tape::Tape
    id::Int        # ID of cooresponding operation on the tape
    val::Real      # value
    origin::Any    # original container from which this var was taken from (default: nothing)
    field::Any     # field name or index in container (default: nothing)
end

Base.show(io::IO, x::TReal) = print(io, "%$(x.id) = $(x.val)")
tracked(tape::Tape, x::Real; cont=nothing, field=nothing) =
    TReal(tape, -1, x, cont, field)
# Base.zero(x::TReal) = tracked(x.tape, zero(x.val))
getvalue(x::TReal) = x.val



## TRACKED ARRAY

mutable struct TArray{T,N} <: AbstractArray{T,N}
    # see definition of TReal above for field meaning
    tape::Tape
    id::Int
    val::AbstractArray{T,N}
    origin::Any
    field::Any
end

Base.show(io::IO, x::TArray{T,N}) where {T,N} =
    print(io, "%$(x.id) = TArray{$T,$N}(size=size(x.val))")
Base.show(io::IO, ::MIME{Symbol("text/plain")}, x::TArray{T,N}) where {T,N} =
    print(io, "%$(x.id) = TArray{$T,$N}(size=$(size(x.val)))")
tracked(tape::Tape, x::AbstractArray; cont=nothing, field=nothing) =
    TArray(tape, -1, x, cont, field)
# Base.zero(x::TArray) = tracked(x.tape, zero(x.val))
getvalue(x::TArray) = x.val


## OTHER TYPES AND UTILS

const TAny = Union{TReal, TArray}

getvalue(x) = x



# "Check if an object is of a struct type, i.e. not a number or array"
# isstruct(::Type{T}) where T = !isbits(T) && !(T <: AbstractArray)
# isstruct(obj) = !isbits(obj) && !isa(obj, AbstractArray)

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
