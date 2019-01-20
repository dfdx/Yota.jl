
# when overloading new functions, don't forget to import them first

# *
for (N1, N2) in [(1, 2), (2, 1), (2, 2)]
    @eval *(x::TArray{T1,$N1}, y::TArray{T2,$N2}) where {T1,T2} =
        record!(x.tape, Call, *, (x, y))
    @eval *(x::TArray{T1, $N1}, y::AbstractArray{T2, $N2}) where {T1, T2} =
        record!(x.tape, Call, *, (x, constant(x.tape, y)))
    @eval *(x::AbstractArray{T1, $N1}, y::TArray{T2, $N2}) where {T1, T2} =
        record!(y.tape, Call, *, (constant(y.tape, x), y))
end

# +
+(x::TArray{T,N}, y::TArray{T,N}) where {T,N} = record!(x.tape, Call, +, (x, y))
+(x::TArray{T1, N}, y::AbstractArray{T2, N}) where {T1, T2, N} =
    record!(x.tape, Call, +, (x, constant(x.tape, y)))
+(x::AbstractArray{T1, N}, y::TArray{T2, N}) where {T1, T2, N} =
    record!(y.tape, Call, +, (constant(y.tape, x), y))

# -
-(x::TArray{T,N}, y::TArray{T,N}) where {T,N} = record!(x.tape, Call, -, (x, y))
-(x::TArray{T1, N}, y::AbstractArray{T2, N}) where {T1, T2, N} =
    record!(x.tape, Call, -, (x, constant(x.tape, y)))
-(x::AbstractArray{T1, N}, y::TArray{T2, N}) where {T1, T2, N} =
    record!(y.tape, Call, -, (constant(y.tape, x), y))


sum(x::TArray, dims) = record!(x.tape, Call, sum, (x,); kwargs=Dict{Symbol,Any}(:dims=>dims))
sum(x::TArray; dims=nothing) = (dims == nothing ? record!(x.tape, Call, sum, (x,)) :
                                record!(x.tape, Call, sum, (x,);
                                        kwargs=Dict{Symbol,Any}(:dims=>dims)))
mean(x::TArray, dims) = record!(x.tape, Call, mean, (x,); kwargs=Dict{Symbol,Any}(:dims=>dims))
mean(x::TArray; dims=nothing) = (dims == nothing ? record!(x.tape, Call, mean, (x,)) :
                                record!(x.tape, Call, mean, (x,);
                                        kwargs=Dict{Symbol,Any}(:dims=>dims)))
dropdims(x::TArray; dims) = record!(x.tape, Call, dropdims, (x,); kwargs=Dict{Symbol,Any}(:dims=>dims))
transpose(x::TArray) = record!(x.tape, Call, transpose, (x,))
# is it correct to replace x' with transpose(x)? please ping me if you believe it's wrong
adjoint(x::TArray) = record!(x.tape, Call, transpose, (x,))
minimum(x::TArray) = record!(x.tape, Call, minimum, (x,))
maximum(x::TArray) = record!(x.tape, Call, maximum, (x,))
getindex(x::TArray, i...) = record!(x.tape, Call, getindex, (x, i...))
# I'm not really sure we should write setindex! to the tape since it isn't proper op
# and can be removed as unused op
# setindex!(x::TArray, v::Real, I::Vararg{<:Real}) =
#     record!(x.tape, Call, setindex!, (x, constant(x.tape, v), [constant(x.tape, i) for i in I]...))
ungetindex(dy::TAny, x::TArray, i::Real) =
    record!(x.tape, Call, ungetindex, (dy, x, constant(x.tape, i)))
ungetindex(dy::TAny, x::TArray, i::Real, j::Real) =
    record!(x.tape, Call, ungetindex, (dy, x, constant(x.tape, i), constant(x.tape, j)))
reshape(x::TArray, dims::Vararg{Int64,N}) where N = record!(x.tape, Call, reshape, (x, dims))

include("linalg.jl")

for fn in (+, -)
    @eval Broadcast.broadcasted(::typeof($fn), x::TArray, y::TArray) =
        record!(x.tape, Bcast, $fn, (x, y))
    @eval Broadcast.broadcasted(::typeof($fn), x::TArray, y::Real) =
        record!(x.tape, Bcast, $fn, (x, constant(x.tape, y)))
    @eval Broadcast.broadcasted(::typeof($fn), x::Real, y::TArray) =
        record!(y.tape, Bcast, $fn, (constant(y.tape, x), y))
    # we may also easily handle constant arrays,
    # but let's see if there's actually a use case for this
end

for fn in (*, /)
    @eval Broadcast.broadcasted(::typeof($fn), x::TArray, y::TArray) =
        record!(x.tape, Bcast, $fn, (x, y))
    # Julia automatically translates `a * x`, where a::Real and x::AbstractArray, into `a .* x`
    # but if write it as Bcast, it breaks Transpose and Adjoint
    @eval Broadcast.broadcasted(::typeof($fn), x::TArray, y::Real) =
        record!(x.tape, Call, $fn, (x, constant(x.tape, y)))
    @eval Broadcast.broadcasted(::typeof($fn), x::Real, y::TArray) =
        record!(y.tape, Call, $fn, (constant(y.tape, x), y))
    # we may also easily handle constant arrays,
    # but let's see if there's actually a use case for this
end


for fn in (sin, cos, abs, abs2)
    @eval Broadcast.broadcasted(::typeof($fn), x::TArray) =
        record!(x.tape, Bcast, $fn, (x,))
end

# functions that require special treatment with CuArrays
for fn in (Base.log, Base.exp, Base.sqrt)
    @eval function Broadcast.broadcasted(::typeof($fn), x::TArray)
        record!(x.tape, Bcast, device_op(x.tape.device, $fn), (x,))        
    end
    # @eval function Broadcast.broadcasted(::typeof($fn), x::TArray)
    #     if is_cuarray(x.val)
    #         record!(x.tape, Bcast, cuda_op($fn), (x,))
    #     else
    #         record!(x.tape, Bcast, $fn, (x,))
    #     end
    # end
end


# another CuArrays special case: ^
function Broadcast.broadcasted(::typeof(^), x::TArray, y::TArray)
    op = device_op(x.tape.devide, ^)
    record!(x.tape, Bcast, op, (x, y))
end
function Broadcast.broadcasted(::typeof(^), x::TArray, y::Real)
    op = device_op(x.tape.device, ^)
    record!(x.tape, Bcast, op, (x, constant(x.tape, y)))
end
function Broadcast.broadcasted(::typeof(^), x::Real, y::TArray)
    op = device_op(x.tape.device, ^)
    record!(x.tape, Bcast, op, (constant(y.tape, x), y))
end


# catchall - might replace previous definitions

Broadcast.broadcasted(fn::Fn, x::TArray{T,N}) where {Fn,T,N} = record!(x.tape, Bcast, fn, (x,))
