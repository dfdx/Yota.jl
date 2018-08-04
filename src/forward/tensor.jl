
# when overloading new functions, don't forget to import them first

*(x::TArray, y::TArray) = record!(x.tape, Call, *, (x, y))
+(x::TArray{T,N}, y::TArray{T,N}) where {T,N} = record!(x.tape, Call, +, (x, y))
-(x::TArray{T,N}, y::TArray{T,N}) where {T,N} = record!(x.tape, Call, -, (x, y))
sum(x::TArray; dims) = record!(x.tape, Call, sum, (x,); kwargs=Dict{Symbol,Any}(:dims=>dims))
sum(x::TArray) = record!(x.tape, Call, sum, (x,))
mean(x::TArray; dims) = record!(x.tape, Call, mean, (x,); kwargs=Dict{Symbol,Any}(:dims=>dims))
mean(x::TArray) = record!(x.tape, Call, mean, (x,))
dropdims(x::TArray; dims) = record!(x.tape, Call, dropdims, (x,); kwargs=Dict{Symbol,Any}(:dims=>dims))
transpose(x::TArray) = record!(x.tape, Call, transpose, (x,))
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

for fn in (*, /, +, -, ^)
    @eval Broadcast.broadcasted(::typeof($fn), x::TArray, y::TArray) =
        record!(x.tape, Bcast, $fn, (x, y))
    @eval Broadcast.broadcasted(::typeof($fn), x::TArray, y::Real) =
        record!(x.tape, Bcast, $fn, (x, constant(x.tape, y)))
    @eval Broadcast.broadcasted(::typeof($fn), x::Real, y::TArray) =
        record!(y.tape, Bcast, $fn, (constant(y.tape, x), y))
    # we may also easily handle constant arrays,
    # but let's see if there's actually a use case for this
end


for fn in (sin, cos, log, exp, abs, abs2)
    @eval Broadcast.broadcasted(::typeof($fn), x::TArray) =
        record!(x.tape, Bcast, $fn, (x,))
end


# catchall - might replace previous definitions

Broadcast.broadcasted(fn::Fn, x::TArray{T,N}) where {Fn,T,N} = record!(x.tape, Bcast, fn, (x,))
