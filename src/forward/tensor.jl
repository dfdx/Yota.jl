*(x::TArray, y::TArray) = record!(x.tape, Call, *, (x, y))
sum(x::TArray; dims) = record!(x.tape, Call, sum, (x,); kwargs=Dict{Symbol,Any}(:dims=>dims))
sum(x::TArray) = record!(x.tape, Call, sum, (x,))
mean(x::TArray; dims) = record!(x.tape, Call, mean, (x,); kwargs=Dict{Symbol,Any}(:dims=>dims))
mean(x::TArray) = record!(x.tape, Call, mean, (x,))
transpose(x::TArray) = record!(x.tape, Call, transpose, (x,))
minimum(x::TArray) = record!(x.tape, Call, minimum, (x,))
maximum(x::TArray) = record!(x.tape, Call, maximum, (x,))
getindex(x::TArray, i...) = record!(x.tape, Call, getindex, (x, i...))
reshape(x::TArray, dims::Vararg{Int64,N}) where N = record!(x.tape, Call, reshape, (x, dims))

for fn in (*, /, +, -)
    @eval Broadcast.broadcasted(::typeof($fn), x::TArray, y::TArray) =
        record!(x.tape, Bcast, $fn, (x, y))
    @eval Broadcast.broadcasted(::typeof($fn), x::TArray, y::Real) =
        record!(x.tape, Bcast, $fn, (x, y))
    @eval Broadcast.broadcasted(::typeof($fn), x::Real, y::TArray) =
        record!(x.tape, Bcast, $fn, (x, y))
end

for fn in (sin, cos, log, exp, abs, abs2)
    @eval Broadcast.broadcasted(::typeof($fn), x::TArray) =
        record!(x.tape, Bcast, $fn, (x,))
end
