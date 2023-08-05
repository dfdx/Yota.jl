# TODO: review and remove unused
function ungetfield(dy, s::Tuple, f::Int)
    T = typeof(s)
    return Tangent{T}([i == f ? dy : ZeroTangent() for i=1:length(s)]...)
end

function ungetindex!(dx::AbstractArray, x::AbstractArray, dy::AbstractArray, I...)
    idx = CartesianIndices(size(x))[I...]
    idx = idx isa CartesianIndex ? [idx] : idx
    return NNlib.scatter!(+, dx, dy, idx)
end

# _array_type_only(A::AT) where AT <: AbstractArray{T, N} where {T, N} = AT


function ungetindex!(dx::AbstractArray{<:Number}, x::AbstractArray{<:Number}, dy::Number, I...)
    d_dy = similar(dx, (1,))
    fill!(d_dy, dy)
    ungetindex!(dx, x, d_dy, I...)
end


function ungetindex!(dx::AbstractArray, x::AbstractArray, dy::Any, I...)
    d_dy = similar(dx, (1,))
    fill!(d_dy, dy)
    ungetindex!(dx, x, d_dy, I...)
end


function ungetindex(x::AbstractArray{<:Number}, dy, I...)
    dx = zero(x)
    return ungetindex!(dx, x, dy, I...)
end


function ungetindex(x::AbstractArray{Any}, dy, I...)
    dx = similar(x)
    fill!(dx, ZeroTangent())
    return ungetindex!(dx, x, dy, I...)
end


function ungetindex(x::Tuple, dy, I...)
    dx = ntuple(length(x)) do i
        i in I ? dy : ZeroTangent()
    end
    return Tangent{typeof(x)}(dx...)
end

"""
    _getfield(value, fld)
This function can be used instead of getfield() to bypass Yota rules
during backpropagation.
"""
_getfield(value, fld) = getfield(value, fld)


# reshape Δ to be consistent with x
trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val(ndims(x))))

const ArrayOrBroadcasted = Union{AbstractArray, Broadcast.Broadcasted}

function unbroadcast(x::ArrayOrBroadcasted, Δ)
    if Δ isa ZeroTangent || Δ isa NoTangent || size(x) == size(Δ)
        return Δ
    elseif length(x) == length(Δ)
        return trim(x, Δ)
    else
        sum_dims = ntuple(i -> size(x, i) == 1 ? i : ndims(Δ)+1, Val(ndims(Δ)))
        return trim(x, sum(Δ, dims=sum_dims))
    end
end

unbroadcast(::Number, Δ) = sum(Δ)

function unbroadcast_prod_x(x::ArrayOrBroadcasted, y::ArrayOrBroadcasted, Δ)
    if Δ isa ZeroTangent || Δ isa NoTangent
        return Δ
    elseif size(x) == size(Δ)
        return Δ .* y
    elseif length(x) == length(Δ)
        return trim(x, Δ .* y)
    else
        sum_dims = ntuple(i -> size(x, i) == 1 ? i : ndims(Δ)+1, Val(ndims(Δ)))
        return trim(x, sum(Δ.* y, dims=sum_dims))
    end
end
unbroadcast_prod_y(x::ArrayOrBroadcasted, y::ArrayOrBroadcasted, Δ) = unbroadcast_prod_x(y, x, Δ)

# device_like(example, a) = (device = guess_device([example]); device(a))
array_like(value, example, sz=(1,)) = fill!(similar(example, sz), value)

# unbroadcast_prod_x(x::Number, y::AbstractArray, Δ) = unbroadcast_prod_x(device_like(y, [x]), y, Δ)[1]
unbroadcast_prod_x(x::Number, y::ArrayOrBroadcasted, Δ) = unbroadcast_prod_x(array_like(x, y), y, Δ)[1]
unbroadcast_prod_x(x::ArrayOrBroadcasted, y::Number, Δ) = unbroadcast_prod_x(x, array_like(y, x), Δ)
unbroadcast_prod_y(x::ArrayOrBroadcasted, y::Number, Δ) = unbroadcast_prod_y(x, array_like(y, x), Δ)[1]
unbroadcast_prod_y(x::Number, y::ArrayOrBroadcasted, Δ) = unbroadcast_prod_y(array_like(x, y), y, Δ)


untranspose_vec(ds::Transpose{T, <:AbstractVector{T}}) where T = transpose(ds)
untranspose_vec(ds::Adjoint{T, <:AbstractVector{T}}) where T = adjoint(ds)
untranspose_vec(ds::AbstractMatrix) = dropdims(transpose(ds); dims=2)


function uncat(dy::AbstractArray, n::Int, arrs::AbstractArray...; dims)
    @assert(dims isa Integer, "Can only undo cat() over a single dimension, " *
            "but dimensions $dims were provided")
    dim = dims
    a = arrs[n]
    from = n == 1 ? 1 : sum(size(arr, dim) for arr in arrs[1:n-1]) + 1
    to = from + size(a, dim) - 1
    return dy[[(:) for i=1:dim - 1]..., from:to, [(:) for i=1:length(size(dy)) - dim]...]
end
uncat(dy::ZeroTangent, n::Int, arrs::AbstractArray...; dims) = dy


namedtuple(names, values) = NamedTuple{names}(values)
namedtuple(d::Dict) = NamedTuple{tuple(keys(d)...)}(values(d))


function rev_perm(perm::NTuple{N, Int}) where N
    rperm = Vector{Int}(undef, length(perm))
    for (i, j) in enumerate(perm)
        rperm[j] = i
    end
    return tuple(rperm...)
end

function ∇permutedims(dy, perm)
    rperm = rev_perm(perm)
    return permutedims(dy, rperm)
end