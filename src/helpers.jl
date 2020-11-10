function ∇getproperty(dy, s, f::Symbol)
    T = typeof(s)
    nt = NamedTuple{(f,)}((dy,))
    return Composite{T}(; nt...)
end


function ∇getfield(dy, s::Tuple, f::Int)
    T = typeof(s)
    return Composite{T}([i == f ? dy : Zero() for i=1:length(s)]...)
end


function ∇__new__(dy, T, idx)
    fldname = fieldnames(T)[idx]
    return getproperty(dy, fldname)
end


function ungetindex!(dx::AbstractArray, x::AbstractArray, ds, i...)
    dx[i...] .+= ds
    return dx
end

function ungetindex!(dx::AbstractArray, x::AbstractArray, ds, i::AbstractArray{Int})
    dx[i] .+= ds
    return dx
end


function ungetindex(x::AbstractArray, ds, i...)
    dx = zero(x)
    return ungetindex!(dx, x, ds, i...)
end


function ungetindex(x::Tuple, dy, I...)
    dx = map(1:length(x)) do i
        i in I ? dy : zero(x[i])
    end
    return dx
end


function sum_grad(x::AbstractArray, ds)
    dx = similar(x)
    dx .= ds
    return dx
end


# function mean_grad(x::AbstractArray, ds)
#     dx = similar(x)
#     dx .= ds ./ length(x)
#     return dx
# end


function ∇mean(x::AbstractArray, dy, dims=1:ndims(x))
    dx = similar(x)
    dx .= dy ./ prod(size(x, d) for d in dims)
    return dx
end


function sum_dropdims(x::AbstractArray, dims)
    return dropdims(sum(x; dims=dims); dims=dims)
end


# unbroadcast from Flux
# in in-place version we can consider sum!(similar(x), ds),
# but we need to carefully measure performance in each case

# reshape Δ to be consistent with x
trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val(ndims(x))))

function unbroadcast(x::AbstractArray, Δ)
    if size(x) == size(Δ)
        return Δ
    elseif length(x) == length(Δ)
        return trim(x, Δ)
    else
        sum_dims = ntuple(i -> size(x, i) == 1 ? i : ndims(Δ)+1, Val(ndims(Δ)))
        return trim(x, sum(Δ, dims=sum_dims))
    end
end

unbroadcast(x::Number, Δ) = sum(Δ)

function unbroadcast_prod_x(x::AbstractArray, y::AbstractArray, Δ)
    if size(x) == size(Δ)
        return Δ .* y
    elseif length(x) == length(Δ)
        return trim(x, Δ .* y)
    else
        sum_dims = ntuple(i -> size(x, i) == 1 ? i : ndims(Δ)+1, Val(ndims(Δ)))
        return trim(x, sum(Δ.* y, dims=sum_dims))
    end
end
unbroadcast_prod_y(x::AbstractArray, y::AbstractArray, Δ) = unbroadcast_prod_x(y, x, Δ)

device_like(example, a) = (device = guess_device([example]); device(a))
unbroadcast_prod_x(x::Number, y::AbstractArray, Δ) = unbroadcast_prod_x(device_like(y, [x]), y, Δ)[1]
unbroadcast_prod_x(x::AbstractArray, y::Number, Δ) = unbroadcast_prod_x(x, device_like(x, [y]), Δ)
unbroadcast_prod_y(x::AbstractArray, y::Number, Δ) = unbroadcast_prod_y(x, device_like(x, [y]), Δ)[1]
unbroadcast_prod_y(x::Number, y::AbstractArray, Δ) = unbroadcast_prod_y(device_like(y, [x]), y, Δ)


untranspose_vec(ds::Transpose{T, <:AbstractVector{T}}) where T = transpose(ds)
untranspose_vec(ds::Adjoint{T, <:AbstractVector{T}}) where T = adjoint(ds)
untranspose_vec(ds::AbstractMatrix) = dropdims(transpose(ds); dims=2)


function unvcat(dy::AbstractArray, n::Int, arrs::AbstractArray...)
    a = arrs[n]
    from = n == 1 ? 1 : sum(size(arr, 1) for arr in arrs[1:n-1]) + 1
    to = from + size(a, 1) - 1
    return dy[from:to, [(:) for i=1:length(size(dy)) - 1]...]
end


function unhcat(dy::AbstractArray, n::Int, arrs::AbstractArray...)
    a = arrs[n]
    from = n == 1 ? 1 : sum(size(arr, 2) for arr in arrs[1:n-1]) + 1
    to = from + size(a, 2) - 1
    return dy[:, from:to, [(:) for i=1:length(size(dy)) - 2]...]
end


function uncat(dy::AbstractArray, n::Int, arrs::AbstractArray...; dims)
    @assert(dims isa Integer, "Can only undo cat() over a single dimension, " *
            "but dimensions $dims were provided")
    dim = dims
    a = arrs[n]
    from = n == 1 ? 1 : sum(size(arr, dim) for arr in arrs[1:n-1]) + 1
    to = from + size(a, dim) - 1
    return dy[[(:) for i=1:dim - 1]..., from:to, [(:) for i=1:length(size(dy)) - dim]...]
end


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
