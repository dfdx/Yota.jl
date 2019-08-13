function ungetindex!(dx::AbstractArray, x::AbstractArray, ds, i...)
    dx[[i...]] .= ds
    return dx
end

function ungetindex!(dx::AbstractArray, x::AbstractArray, ds, i::AbstractArray{Int})
    dx[i] .= ds
    return dx
end


function ungetindex(x::AbstractArray, ds, i...)
    dx = zero(x)
    return ungetindex!(dx, x, ds, i...)
end


function sum_grad(x::AbstractArray, ds)
    dx = similar(x)
    dx .= ds
    return dx
end


function mean_grad(x::AbstractArray, ds)
    dx = similar(x)
    dx .= ds ./ length(x)
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


unbroadcast_prod_x(x::Number, y::AbstractArray, Δ) = unbroadcast_prod_x([x], y, Δ)[1]
unbroadcast_prod_x(x::AbstractArray, y::Number, Δ) = unbroadcast_prod_x(x, [y], Δ)
unbroadcast_prod_y(x::AbstractArray, y::Number, Δ) = unbroadcast_prod_y(x, [y], Δ)[1]
unbroadcast_prod_y(x::Number, y::AbstractArray, Δ) = unbroadcast_prod_y([x], y, Δ)


untranspose_vec(ds::Transpose{T, <:AbstractVector{T}}) where T = transpose(ds)
untranspose_vec(ds::Adjoint{T, <:AbstractVector{T}}) where T = adjoint(ds)
untranspose_vec(ds::AbstractMatrix) = dropdims(transpose(ds); dims=2)

namedtuple(names, values) = NamedTuple{names}(values)
