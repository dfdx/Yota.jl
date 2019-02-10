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


function sum_grad(x::AbstractArray, ds; opts...)
    return ones(typeof(x), size(x)) .* ds
end


function mean_grad(x::AbstractArray, ds)
    return ones(typeof(x), size(x)) ./ length(x) .* ds
end


function sum_dropdims(x::AbstractArray, dims)
    return dropdims(sum(x; dims=dims); dims=dims)
end
