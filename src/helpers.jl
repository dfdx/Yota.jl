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
