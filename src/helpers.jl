
# function ungetindex!(dx::AbstractArray, x::AbstractArray, dy, ti...)
#     i = map(getvalue, ti)
#     dx[[i...]] .= ds
#     return dx
# end

# function ungetindex!(dx::AbstractArray, x::AbstractArray, ds, i::AbstractArray{Int})
#     dx[i] .= ds
#     return dx
# end


# function ungetindex(x::AbstractArray, ds, i...)
#     dx = zero(x)
#     return ungetindex!(dx, x, ds, i...)
# end



function ungetindex(dy, x::AbstractArray, i::Real)
    dx = zero(x)
    dx[i] = dy
    return dx
end


function sum_grad(x::AbstractArray, ds; opts...)
    return ones(size(x)) .* ds
end


function mean_grad(x::AbstractArray, ds; opts...)
    return ones(size(x)) ./ length(x) .* ds
end


@require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    import CuArrays: CuArray
    
    function sum_grad(x::CuArray, ds; opts...)
        return cu(ones(size(x))) .* ds
    end
    

    function mean_grad(x::CuArray, ds; opts...)
        return cu(ones(size(x))) ./ length(x) .* ds
    end
end
