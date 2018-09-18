
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


# TODO: these functions must not only be used with TAny during initial pass
#       but also they must be compiled into generated code and work with ordinary data
#       thus we can't use to_device here or assume presense of a tape
# SOLUTION: to_device(deviceof(x), dx)
#           deviceof(x::TAny) = x.device
#           deviceof(x) = is_cuarray(x) ? GPU(1) : CPU()
function ungetindex(dy, x::AbstractArray, i::Real)
    dx = zero(x)
    dx[i] = dy
    return to_device(deviceof(x), dx)
end


function sum_grad(x::AbstractArray, ds; opts...)
    dx = ones(size(x)) .* ds
    return to_device(deviceof(x), dx)
end


function mean_grad(x::AbstractArray, ds; opts...)
    dx = ones(size(x)) ./ length(x) .* ds
    return to_device(deviceof(x), dx)
end
