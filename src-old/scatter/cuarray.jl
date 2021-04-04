using CUDA


# Integer
for op = [:add, :sub, :max, :min, :and, :or, :xor]
    fn = Symbol("scatter_$(op)!")
    atm_op = Symbol("atomic_$(op)!")
    @eval function $fn(ys::CuMatrix{T}, us::CuArray{T}, xs::CuArray{Int}) where {T<:Integer}
        function kernel!(ys, us, xs)
            li = threadIdx().y + (blockIdx().y - 1) * blockDim().y
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

            @inbounds if li <= length(xs) && i <= size(ys, 1)
                ind = CartesianIndices(xs)[li]
                j = Base._to_linear_index(ys, i, xs[li])
                CUDA.$atm_op(pointer(ys, j), us[i, ind])
            end

            return
        end

        thread_x = min(MAX_THREADS, size(ys, 1))
        thread_y = min(MAX_THREADS ÷ thread_x, length(xs))
        threads = (thread_x, thread_y)
        blocks = ceil.(Int, (size(ys, 1), length(xs)) ./ threads)
        @cuda blocks=blocks threads=threads kernel!(ys, us, xs)
        return ys
    end

    @eval function $fn(ys::CuArray{T}, us::CuArray{T}, xs::CuArray{<:Tuple}) where {T<:Integer}
        function kernel!(ys, us, xs)
            li = threadIdx().y + (blockIdx().y - 1) * blockDim().y
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

            @inbounds if li <= length(xs) && i <= size(ys, 1)
                ind = CartesianIndices(xs)[li]
                j = Base._to_linear_index(ys, i, xs[li]...)
                CUDA.$atm_op(pointer(ys, j), us[i, ind])
            end

            return
        end

        thread_x = min(MAX_THREADS, size(ys, 1))
        thread_y = min(MAX_THREADS ÷ thread_x, length(xs))
        threads = (thread_x, thread_y)
        blocks = ceil.(Int, (size(ys, 1), length(xs)) ./ threads)
        @cuda blocks=blocks threads=threads kernel!(ys, us, xs)
        return ys
    end
end


# Floating point
for op = [:add, :sub, :mul, :div, :max, :min]
    fn = Symbol("scatter_$(op)!")
    atm_op = Symbol("atomic_$(op)!")
    @eval function $fn(ys::CuMatrix{T}, us::CuArray{T}, xs::CuArray{Int}) where {T<:AbstractFloat}
        function kernel!(ys::CuDeviceArray{T}, us::CuDeviceArray{T}, xs)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

            @inbounds if i <= size(ys, 1) && j <= length(xs)
                ind = CartesianIndices(xs)[j]
                k = Base._to_linear_index(ys, i, xs[j])
                CUDA.$atm_op(pointer(ys, k), us[i, ind])
            end

            return
        end

        thread_i = min(MAX_THREADS, size(ys, 1))
        thread_j = min(MAX_THREADS ÷ thread_i, length(xs))
        threads = (thread_i, thread_j)
        blocks = ceil.(Int, (size(ys, 1), length(xs)) ./ threads)
        @cuda blocks=blocks threads=threads kernel!(ys, us, xs)
        return ys
    end

    @eval function $fn(ys::CuArray{T}, us::CuArray{T}, xs::CuArray{<:Tuple}) where {T<:AbstractFloat}
        function kernel!(ys::CuDeviceArray{T}, us::CuDeviceArray{T}, xs)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

            @inbounds if i <= size(ys, 1) && j <= length(xs)
                ind = CartesianIndices(xs)[j]
                k = Base._to_linear_index(ys, i, xs[j]...)
                CUDA.$atm_op(pointer(ys, k), us[i, ind])
            end

            return
        end

        thread_i = min(MAX_THREADS, size(ys, 1))
        thread_j = min(MAX_THREADS ÷ thread_i, length(xs))
        threads = (thread_i, thread_j)
        blocks = ceil.(Int, (size(ys, 1), length(xs)) ./ threads)
        @cuda blocks=blocks threads=threads kernel!(ys, us, xs)
        return ys
    end
end


function scatter_mean!(ys::CuMatrix{T}, us::CuArray{T}, xs::CuArray{<:IntOrTuple}) where {T<:AbstractFloat}
    yt = CUDA.zero(ys)
    ot = CUDA.zero(ys)
    os = CUDA.one.(us)
    scatter_add!(ot, os, xs)
    scatter_add!(yt, us, xs)
    ys .+= save_div.(yt, ot)
    return ys
end
