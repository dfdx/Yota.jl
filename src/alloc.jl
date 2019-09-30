## Memory allocation strategies

abstract type AbstractMemoryPool end


mutable struct SimplePool <: AbstractMemoryPool
end


alloc(mp::SimplePool, T, sz) = T(undef, sz...)
free(mp::SimplePool, a) = ()


mutable struct CachingPool
    cache::Dict{Any, Any}    # (T, sz) => [buffer1, buffer2, ...]
end
CachingPool() = CachingPool(Dict())


function alloc(mp::CachingPool, T, sz)
    key = (T, sz)
    if haskey(mp.cache, key)
        buffers = mp.cache[key]
        arr = pop!(buffers)
        isempty(buffers) &&  delete!(mp.cache, key) 
        return arr
    else
        try
            return T(undef, sz)
        catch e
            if e isa OutOfMemoryError
                # worst case - release all cached memory, run GC and repeat attempt
                mp.cache = Dict()
                GC.gc()
                return T(undef, sz)
            end
        end
    end
end

function free(mp::CachingPool, arr)
    key = (typeof(arr), size(arr))
    if !haskey(mp.cache, key)
        mp.cache[key] = Any[]
    end
    push!(mp.cache[key], arr)
end




using LinearAlgebra

function usage()
    a = rand(5, 4) |> cu
    b = rand(4, 10) |> cu

    mp = CachingPool()
    # alloc x
    x = alloc(mp, CuArray{Float32, 2}, (5, 10))
    mul!(x, a, b)
    # free x when it's not needed anymore
    free(mp, x)
    x = nothing
end
