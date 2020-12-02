module Scatter

using CUDA: @cuda
const MAX_THREADS = 1024


const IntOrTuple = Union{Integer,Tuple}
const ops = [:add, :sub, :mul, :div, :max, :min, :mean]
const name2op = Dict(:add => :+, :sub => :-, :mul => :*, :div => :/)


include("densearray.jl")
include("cuarray.jl")


to_cpu(A) = A
to_cpu(A::CuArray) = convert(Array, A)
to_cuda(A) = cu(A)
to_cuda(A::CuArray) = A
to_same_device(A, E) = E isa CuArray ? to_cuda(A) : to_cpu(A)


function scatter_add2!(A::AbstractArray, v::AbstractArray, I...)
    # replace (:) with actual index in A
    I = [i == (:) ? (1:size(A, d)) : to_cpu(i) for (d, i) in enumerate(I)]
    # corner case: original scatter_add!() adds (:) to indices which breaks
    # code that uses linear indexing (e.g. `x[1]` where x has > 1 dimensions)
    # to fix it, we fully specify the index
    I = vcat(I, [1 for d in length(I) : ndims(A)-1])
    II = collect(Iterators.product(I...))
    II = to_same_device(II, A)
    A_ = reshape(A, 1, size(A)...)
    v_ = reshape(v, 1, size(v)...)
    A_ = scatter_add!(A_, v_, II)
    return dropdims(A_, dims=1)
end

end
