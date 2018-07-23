## update inputs vars using calculated gradients

function getfield_nested(s, path)
    v = s
    for name in path
        v = getfield(v, name)
    end
    return v
end

function setfield_nested!(s, path, val)
    container = s
    for name in path[1:end-1]
        container = getfield(container, name)
    end
    setfield!(container, path[end], val)
end


minus_updater(x::AbstractArray, gx) = (x .- gx)
minus_updater(x::Real, gx) = (x - gx)


function update!(x::AbstractArray{T,N}, gx::AbstractArray{T,N}, fn::Function=minus_updater) where {T,N}
    x .= fn(x, gx)
end

function update!(x::Real, gx::Real, fn::Function=minus_updater)
    error("Can't update value of $(typeof(x)) in-place!")
end

function update!(x, gx, fn::Function=minus_updater)
    @assert isstruct(x) "Expected mutable struct as 1st argument"
    @assert gx isa Dict "Gradient for structs should be a dict of (field path -> grad value)"
    for (path, gv) in gx
        v = getfield_nested(x, path)
        setfield_nested!(x, path, fn(v, gv))
    end
end
