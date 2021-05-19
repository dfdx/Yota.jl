abstract type AbstractDevice end

struct CPU <: AbstractDevice
end

struct GPU <: AbstractDevice
    id::Int
end

# currently GPU's ID is just a placeholder
GPU() = GPU(1)

device_of(A) = A isa CuArray ? GPU(1) : CPU()
to_same_device(A, example) = device_of(example)(A)


to_device_simple(::CPU, x::CuArray) = convert(Array, x)
to_device_simple(::CPU, x::AbstractArray) = x
to_device_simple(::CPU, x::Real) = x
to_device_simple(::GPU, x::AbstractArray) = cu(x)
to_device_simple(::GPU, x::Real) = Float32(x)


"""
Convert object to a compatible with the specified device.

For CPU it's usually no-op. For GPU behavior differs between object types:

 * Arrays are converted to CuArrays
 * structs are converted recursively
 * all other objects are returned as is
"""
function to_device(device::Union{CPU, GPU}, x)
    T = typeof(x)
    flds = fieldnames(T)
    if isa(x, Tuple)
        return ((to_device(device, el) for el in x)...,)
    elseif x isa AbstractArray
        return to_device_simple(device, x)
    elseif isempty(flds)
        # primitive or array
        return to_device_simple(device, x)
    else
        # struct, recursively convert and construct type from fields
        fld_vals = [to_device(device, getfield(x, fld)) for fld in flds]
        return T(fld_vals...)
    end
end


(device::CPU)(x) = to_device(device, x)
(device::GPU)(x) = to_device(device, x)

to_cpu(A) = A
to_cpu(A::CuArray) = convert(Array, A)
to_cuda(A) = cu(A)
to_cuda(A::CuArray) = A
