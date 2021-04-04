
@diffrule CUDA.exp(u::Real) u CUDA.exp(u) * dy
@diffrule CUDA.pow(u::Real, v::Real) u (v * CUDA.pow(u, (v-1)) * dy)
@diffrule CUDA.pow(u::Real, v::Real) v CUDA.log(u) * CUDA.pow(u, v) * dy
@diffrule CUDA.log(u::Real) u dy / u
@diffrule CUDA.sqrt(u::Real) u (0.5f0 * CUDA.pow(u, -0.5f0) * dy)


# # const NON_DISPATCHED_OPS = [log, exp, sqrt, ^, ones]
# # const CUDA_NATIVE_OPS = Dict{Function,Function}(op => op for op in NON_DISPATCHED_OPS)
const CUDANATIVE_OPS = Dict{Function,Function}()

CUDANATIVE_OPS[log] = CUDA.log
CUDANATIVE_OPS[exp] = CUDA.exp
CUDANATIVE_OPS[sqrt] = CUDA.sqrt
CUDANATIVE_OPS[^] = CUDA.pow
CUDANATIVE_OPS[ones] = CUDA.ones

device_function(device::GPU, f::Function) = get(CUDANATIVE_OPS, f, f)


to_device(device::CPU, x::CuArray) = convert(Array, x)


function to_device(device::GPU, x)
    T = typeof(x)
    flds = fieldnames(T)
    if is_cuarray(x)
        return x
    elseif isa(x, AbstractFloat)
        return Float32(x)
    elseif isa(x, Tuple)
        return ((to_device(device, el) for el in x)...,)
    elseif isempty(flds)
        # primitive or array
        return cu(x)
    else
        # struct, recursively convert and construct type from fields
        fld_vals = [to_device(device, getfield(x, fld)) for fld in flds]
        return T(fld_vals...)
    end
end