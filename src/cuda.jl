import CUDAnative
using CuArrays


CuArrays.cufunc(::typeof(^)) = CUDAnative.pow

@diffrule CUDAnative.exp(x::Real) x CUDAnative.exp(x) * ds
@diffrule CUDAnative.pow(x::Real, y::Real) x (y * CUDAnative.pow(x, (y-1)) * ds)
@diffrule CUDAnative.pow(x::Real, y::Real) y CUDAnative.log(x) * CUDAnative.pow(x, y) * ds
@diffrule CUDAnative.log(x::Real) x ds / x
@diffrule CUDAnative.sqrt(x::Real) x (0.5f0 * CUDAnative.pow(x, -0.5f0) * ds)


# # const NON_DISPATCHED_OPS = [log, exp, sqrt, ^, ones]
# # const CUDA_NATIVE_OPS = Dict{Function,Function}(op => op for op in NON_DISPATCHED_OPS)
const CUDANATIVE_OPS = Dict{Function,Function}()

CUDANATIVE_OPS[log] = CUDAnative.log
CUDANATIVE_OPS[exp] = CUDAnative.exp
CUDANATIVE_OPS[sqrt] = CUDAnative.sqrt
CUDANATIVE_OPS[^] = CUDAnative.pow
CUDANATIVE_OPS[ones] = CUDAnative.ones

device_function(device::GPU, f::Function) = get(CUDANATIVE_OPS, f, f)
# device_function(::GPU, f::Function) = CuArrays.cufunc(f)
to_device(device::GPU, x) = cu(x)


function to_cuda(x)
    T = typeof(x)
    flds = fieldnames(T)
    if is_cuarray(x)
        return x
    elseif isempty(flds)
        # primitive or array
        return cu(x)
    else
        # struct, recursively convert and construct type from fields
        fld_vals = [to_cuda(getfield(x, fld)) for fld in flds]
        return T(fld_vals...)
    end
end


# function cuarray_compatible_tform(tape::Tape)
#     new_tape = similar(tape)
#     changed = false
#     for op in tape
#         if op isa Call && haskey(CUDANATIVE_OPS, op.fn)
#             changed = true
#             push!(new_tape, copy_with(op, fn=CUDANATIVE_OPS[op.fn]))
#         else
#             push!(new_tape, op)
#         end
#     end
#     return new_tape, changed
# end


# """
# Transform function to CuArrays compatible.
# """
# function cuda_compatible(f, args)
#     cf = CuArrays.cufunc(f)
#     if f === cf
#         return cf
#     else
#         return transform(cuarray_compatible_tform, f, args)
#     end
# end
