import CUDAnative
using CuArrays


# CuArrays.cufunc(::typeof(^)) = CUDAnative.pow

@diffrule CUDAnative.exp(u::Real) u CUDAnative.exp(u) * dy
@diffrule CUDAnative.pow(u::Real, v::Real) u (v * CUDAnative.pow(u, (v-1)) * dy)
@diffrule CUDAnative.pow(u::Real, v::Real) v CUDAnative.log(u) * CUDAnative.pow(u, v) * dy
@diffrule CUDAnative.log(u::Real) u dy / u
@diffrule CUDAnative.sqrt(u::Real) u (0.5f0 * CUDAnative.pow(u, -0.5f0) * dy)


# # const NON_DISPATCHED_OPS = [log, exp, sqrt, ^, ones]
# # const CUDA_NATIVE_OPS = Dict{Function,Function}(op => op for op in NON_DISPATCHED_OPS)
const CUDANATIVE_OPS = Dict{Function,Function}()

CUDANATIVE_OPS[log] = CUDAnative.log
CUDANATIVE_OPS[exp] = CUDAnative.exp
CUDANATIVE_OPS[sqrt] = CUDAnative.sqrt
CUDANATIVE_OPS[^] = CUDAnative.pow
CUDANATIVE_OPS[ones] = CUDAnative.ones

device_function(device::GPU, f::Function) = get(CUDANATIVE_OPS, f, f)


function to_device(device::GPU, x)
    T = typeof(x)
    flds = fieldnames(T)
    if is_cuarray(x)
        return x
    elseif isempty(flds)
        # primitive or array
        return cu(x)
    else
        # struct, recursively convert and construct type from fields
        fld_vals = [to_device(device, getfield(x, fld)) for fld in flds]
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
