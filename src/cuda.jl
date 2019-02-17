import CUDAnative
using CuArrays

# const NON_DISPATCHED_OPS = [log, exp, sqrt, ^, ones]
# const CUDA_NATIVE_OPS = Dict{Function,Function}(op => op for op in NON_DISPATCHED_OPS)
const CUDANATIVE_OPS = Dict{Function,Function}()

CUDANATIVE_OPS[log] = CUDAnative.log
CUDANATIVE_OPS[exp] = CUDAnative.exp
CUDANATIVE_OPS[sqrt] = CUDAnative.sqrt
CUDANATIVE_OPS[^] = CUDAnative.pow
CUDANATIVE_OPS[ones] = CUDAnative.ones

device_function(device::GPU, f::Function) = get(CUDANATIVE_OPS, f, f)


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


function cuarray_compatible_tform(tape::Tape)
    new_tape = similar(tape)
    changed = false
    for op in tape
        if op isa Call && haskey(CUDANATIVE_OPS, op.fn)
            changed = true
            push!(new_tape, copy_with(op, fn=CUDANATIVE_OPS[op.fn]))
        else
            push!(new_tape, op)
        end
    end
    return new_tape, changed
end


"""
Transform function to CuArrays compatible.
"""
function cuda_compatible(f, args)
    if haskey(CUDANATIVE_OPS, f)
        return CUDANATIVE_OPS[f]
    else
        return transform(cuarray_compatible_tform, f, args)
    end
end


to_device(device::GPU, x) = cu(x)
to_device(device::GPU, f::Function, args) = cuda_compatible(f, args)
