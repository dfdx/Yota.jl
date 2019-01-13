abstract type AbstractDevice end

struct CPU <: AbstractDevice
end

struct GPU <: AbstractDevice
    id::Int
end


"""
Check if the argument is of type CuArray. Doesn't require CuArrays.jl to be loaded
"""
is_cuarray(x) = startswith(string(typeof(x)), "CuArray")

# function has_cuda_inputs(tape::Tape)
#     res = false
#     for op in tape
#         if op isa Input && op.val isa CuArray
#             res = true
#             break
#         end
#     end
#     return res
# end


# currently GPU's ID is just a placeholder
guess_device(args) = any(is_cuarray, args) ? GPU(1) : CPU()

# also see cuda.jl
to_device(device::CPU, x) = x

# also see cuda.jl
device_op(device::CPU, op) = op

deviceof(x) = is_cuarray(x) ? GPU(1) : CPU()
