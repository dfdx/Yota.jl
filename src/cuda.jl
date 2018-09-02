import CUDAnative
using CuArrays

# const NON_DISPATCHED_OPS = [log, exp, sqrt, ^, ones]
# const CUDA_NATIVE_OPS = Dict{Function,Function}(op => op for op in NON_DISPATCHED_OPS)
const CUDA_NATIVE_OPS = Dict{Function,Function}()

CUDA_NATIVE_OPS[log] = CUDAnative.log
CUDA_NATIVE_OPS[exp] = CUDAnative.exp
CUDA_NATIVE_OPS[sqrt] = CUDAnative.sqrt
CUDA_NATIVE_OPS[^] = CUDAnative.pow
CUDA_NATIVE_OPS[ones] = CUDAnative.ones


to_device(device::GPU, x) = cu(x)
device_op(device::GPU, op) = get(CUDA_NATIVE_OPS, op, op)


# import Espresso: rewrite, rewrite_all

# const CUDA_NATIVE_RULES = [
#     :($log.(x)) => :(CUDAnative.log.(x)),
#     :($exp.(x)) => :(CUDAnative.exp.(x)),
#     :($sqrt.(x)) => :(CUDAnative.sqrt.(x)),
#     :($(^).(x, n)) => :(CUDAnative.pow.(x, Float32(n))),
#     :($ones(n)) => :(CuArray(ones(Float32, n))),
#     # :(transpose(x)) => :(permutedims(x, (2,1))),  -- seems to cauase segfault in complex cases
# ]


# # note: rewriting function objects to symbolic names
# to_cuda_ex(ex::Expr) = rewrite_all(ex, CUDA_NATIVE_RULES; phs=[:x, :n])


# with_cuda() = false
# has_cuda_inputs(tape::Tape) = false
# @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
#     with_cuda() = true

#     function has_cuda_inputs(tape::Tape)
#         res = false
#         for op in tape
#             if op isa Input && op.val isa CuArray
#                 res = true
#                 break
#             end
#         end
#         return res
#     end
# end
