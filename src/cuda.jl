# CUDA specific parts; doesn't require any CUDA package

import Espresso: rewrite, rewrite_all

const CUDA_NATIVE_RULES = [
    :($log.(x)) => :(CUDAnative.log.(x)),
    :($exp.(x)) => :(CUDAnative.exp.(x)),
    :($sqrt.(x)) => :(CUDAnative.sqrt.(x)),
    :($(^).(x, n)) => :(CUDAnative.pow.(x, Float32(n))),
    :($ones(n)) => :(CuArray(ones(Float32, n))),
    # :(transpose(x)) => :(permutedims(x, (2,1))),  -- seems to cauase segfault in complex cases
]


# note: rewriting function objects to symbolic names
to_cuda_ex(ex::Expr) = rewrite_all(ex, CUDA_NATIVE_RULES; phs=[:x, :n])
