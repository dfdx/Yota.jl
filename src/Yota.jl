module Yota

export
    grad,
    update!,
    @diffrule,
    @diffrule_kw,
    @nodiff,
    best_available_device,
    to_device,
    CPU,
    GPU


include("core.jl")

end
