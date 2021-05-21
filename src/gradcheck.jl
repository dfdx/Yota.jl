import FiniteDifferences

function ngradient(f, args...)
  fdm = FiniteDifferences.central_fdm(5, 1)
  FiniteDifferences.grad(fdm, f, args...)
end


# # from https://github.com/FluxML/Zygote.jl/blob/master/test/gradcheck.jl

# function ngradient(f, xs::AbstractArray...)
#   grads = zero.(xs)
#   for (x, Δ) in zip(xs, grads), i in 1:length(x)
#     δ = sqrt(eps())
#     tmp = x[i]
#     x[i] = tmp - δ/2
#     y1 = f(xs...)
#     x[i] = tmp + δ/2
#     y2 = f(xs...)
#     x[i] = tmp
#     Δ[i] = (y2-y1)/δ
#   end
#   return grads
# end


# function ngradient2(f, xs, n)
#     x = xs[n]
#     Δ = zero(x)
#     for i in 1:length(x)
#         δ = sqrt(eps())
#         tmp = x[i]
#         x[i] = tmp - δ/2
#         y1 = f(xs...)
#         x[i] = tmp + δ/2
#         y2 = f(xs...)
#         x[i] = tmp
#         Δ[i] = (y2-y1)/δ
#     end
#     return Δ
# end


function gradcheck(f, args...)
    y_grads = grad(f, args...)[2]
    # don't check gradient w.r.t. function since ngradient can't do it
    y_grads = y_grads[2:end]
    n_grad = ngradient(f, args...)
    results = []
    for n in 1:length(args)
        push!(results, isapprox(y_grads[n], n_grad[n], rtol = 1e-5, atol = 1e-5))
    end
    return all(results)
end


# gradcheck = gradcheck2

# # gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
# # gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)
