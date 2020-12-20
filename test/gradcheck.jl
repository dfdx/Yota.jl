# from https://github.com/FluxML/Zygote.jl/blob/master/test/gradcheck.jl

function ngradient(f, xs::AbstractArray...)
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end


# function gradcheck(f, xs...)
#     n_grads = ngradient(f, xs...)
#     y_grads = Yota._grad(f, xs...)[2] |> collect
#     all(isapprox.(n_grads, y_grads, rtol = 1e-5, atol = 1e-5))
# end



function ngradient2(f, xs, n)
    x = xs[n]
    Δ = zero(x)
    for i in 1:length(x)
        δ = sqrt(eps())
        tmp = x[i]
        x[i] = tmp - δ/2
        y1 = f(xs...)
        x[i] = tmp + δ/2
        y2 = f(xs...)
        x[i] = tmp
        Δ[i] = (y2-y1)/δ
    end
    return Δ
end


function gradcheck2(f, xs...; var_idxs=1:length(xs))
    y_grads = Yota._grad(f, xs...)[2] |> collect
    results = []
    for n in var_idxs
        n_grad = ngradient2(f, xs, n)
        push!(results, isapprox(y_grads[n], n_grad, rtol = 1e-5, atol = 1e-5))
    end
    return all(results)
end


gradcheck = gradcheck2

# gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
# gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)
