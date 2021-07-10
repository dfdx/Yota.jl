import FiniteDifferences

function ngradient(f, args...)
  fdm = FiniteDifferences.central_fdm(5, 1)
  FiniteDifferences.grad(fdm, f, args...)
end


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