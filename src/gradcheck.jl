import FiniteDifferences

function ngradient(f, args...)
    fdm = FiniteDifferences.central_fdm(5, 1)
    FiniteDifferences.grad(fdm, f, args...)
end


function gradcheck(f, args...; atol=1e-5, rtol=1e-5)
    y_grads = grad(f, args...)[2]
    # don't check gradient w.r.t. function since ngradient can't do it
    y_grads = y_grads[2:end]
    n_grad = ngradient(f, args...)
    results = []
    for n in 1:length(args)
        if y_grads[n] isa NoTangent
            push!(results, true)
        else
            push!(results, isapprox(y_grads[n], n_grad[n], rtol=rtol, atol=atol))
        end
    end
    return all(results)
end