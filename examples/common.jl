logistic(x) = 1 ./ (1 + exp.(-x))
@diffrule logistic(x::Number) x (logistic(x) .* (1 .- logistic(x)) .* dy)

softplus(x) = log(exp(x) + 1)
@diffrule softplus(x::Number) x logistic(x) .* dy


function xavier_init(dim_in, dim_out; c=1)
    low = -c * sqrt(6.0 / (dim_in + dim_out))
    high = c * sqrt(6.0 / (dim_in + dim_out))
    return rand(Uniform(low, high), dim_in, dim_out)
end
