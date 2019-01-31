using Yota
using Zygote
using Zygote: @adjoint
using BenchmarkTools
using Distributions
import NNlib


logistic(x) = 1 / (1 + exp(-x))
softplus(x) = log(exp(x) + 1)

# Yota diff rules
@diffrule logistic(x::Number) x (logistic(x) * (1 - logistic(x)) * ds)
@diffrule softplus(x::Number) x logistic(x) * ds

# Zygote diff rules
@adjoint logistic(x) = logistic(x), Δ -> ((logistic(x) * (1 - logistic(x)) * Δ),)
@adjoint softplus(x) = softplus(x), Δ -> ((logistic(x) * Δ),)


function perf_test(f, args...)
    println("Compiling derivatives using Yota")
    @time Yota.grad(f, args...)
    r1 = @benchmark Yota.grad($f, $(args)...)
    show(stdout, MIME{Symbol("text/plain")}(), r1)
    println("\n")

    println("Compiling derivatives using Zygote")
    @time Zygote.gradient(f, args...)
    r2 = @benchmark Zygote.gradient($f, $(args)...)
    show(stdout, MIME{Symbol("text/plain")}(), r2)
    println("\n")
end


# VAE benchmark

function xavier_init(dim_in, dim_out; c=1)
    low = -c * sqrt(6.0 / (dim_in + dim_out))
    high = c * sqrt(6.0 / (dim_in + dim_out))
    return rand(Uniform(low, high), dim_in, dim_out)
end


# see examples/vae.jl for a more convenient API
function encode(We1, be1, We2, be2, We3, be3, We4, be4, x)
    he1 = softplus.(We1 * x .+ be1)
    he2 = softplus.(We2 * he1 .+ be2)
    mu = We3 * he2 .+ be3
    log_sigma2 = We4 * he2 .+ be4
    return mu, log_sigma2
end

function decode(Wd1, bd1, Wd2, bd2, Wd3, bd3, z)
    hd1 = softplus.(Wd1 * z .+ bd1)
    hd2 = softplus.(Wd2 * hd1 .+ bd2)
    x_rec = logistic.(Wd3 * hd2 .+ bd3)
    return x_rec
end


function vae_cost(We1, be1, We2, be2, We3, be3, We4, be4,
                  Wd1, bd1, Wd2, bd2, Wd3, bd3, eps, x)
    mu, log_sigma2 = encode(We1, be1, We2, be2, We3, be3, We4, be4, x)
    z = mu .+ sqrt.(exp.(log_sigma2)) .* eps
    x_rec = decode(Wd1, bd1, Wd2, bd2, Wd3, bd3, z)
    # loss
    rec_loss = -sum(x .* log.(1e-10 .+ x_rec) .+ (1 .- x) .* log.(1e-10 + 1.0 .- x_rec); dims=1)
    # KLD = -0.5 .* sum(1 .+ log_sigma2 .- mu .^ 2.0f0 - exp.(log_sigma2); dims=1)
    # cost = mean(rec_loss .+ KLD)
    cost = mean(rec_loss)
end


function benchmark_vae()
    n_inp, n_he1, n_he2, n_z, n_hd1, n_hd2, n_out = 784, 500, 500, 20, 500, 500, 784
    We1, be1, We2, be2, We3, be3, We4, be4 = (xavier_init(n_he1, n_inp),
                                              zeros(n_he1),
                                              xavier_init(n_he2, n_he1),
                                              zeros(n_he2),
                                              xavier_init(n_z, n_he2),
                                              zeros(n_z),
                                              xavier_init(n_z, n_he2),
                                              zeros(n_z))
    Wd1, bd1, Wd2, bd2, Wd3, bd3 = (xavier_init(n_hd1, n_z),
                                    zeros(n_hd1),
                                    xavier_init(n_hd2, n_hd1),
                                    zeros(n_hd2),
                                    xavier_init(n_out, n_hd2),
                                    zeros(n_out))
    eps = (rand(Normal(0, 1), size(We3, 1), 100))
    x = rand(784, 100)
    perf_test(vae_cost, We1, be1, We2, be2, We3, be3, We4, be4, Wd1, bd1, Wd2, bd2, Wd3, bd3, eps, x)
end
