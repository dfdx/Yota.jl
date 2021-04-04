################################################################################
#                                 MvNormal                                     #
################################################################################

# gradients verified empirically using PyTorch version

function ∇logpdf_mu(dy, d::MvNormal, x)
    # device = iscuarray(x) ? 
    iΣ = inv(d.Σ)
    iΣf = Matrix(iΣ)  # TODO: make device agnostic
    return dy .* iΣf * (x .- d.μ)
end


function ∇logpdf_sigma(dy, d::MvNormal, x)
    iΣ = inv(d.Σ)
    iΣf = Matrix(iΣ)  # TODO: make device agnostic
    diff = x .- d.μ
    return -0.5f0 .* (iΣf .- iΣ * diff * diff' * iΣf) .* dy
end


function ∇logpdf_x(dy::AbstractVector, d::MvNormal, x::AbstractMatrix)
    ret = similar(x)
    for j in 1:size(x, 2)
        ret[:, j] = Distributions.gradlogpdf(d, @view x[:, j]) .* dy[j]
    end
    return ret
end


function ∇logpdf_x(dy::Real, d::MvNormal, x::AbstractVector)
    return Distributions.gradlogpdf(d, x) .* dy
end


@ctor MvNormal(μ, Σ)
@diffrule logpdf(_d::MvNormal, x) _d.μ ∇logpdf_mu(dy, _d, x)
@diffrule logpdf(_d::MvNormal, x) _d.Σ ∇logpdf_sigma(dy, _d, x)
@diffrule logpdf(_d::MvNormal, x) x ∇logpdf_x(dy, _d, x)
