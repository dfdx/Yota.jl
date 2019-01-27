using Zygote


function cost(m, x)
    sum(m.W * x .+ m.b)
end


function main()
    m = Linear(rand(3,4), rand(3))
    x = rand(4)
    cost(m, x)

    _, g = grad(cost, m, x)
    r = Zygote.gradient(cost, m, x)
end
