
using Yota
import Yota: TReal, TAny, record!, Call, grad!
using XGrad
using BenchmarkTools

include("functions.jl")

# xgrad derivatives
@diffrule logistic(x::Number) x (logistic(x) .* (1 .- logistic(x)) .* ds)


# yota wrappers and derivatives
logistic(x::TReal) = record!(x.tape, Call, logistic, (x,))

function grad!(dy::TAny, ::Val{1}, op::Call{typeof(logistic), Tuple{TReal}})
    x = op.args[1]
    ll = logistic(x)
    return (ll * (1 - ll) * dy)
end


# runner

function perf_test(f; ctx=Dict(), compile_tape=true, inputs...)
    vals = ([val for (name, val) in inputs]...,)

    println("Compiling derivatives using XGrad")
    @time df = xdiff(f; ctx=ctx, inputs...)
    mem = Dict()
    println("Testing XGrad...")
    r1 = @benchmark $df($vals...; mem=$mem)
    show(stdout, MIME{Symbol("text/plain")}(), r1)
    println("\n")

    println("Compiling derivatives using Yota")
    @time grad(f, vals...)
    r2 = @benchmark grad($f, $(vals)...)
    
    show(stdout, MIME{Symbol("text/plain")}(), r2)
    println("\n")
end


# benchmarks

function benchmark_autoencoder()
    f = autoencoder_cost
    println("\n## On larger data\n")
    We1 = rand(2000, 10_000); b1 = rand(2000); We2 = rand(1000, 2000); b2 = rand(1000);
    Wd = rand(10_000, 1000); x = rand(10_000, 100);
    inputs = [:We1 => We1, :We2 => We2, :Wd => Wd, :b1 => b1, :b2 => b2, :x => x];
    perf_test(f; inputs...)

    println("\n## On smaller data\n")
    We1 = rand(200, 1000); b1 = rand(200); We2 = rand(100, 200); b2 = rand(100);
    Wd = rand(1000, 100); x = rand(1000, 100);
    inputs = [:We1 => We1, :We2 => We2, :Wd => Wd, :b1 => b1, :b2 => b2, :x => x];
    perf_test(f; compile_tape=false, inputs...)
end


function benchmark_mlp1()
    f = mlp1
    println("\n## On larger data\n")
    w1=rand(2000, 10000); w2=rand(1000, 2000); w3=rand(1000, 1000); x1=rand(10000, 500);
    inputs = [:w1=>w1, :w2=>w2, :w3=>w3, :x1=>x1];
    perf_test(f; inputs...)

    println("\n## On smaller data\n")
    w1=rand(200, 1000); w2=rand(100, 200); w3=rand(100, 100); x1=rand(1000, 10);
    inputs = [:w1=>w1, :w2=>w2, :w3=>w3, :x1=>x1];
    perf_test(f; inputs...)
end
