import Yota
import Zygote
using BenchmarkTools

include("functions.jl")


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


# benchmarks

function benchmark_autoencoder()
    f = autoencoder_cost
    println("\n## On larger data\n")
    We1 = rand(2000, 10_000); b1 = rand(2000); We2 = rand(1000, 2000); b2 = rand(1000);
    Wd = rand(10_000, 1000); x = rand(10_000, 100);
    args = (We1, We2, Wd, b1, b2, x)
    perf_test(f, args...)

    println("\n## On smaller data\n")
    We1 = rand(200, 1000); b1 = rand(200); We2 = rand(100, 200); b2 = rand(100);
    Wd = rand(1000, 100); x = rand(1000, 100);
    args = (We1, We2, Wd, b1, b2, x)
    perf_test(f, args...)
end


function benchmark_mlp1()
    f = mlp1
    println("\n## On larger data\n")
    w1=rand(2000, 10000); w2=rand(1000, 2000); w3=rand(1000, 1000); x1=rand(10000, 500);
    args = (w1, w2, w3, x1)
    perf_test(f, args...)

    println("\n## On smaller data\n")
    w1=rand(200, 1000); w2=rand(100, 200); w3=rand(100, 100); x1=rand(1000, 10);
    args = (w1, w2, w3, x1)
    perf_test(f, args...)
end
