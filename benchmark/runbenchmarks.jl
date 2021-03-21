# This file was adapted from Transducers.jl
# which is available under an MIT license (see LICENSE).
using PkgBenchmark
include("pprinthelper.jl")


mkconfig(; kwargs...) =
    BenchmarkConfig(
        env = Dict(
            "JULIA_NUM_THREADS" => "1",
        );
        kwargs...
    )

group_target = benchmarkpkg(
    dirname(@__DIR__),
    mkconfig(),
)

group_baseline = benchmarkpkg(
    dirname(@__DIR__),
    mkconfig(id = "baseline"),
)

judgement = judge(group_target, group_baseline)

displayresult(judgement)

printnewsection("Target result")
displayresult(group_target)

printnewsection("Baseline result")
displayresult(group_baseline)