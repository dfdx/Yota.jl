# Copied from
# https://github.com/kul-forbes/ProximalOperators.jl/tree/master/benchmark
using ArgParse
using PkgBenchmark
using Markdown

function displayresult(result)
    md = sprint(export_markdown, result)
    md = replace(md, ":x:" => "❌")
    md = replace(md, ":white_check_mark:" => "✅")
    display(Markdown.parse(md))
end

function printnewsection(name)
    println()
    println()
    println()
    printstyled("▃" ^ displaysize(stdout)[2]; color=:blue)
    println()
    printstyled(name; bold=true)
    println()
    println()
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--target"
            help = "the branch/commit/tag to use as target"
            default = "HEAD"
        "--baseline"
            help = "the branch/commit/tag to use as baseline"
            default = "main"
        "--retune"
            help = "force re-tuning (ignore existing tuning data)"
            action = :store_true
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    mkconfig(; kwargs...) =
        BenchmarkConfig(
            env = Dict(
                "JULIA_NUM_THREADS" => "1",
            );
            kwargs...
        )

    target = parsed_args["target"]
    group_target = benchmarkpkg(
        dirname(@__DIR__),
        mkconfig(id = target),
        resultfile = joinpath(@__DIR__, "result-$(target).json"),
        retune = parsed_args["retune"],
    )

    baseline = parsed_args["baseline"]
    group_baseline = benchmarkpkg(
        dirname(@__DIR__),
        mkconfig(id = baseline),
        resultfile = joinpath(@__DIR__, "result-$(baseline).json"),
    )

    printnewsection("Target result")
    displayresult(group_target)

    printnewsection("Baseline result")
    displayresult(group_baseline)

    judgement = judge(group_target, group_baseline)

    printnewsection("Judgement result")
    displayresult(judgement)
end

main()



# # This file was adapted from Transducers.jl
# # which is available under an MIT license (see LICENSE).
# using PkgBenchmark
# include("pprinthelper.jl")


# mkconfig(; kwargs...) =
#     BenchmarkConfig(
#         env = Dict(
#             "JULIA_NUM_THREADS" => "1",
#         );
#         kwargs...
#     )

# group_target = benchmarkpkg(
#     dirname(@__DIR__),
#     mkconfig(),
# )

# # @show readdir("benchmark")
# # out = run(`git diff`)
# # @show read(out, String)
# # out = run(`cat benchmark/Project.toml`)
# # @show read(out, String)


# group_baseline = benchmarkpkg(
#     dirname(@__DIR__),
#     mkconfig(id = "baseline"),
# )

# judgement = judge(group_target, group_baseline)

# displayresult(judgement)

# printnewsection("Target result")
# displayresult(group_target)

# printnewsection("Baseline result")
# displayresult(group_baseline)