using BenchmarkTools
using CUDA
using Yota

const SUITE = BenchmarkGroup()

#################################### TRACE ####################################

SUITE["trace"] = BenchmarkGroup()


# x = rand(8, 8)

# const TRACE_FUNCTIONS = [
#     "simple" => x -> sum(x),
#     "loop" => x -> (for _=1:100 x = x'x end; x),
# ]


# for (name, f) in TRACE_FUNCTIONS
#     SUITE["trace"][name] = @benchmarkable Yota.trace($f, $x) samples=10 seconds=60
# end


################################### HELPERS ###################################

# SUITE["helpers"] = BenchmarkGroup()

# x = rand(4, 5)
# dx = zero(x)
# dy = ones(4, 3)

# SUITE["helpers"]["ungetindex!"] = @benchmarkable Yota.ungetindex!($dx, $x, $dy, :, $[1, 3, 1])
# SUITE["helpers"]["ungetindex"] = @benchmarkable Yota.ungetindex($x, $dy, :, $[1, 3, 1])

# if CUDA.functional()
#     x = rand(4, 5) |> cu
#     dx = zero(x) |> cu
#     dy = ones(4, 3) |> cu

#     SUITE["helpers"]["cu:ungetindex!"] = @benchmarkable Yota.ungetindex!($dx, $x, $dy, :, $(cu([1, 3, 1])))
#     SUITE["helpers"]["cu:ungetindex"] = @benchmarkable Yota.ungetindex($x, $dy, :, $(cu([1, 3, 1])))
# end