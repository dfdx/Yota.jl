
for fn in (*, /, +, -, ^)
    fns = Symbol(fn)
    @eval $fns(x::TReal, y::TReal) = record!(x.tape, Call, $fn, (x, y))
    # @eval $fns(x::Real, y::TReal) = record!(y.tape, Call, $fn, (x, y))
    # @eval $fns(x::TReal, y::Real) = record!(x.tape, Call, $fn, (x, y))
    @eval $fns(x::TReal, y::Real) = $fn(x, constant(x.tape, y))
    @eval $fns(x::Real, y::TReal) = $fn(constant(y.tape, x), y)
end

^(x::TReal, p::Integer) = record!(x.tape, Call, ^, (x, constant(x.tape, p)))
-(x::TReal) = record!(x.tape, Call, -, (x,))

for fn in (sin, cos, log, exp, abs, abs2, sign, tanh, sqrt)
    fns = Symbol(fn)
    @eval $fns(x::TReal) = record!(x.tape, Call, $fn, (x,))
end
