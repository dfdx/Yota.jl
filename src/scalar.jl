
for fn in (*, /, +, -)
    fns = Symbol(fn)
    @eval $fns(x::TReal, y::TReal) = record!(x.tape, Call, $fn, (x, y))
end
    
-(x::TReal) = record!(x.tape, Call, -, (x,))

for fn in (sin, cos, log, exp, abs, abs2, sign, tanh)
    fns = Symbol(fn)
    @eval $fns(x::TReal) = record!(x.tape, Call, $fn, (x,))
end
# cos(x::TReal) = record!(x.tape, Call, cos, (x,))
# log(x::TReal) = record!(x.tape, Call, log, (x,))
# exp(x::TReal) = record!(x.tape, Call, exp, (x,))
# abs(x::TReal) = record!(x.tape, Call, abs, (x,))
# abs2(x::TReal) = record!(x.tape, Call, abs2, (x,))
# sign(x::TReal) = record(x.tape, Call, sign, (x,))
