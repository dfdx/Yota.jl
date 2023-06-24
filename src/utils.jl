# not the most robust function, but works in practise
if VERSION < v"1.9.0"
    is_kwfunc(f) = (name = string(f); endswith(name, "##kw") || endswith(name, "##kw\""))
else
    is_kwfunc(f) = (f === Core.kwcall)
end
is_kwfunc(v::Variable) = is_kwfunc(v._op.val)

function unkwfunc(f, args...)
    @assert is_kwfunc(f) "Trying to undo Core.kwfunc() on f, but f is not a kw func"
    nokw_f = args[2]
    @assert Core.kwfunc(nokw_f) === f
    return nokw_f
end


Base.names(_::Tangent{NamedTuple{F, TT}}) where {F, TT} = F


iszerotangent(x::ZeroTangent) = true
iszerotangent(x::NoTangent) = true
iszerotangent(x) = false


# REPL utils - unstable API! don't use in library code!
Base.:(//)(tape::Tape, i::Integer) = tape[V(i)]
Base.:(:)(tape::Tape, i::Integer) = tape[V(i)].val
