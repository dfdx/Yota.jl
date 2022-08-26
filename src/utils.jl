# not the most robust function, but works in practise
is_kwfunc(f) = (name = string(f); endswith(name, "##kw") || endswith(name, "##kw\""))
is_kwfunc(v::Variable) = is_kwfunc(v._op.val)

# REPL utils - unstable API! don't use in library code!
Base.:(^)(tape::Tape, i::Integer) = tape[V(i)].val