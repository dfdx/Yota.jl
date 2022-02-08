# not the most robust function, but works in practise
is_kwfunc(f) = (name = string(f); endswith(name, "##kw") || endswith(name, "##kw\""))
is_kwfunc(v::Variable) = is_kwfunc(v._op.val)
