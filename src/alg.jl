import Base: *, /, +, -


function *(x::TReal, y::TReal)
    var = tracked(x.tape, x.val * y.val)
    op = Call(var, *, (x, y))
    record!(tape, op)
    return var
end

function /(x::TReal, y::TReal)
    var = tracked(x.tape, x.val / y.val)
    op = Call(var, /, (x, y))
    record!(tape, op)
    return var
end

function +(x::TReal, y::TReal)
    var = tracked(x.tape, x.val + y.val)
    op = Call(var, +, (x, y))
    record!(tape, op)
    return var
end

function -(x::TReal, y::TReal)
    var = tracked(x.tape, x.val - y.val)
    op = Call(var, -, (x, y))
    record!(tape, op)
    return var
end
