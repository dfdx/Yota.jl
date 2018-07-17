import Base: *, /, +, -
import LinearAlgebra: mul!

    

function *(x::TReal, y::TReal)
    # TODO: don't calculate it twice (record! always executes)
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

function -(x::TReal)
    var = tracked(tape, -1)
    return record!(tape, Call(var, -, (x,)))
end



## ARRAYS

function *(x::TArray, y::TArray)
    var = zero(x)    
    record!(tape, Call(var, *, (x, y)))
    return var
end


# function LinearAlgebra.mul!(C::TArray, A::TArray, B::TArray)
#     var = tracked(C.tape)
#     mul!(C.val, A.val, B.val)
#     nd = ExNode{:call}(C.var, :($(A.var) * $(B.var)); val=C.val)
#     push!(A.graph, nd)
#     return TArray(C.var, C.val)
# end
