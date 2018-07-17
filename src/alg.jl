import Base: *, /, +, -
import LinearAlgebra: mul!


## REAL

*(x::TReal, y::TReal) = record!(x.tape, Call, *, (x, y))
/(x::TReal, y::TReal) = record!(x.tape, Call, /, (x, y))
+(x::TReal, y::TReal) = record!(x.tape, Call, +, (x, y))
-(x::TReal, y::TReal) = record!(x.tape, Call, -, (x, y))


## ARRAYS

*(x::TArray, y::TArray) = record!(x.tape, Call, *, (x, y))



# function LinearAlgebra.mul!(C::TArray, A::TArray, B::TArray)
#     var = tracked(C.tape)
#     mul!(C.val, A.val, B.val)
#     nd = ExNode{:call}(C.var, :($(A.var) * $(B.var)); val=C.val)
#     push!(A.graph, nd)
#     return TArray(C.var, C.val)
# end
