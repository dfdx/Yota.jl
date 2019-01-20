# linear algebra matrix types
import Base.*

const Untracked = Input

# the following types have immutable data, but aren't constants
# so we create Input op for them, but don't assign argid

# WARNING: highly experimental, only partially implemented!

# function *(A::UpperTriangular{T1,A}, B::TArray{T2,1}) where {T1,T2,A<:TArray}
#     println("oooo!")
#     tA = record!(B.tape, Untracked, to_device(B.tape.device, A))
#     return tA * B
# end


# UpperTriangular(x::TArray) = record!()

# when this conversion happens?
# 1. in record_struct!
# 2. as part of subroutine -> record!(tape, Call, UpperTriangular, (x,))


# tA = record!(tape, Input, UpperTriangular(rand(3,3)))
# tA * B
# ^ works fine


# UpperTriangular{T,TArray{T,N}} -> untrack -> wrap into UpperTriangular -> track result
# record!(tape, Call, *, (A, tB))
