## data types to track execution and write operations to a tape

# we need AbstractTape as a way to make forward declaration for real Tape
abstract type AbstractTape end
abstract type AbstractOp end


mutable struct TReal <: Real
    tape::AbstractTape
    id::Int             # ID of cooresponding operation on the tape
    val::Real
    cont::Any           # container from which this var was taken from (default: nothing)
    field::Any          # field name or index in container (default: nothing)
end

Base.show(io::IO, x::TReal) = print(io, "%$(x.id) = $(x.val)")
tracked(tape::AbstractTape, x::Real; cont=nothing, field=nothing) =
    TReal(tape, -1, x, cont, field)
Base.zero(x::TReal) = tracked(x.tape, zero(x.val))
getvalue(x::TReal) = x.val





const TAny = Union{TReal}

getvalue(x) = x
