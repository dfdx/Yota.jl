## data types to track execution and write operations to a tape

# we need AbstractTape as a way to make forward declaration for real Tape
abstract type AbstractTape end
abstract type AbstractOp end


mutable struct TReal <: Real
    tape::AbstractTape
    id::Int  # ID of cooresponding operation on the tape
    data::Real
    grad::Real
end

Base.show(io::IO, x::TReal) = print(io, "%$(x.id) = $(x.data)")


const TAny = Union{TReal}

tracked(tape::AbstractTape, x::Real) = TReal(tape, -1, x, zero(x))
