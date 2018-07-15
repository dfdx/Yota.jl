## tape

mutable struct Tape <: AbstractTape
    ops::Vector{<:AbstractOp}   # linearized execution graph
    Tape() = new(AbstractOp[])
end

function Base.show(io::IO, tape::Tape)
    println(io, "Tape")
    for op in tape.ops
        println(io, "  $op")
    end
end

Base.getindex(tape::Tape, i...) = getindex(tape.ops, i...)
Base.lastindex(tape::Tape) = lastindex(tape.ops)
Base.length(tape::Tape) = length(tape.ops)


"""
Record operation to a tape, set its var's .id and .data.
Return op's variable.
"""
function record!(tape::Tape, op::AbstractOp)
    push!(tape.ops, op)
    op.var.id = length(tape)
    return exec!(tape, op)
end
