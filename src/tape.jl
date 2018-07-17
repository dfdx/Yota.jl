## tape

mutable struct Tape
    ops::Vector{<:AbstractOp}   # linearized execution graph
    derivs::Dict{Int,Int}       # derivs[var.id] == grad_var.id
    Tape() = new(AbstractOp[], Dict())
end

function Base.show(io::IO, tape::Tape)
    rev_derivs = Dict((j, i) for (i, j) in tape.derivs)
    println(io, "Tape")
    for (i, op) in enumerate(tape.ops)
        hint = haskey(rev_derivs, i) ? "\t# deriv for %$(rev_derivs[i])" : ""
        println(io, "  $op$hint")
    end
end

Base.getindex(tape::Tape, i...) = getindex(tape.ops, i...)
Base.lastindex(tape::Tape) = lastindex(tape.ops)
Base.length(tape::Tape) = length(tape.ops)


"""
Record an operation onto a tape, assign new ID to op's var.
"""
function _record!(tape::Tape, op::AbstractOp)
    push!(tape.ops, op)
    op.var.id = length(tape)
    nothing
end
