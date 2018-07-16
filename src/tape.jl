## tape

mutable struct Tape <: AbstractTape
    ops::Vector{<:AbstractOp}   # linearized execution graph
    derivs::Dict{Int,Int}       # derivs[var.id] == grad_var.id
    Tape() = new(AbstractOp[], Dict())
end

function Base.show(io::IO, tape::Tape)
    # println(io, "Tape")
    # for op in tape.ops
    #     println(io, "  $op")
    # end
    show_annotated(io, tape)
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


function show_annotated(io::IO, tape::Tape)
    rev_derivs = Dict((j, i) for (i, j) in tape.derivs)
    println(io, "Tape")
    for (i, op) in enumerate(tape.ops)
        hint = haskey(rev_derivs, i) ? "\t# deriv for %$(rev_derivs[i])" : ""
        println(io, "  $op$hint")
    end
end
