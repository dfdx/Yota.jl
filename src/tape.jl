########################################################################
#                            OPERATIONS                                #
########################################################################

abstract type Op end


struct Input <: Op
    id::Int
    typ::Type
end

Base.show(io::IO, op::Input) = print(io, "inp %$(op.id)::$(op.typ)")

struct Constant <: Op
    id::Int
    typ::Type
    val
end

Constant(id::Int, val) = Constant(id, typeof(val), val)

Base.show(io::IO, op::Constant) = print(io, "const %$(op.id) = $(op.val)::$(op.typ)")


struct Call <: Op
    id::Int
    typ::Type
    fn::Function
    args::Vector{Int}
end

function Base.show(io::IO, op::Call)
    arg_str = join(["%$(id)" for id in op.args], ", ")
    print(io, "%$(op.id) = $(op.fn)($arg_str)::$(op.typ)")
end


########################################################################
#                                 TAPE                                 #
########################################################################

mutable struct Tape
    ops::Vector{<:Op}
end

Tape() = Tape(Op[])

function Base.show(io::IO, tape::Tape)
    println(io, "Tape")
    for op in tape.ops
        println(io, "  $op")
    end
end

Base.getindex(tape::Tape, i...) = tape.ops[i...]
Base.length(tape::Tape) = length(tape.ops)
Base.push!(tape::Tape, op::Op) = push!(tape.ops, op)

function record!(tape::Tape, optype::Type{<:Op}, args...)
    ret_id = length(tape) + 1
    push!(tape, optype(ret_id, args...))
    return ret_id
end
