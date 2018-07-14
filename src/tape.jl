import Base: *, /, +, -


# we need AbstractTape as a way to make forward declaration for real Tape
abstract type AbstractTape end
abstract type AbstractOp end


mutable struct TReal <: Real
    tape::AbstractTape
    id::Int               # ID of cooresponding
    val::Real
end

Base.show(io::IO, x::TReal) = print(io, "%$(x.id) = $(x.val)")


const TAny = Union{TReal}


struct Input <: AbstractOp
    var::TAny
end

struct Call <: AbstractOp
    var::TAny
    fn::Function
    args::Tuple
    kwargs::Dict{Symbol, Any}
    Call(var::TAny, fn::Function, args...; kwargs...) = new(var, fn, args, kwargs)
end

function Base.show(io::IO, op::Call)
    args_str = join(["%$(var.id)" for var in op.args], ", ")
    kwargs_str = isempty(op.kwargs) ? "" : "; " * join(["$k=$v" for (k, v) in op.kwargs], ", ")
    print(io, "Call(%$(op.var.id) = $(op.fn)($(args_str)$kwargs_str))")
end


mutable struct Tape <: AbstractTape
    ops::Vector{<:AbstractOp}
    Tape() = new(AbstractOp[])
end

function Base.show(io::IO, tape::Tape)
    println(io, "Tape")
    for op in tape.ops
        println(io, "  $op")
    end
end

Base.length(tape::Tape) = length(tape.ops)

# Call(var, fn, args, kwargs, val)


tracked(tape::Tape, x::Real) = TReal(tape, -1, x)





function *(x::TReal, y::TReal)
    var = tracked(x.tape, x.val * y.val)
    op = Call(var, *, x, y)
    record!(tape, op)
    return var
end

function /(x::TReal, y::TReal)
    var = tracked(x.tape, x.val / y.val)
    op = Call(var, /, x, y)
    record!(tape, op)
    return var
end

function +(x::TReal, y::TReal)
    var = tracked(x.tape, x.val + y.val)
    op = Call(var, +, x, y)
    record!(tape, op)
    return var
end

function -(x::TReal, y::TReal)
    var = tracked(x.tape, x.val - y.val)
    op = Call(var, -, x, y)
    record!(tape, op)
    return var
end





function record!(tape::Tape, op::AbstractOp)
    push!(tape.ops, op)
    op.var.id = length(tape)
    nothing
end


function main_1715()
    tape = Tape()
    x = TReal(tape, -1, 2.0)
    y = TReal(tape, -1, 3.0)
    record!(tape, Input(x))
    record!(tape, Input(y))
    z = x * y + y - x

end
