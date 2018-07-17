## operations on Tape

## Input

struct Input <: AbstractOp
    var::TAny
end

@inline exec!(tape::AbstractTape, op::Input) = op.var


## Constant

struct Constant <: AbstractOp
    var::TAny
end

@inline exec!(tape::AbstractTape, op::Constant) = op.var


## Call

"""
Method call
"""
struct Call{Fn, ARGS <: Tuple} <: AbstractOp
    var::TAny                  # tracked var, result of the call
    fn::Fn                     # function to apply to args
    args::ARGS                 # call arguments
    kwargs::Dict{Symbol, Any}  # keyword arguments
    Call(var::TAny, fn::Fn, args::ARGS; kwargs=Dict()) where {Fn, ARGS} =
        new{Fn, ARGS}(var, fn, args, kwargs)
end

function Base.show(io::IO, op::Call)
    args_str = join(["%$(var.id)" for var in op.args], ", ")
    kwargs_str = isempty(op.kwargs) ? "" : "; " * join(["$k=$v" for (k, v) in op.kwargs], ", ")
    print(io, "Call(%$(op.var.id) = $(op.fn)($(args_str)$kwargs_str))")
end


"""
Execute operation on a tape, store result to op's var.
"""
function exec!(tape::AbstractTape, op::Call)
    arg_data = map(getvalue, op.args)
    op.var.val = op.fn(arg_data...; op.kwargs...)
    return op.var
end


## Assign

struct Assign <: AbstractOp
    var::TAny
    src::TAny
end

Base.show(io::IO, op::Assign) = print("Assign(%$(op.var.id) ← %$(op.src.id))")

function exec!(tape::AbstractTape, op::Assign)
    op.var.val = op.src.val
    return op.var
end


## AddGrad

# """
# Addition of calculated gradient (stored in .data field of grad_var)
# to .grad field of var
# """
# struct AddGrad <: AbstractOp
#     var::TAny
#     grad_var::TAny
# end

# Base.show(io::IO, op::AddGrad) = print(io, "AddGrad(%$(op.var.id) ← %$(op.grad_var.id))")

# function exec!(tape::AbstractTape, op::AddGrad)
#     op.var.grad += op.grad_var.data
# end
