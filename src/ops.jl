## operations on Tape

## Input

struct Input <: AbstractOp
    var::TAny
    argid::Int
end

@inline exec!(tape::Tape, op::Input) = op.var

function record!(tape::Tape, ::Type{Input}, val; argid=-1)
    op = Input(tracked(tape, val), argid)
    push!(tape.ops, op)
    op.var.id = length(tape)
    return op.var
end


## Constant

struct Constant <: AbstractOp
    var::TAny
end

@inline exec!(tape::Tape, op::Constant) = op.var

function record!(tape::Tape, ::Type{Constant}, val)
    op = Constant(tracked(tape, val))
    push!(tape.ops, op)
    op.var.id = length(tape)
    return op.var
end


## Call

"""
Method call
"""
struct Call{Fn, ARGS <: Tuple} <: AbstractOp
    var::TAny                  # tracked var, result of the call
    fn::Fn                     # function to apply to args
    args::ARGS                 # call arguments
    kwargs::Dict{Symbol, Any}  # keyword arguments
    # Call(var::TAny, fn::Fn, args::ARGS; kwargs=Dict()) where {Fn, ARGS} =
    #     new{Fn, ARGS}(var, fn, args, kwargs)
end

function Base.show(io::IO, op::Call)
    args_str = join([var isa TAny ? "%$(var.id)" : var for var in op.args], ", ")
    kwargs_str = isempty(op.kwargs) ? "" : "; " * join(["$k=$v" for (k, v) in op.kwargs], ", ")
    print(io, "Call(%$(op.var.id) = $(op.fn)($(args_str)$kwargs_str))")
end

function record!(tape::Tape, ::Type{Call}, fn::Fn, args::ARGS;
                 kwargs=Dict{Symbol,Any}()) where {Fn, ARGS<:Tuple}
    arg_vals = map(getvalue, args)
    val = fn(arg_vals...; kwargs...)
    var = tracked(tape, val)
    op = Call(var, fn, args, kwargs)
    _record!(tape, op)
    return var
end

"""
Execute operation on a tape, store result to op's var.
"""
function exec!(tape::Tape, op::Call)
    arg_data = map(getvalue, op.args)
    op.var.val = op.fn(arg_data...; op.kwargs...)
    return op.var
end


## Bcast

"""
Broadcasting
"""
struct Bcast{Fn, ARGS <: Tuple} <: AbstractOp
    var::TAny
    fn::Fn
    args::ARGS
end

function Base.show(io::IO, op::Bcast)
    args_str = join([var isa TAny ? "%$(var.id)" : var for var in op.args], ", ")
    print(io, "Bcast(%$(op.var.id) = $(op.fn).($args_str))")
end

function record!(tape::Tape, ::Type{Bcast}, fn::Fn, args::ARGS) where {Fn, ARGS<:Tuple}
    arg_vals = map(getvalue, args)
    val = fn.(arg_vals...)
    var = tracked(tape, val)
    op = Bcast(var, fn, args)
    _record!(tape, op)
    return var
end

"""
Execute operation on a tape, store result to op's var.
"""
function exec!(tape::Tape, op::Bcast)
    arg_data = map(getvalue, op.args)
    op.var.val = op.fn.(arg_data...)
    return op.var
end


## Assign

struct Assign <: AbstractOp
    var::TAny
    src::TAny
end

Base.show(io::IO, op::Assign) = print("Assign(%$(op.var.id) ← %$(op.src.id))")

function exec!(tape::Tape, op::Assign)
    op.var.val = op.src.val
    return op.var
end


function record!(tape::Tape, ::Type{Assign}, src::TAny)
    var = tracked(tape, src.val)
    op = Assign(var, src)
    _record!(tape, op)
    exec!(tape, op)
    return var
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

# function exec!(tape::Tape, op::AddGrad)
#     op.var.grad += op.grad_var.data
# end


## mutable structs: writing all trackable fields to the tape

"""
Traverse mutable struct and write all trackable fields to the tape,
keeping mapping from field paths to tracked vars.
"""
function record_struct!(tape::Tape, s, input_idx::Int; field_path=[])
    for name in fieldnames(typeof(s))
        full_field_path = vcat(field_path, name)
        field = getfield(s, name)
        if (field isa Real && !isa(field, Bool)) || field isa AbstractArray
            var = record!(tape, Input, field)
            setfield!(s, name, var)
            # save mapping field_path → var
            if !haskey(tape.sfields, input_idx)
                tape.sfields[input_idx] = []
            end
            push!(tape.sfields[input_idx], (full_field_path, var.id))
        elseif isstruct(field)
            record_struct!(tape, field, input_idx; field_path=full_field_path)
        end
    end
end
