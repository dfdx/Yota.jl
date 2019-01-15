########################################################################
#                            OPERATIONS                                #
########################################################################

abstract type AbstractOp end

Base.getproperty(op::AbstractOp, f::Symbol) = f == :typ ? typeof(op.val) : getfield(op, f)

## Input

struct Input <: AbstractOp
    id::Int
    val::Any
end

Base.show(io::IO, op::Input) = print(io, "inp %$(op.id)::$(op.typ)")


## Constant

struct Constant <: AbstractOp
    id::Int
    typ::Type
    val
end


Constant(id::Int, val) = Constant(id, typeof(val), val)
Base.show(io::IO, op::Constant) = print(io, "const %$(op.id) = $(op.val)::$(op.typ)")


## Assign

struct Assign <: AbstractOp
    id::Int
    src_id::Int
    val
end

Base.show(io::IO, op::Assign) = print(io, "%$(op.id) = %$(op.src_id)::$(op.typ)")


## Call

struct Call <: AbstractOp
    id::Int
    val::Any
    fn::Function
    args::Vector{Int}
    kwargs::Dict          # currently not used
end

Call(id::Int, val::Any, fn::Function, args::Vector{Int}; kwargs=Dict()) =
    Call(id, val, fn, args, kwargs)
Base.getproperty(op::Input, f::Call) = f == :typ ? typeof(op.val) : getfield(op, f)


function Base.show(io::IO, op::Call)
    arg_str = join(["%$(id)" for id in op.args], ", ")
    kwarg_str = (isempty(op.kwargs) ? "" : "; " *
                 join(["$k=$v" for (k, v) in op.kwargs], ", "))
    print(io, "%$(op.id) = $(op.fn)($arg_str$kwarg_str)::$(op.typ)")
end


"""
Copy struct `x` replacing specified fields with new values
"""
function copy_with(x; kvs...)
    T = typeof(x)
    d = Dict(kvs)
    flds = fieldnames(T)
    new_flds = [get(d, f, getfield(x, f)) for f in flds]
    return T(new_flds...)
end


########################################################################
#                                 TAPE                                 #
########################################################################

const MaybeFunction = Union{Function, Nothing}

mutable struct Tape
    ops::Vector{<:AbstractOp}   # linearized execution graph
    resultid::Int               # id of result variable
    derivs::Dict{Int,Int}       # derivs[var_id] == grad_id
    # sfields::Dict{Int, Dict}    # mapping of argid -> Dict(struct field paths -> var id)
    compiled::MaybeFunction     # compiled tape or nothing
    device::AbstractDevice      # function to use for moving intermediate results to device
end

Tape(device::AbstractDevice) = Tape(AbstractOp[], -1, Dict(), nothing, device)
Tape() = Tape(CPU())


function Base.show(io::IO, tape::Tape)
    println(io, "Tape")
    for op in tape.ops
        println(io, "  $op")
    end
end

Base.getindex(tape::Tape, i::Integer...) = getindex(tape.ops, i...)
# Base.getindex(tape::Tape, i::String...) =
#     getindex(tape.ops, [parse(Int, s[2:end]) for s in i]...)
# Base.getindex(tape::Tape, i::Symbol...) = getindex(tape, map(string, i)...)
Base.setindex!(tape::Tape, op::AbstractOp, i::Integer) = (tape.ops[i] = op)
Base.lastindex(tape::Tape) = lastindex(tape.ops)
Base.length(tape::Tape) = length(tape.ops)
Base.iterate(tape::Tape) = iterate(tape.ops)
Base.iterate(tape::Tape, s) = iterate(tape.ops, s)
Base.push!(tape::Tape, op::AbstractOp) = push!(tape.ops, op)


"""
Record a new operation (defined by its type and arguments for constructor)
to a tape and return ID of this new op.
"""
function record!(tape::Tape, optype::Type{<:AbstractOp}, args...)
    ret_id = length(tape) + 1
    push!(tape, optype(ret_id, args...))
    return ret_id
end


"""
Parse a complex call expression and record corresponding operations to a tape.

Optionally takes substitution table (st parameter) to replace known symbols with
provided values.
"""
function record_expr!(tape::Tape, ex::Expr; st=Dict(), bcast=false)
    @assert Meta.isexpr(ex, :call) "Expression isn't a call"
    # TODO: handle kw args
    new_op_args = Vector{Int}(undef, length(ex.args) - 1)
    for (i, x) in enumerate(ex.args[2:end])
        if haskey(st, x)
            # op ID comes from substitution table
            new_op_args[i] = st[x]
        elseif Meta.isexpr(x, :call)
            # recursively record arg expression
            arg_id = record_expr!(tape, x; st=st)
            new_op_args[i] = arg_id
        else
            # treat as constant
            arg_id = record!(tape, Constant, x)
            new_op_args[i] = arg_id
        end
    end
    fn = ex.args[1]
    if !bcast
        retval = fn([tape[id].val for id in new_op_args]...)
        return record!(tape, Call, retval, fn, new_op_args)
    else
        retval = broadcast(fn, [tape[id].val for id in new_op_args]...)
        fn_id = record!(tape, Constant, fn)
        return record!(tape, Call, retval, broadcast, [fn_id; new_op_args])
    end
end


function record_expr!(tape::Tape, x::Symbol; st=Dict(), bcast=false)
    ds_id = st[:ds]
    return record!(tape, Assign, ds_id, tape[ds_id].val)
end


function record_expr!(tape::Tape, x; st, bcast=false)
    return record!(tape, Constant, x)
end




########################################################################
#                           TRANSFORMATIONS                            #
########################################################################


"""
Replace all call arguments according to the provided dict
"""
function replace_in_args!(tape::Tape, st::Dict)
    for (i, op) in enumerate(tape)
        if op isa Call
            new_args = [get(st, x, x) for x in op.args]
            new_kwargs = Dict(k => get(st, x, x) for (k, x) in op.kwargs)
            tape[i] = copy_with(op, args=new_args, kwargs=new_kwargs)
        end
    end
end


"""
Replace a sequence of `broadcasted()` => `materizalize()` calls with a single `broadcast()`
"""
function squash_broadcast(tape::Tape)
    new_tape = copy_with(tape; ops=AbstractOp[])
    st = Dict()
    for (id, op) in enumerate(tape)
        if op isa Call && op.fn === Broadcast.broadcasted
            # replace Broadcast.broadcasted with just broadcast & materialize value
            val = Broadcast.materialize(op.val)
            new_id = length(new_tape) + 1
            push!(new_tape.ops, copy_with(op; fn=broadcast, val=val))
            st[id] = length(new_tape)
        elseif op isa Call && op.fn === Broadcast.materialize
            # materialize becomes just assignment
            new_id = record!(new_tape, Assign, op.args[1], op.val)
            st[id] = new_id
        else
            # record any other operations as is
            new_id = length(new_tape) + 1
            push!(new_tape.ops, copy_with(op; id=new_id))
            st[id] = length(new_tape)
        end
    end
    replace_in_args!(new_tape, st)
    new_tape.resultid = get(st, tape.resultid, tape.resultid)
    return new_tape
end



function simplify(tape::Tape)
    return squash_broadcast(tape)
end


########################################################################
#                              EXECUTION                               #
########################################################################

function play!(tape::Tape)
    vals = Vector{Any}(undef, length(tape))
    for (i, op) in enumerate(tape)
        vals[i] = exec(vals, op)
    end
    return vals[tape.resultid]
end

exec(vals::Vector, op::Input) = op.val
exec(vals::Vector, op::Constant) = op.val
exec(vals::Vector, op::Assign) = vals[op.src_id]
exec(vals::Vector, op::Call) = op.fn([vals[id] for id in op.args]...)
