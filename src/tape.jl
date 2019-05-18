########################################################################
#                            OPERATIONS                                #
########################################################################

abstract type AbstractOp end

Base.getproperty(op::AbstractOp, f::Symbol) = f == :typ ? typeof(op.val) : getfield(op, f)

## Input

mutable struct Input <: AbstractOp
    id::Int
    val::Any
end

Base.show(io::IO, op::Input) = print(io, "inp %$(op.id)::$(op.typ)")


## Constant

mutable struct Constant <: AbstractOp
    id::Int
    typ::Type
    val
end


Constant(id::Int, val) = Constant(id, typeof(val), val)
Base.show(io::IO, op::Constant) = print(io, "const %$(op.id) = $(op.val)::$(op.typ)")


## Assign

mutable struct Assign <: AbstractOp
    id::Int
    src_id::Int
    val
end

Base.show(io::IO, op::Assign) = print(io, "%$(op.id) = %$(op.src_id)::$(op.typ)")


## Call

mutable struct Call <: AbstractOp
    id::Int
    val::Any
    fn::Union{Function, Type}
    args::Vector{Int}
    kwargs::Dict          # currently not used
end

Call(id::Int, val::Any, fn::Union{Function, Type}, args::Vector{Int}; kwargs=Dict()) =
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
    # linearized execution graph
    ops::Vector{<:AbstractOp}
    # id of result variable
    resultid::Int
    # derivs[var_id] == grad_id
    derivs::Dict{Int,Int}
    # mapping of argid -> Dict(struct field paths -> var id)
    fieldpaths::Dict{Int, Dict}
    # compiled tape or nothing
    compiled::MaybeFunction
    # device of the tape
    device::AbstractDevice
end

Tape(device::AbstractDevice) = Tape(AbstractOp[], -1, Dict(), Dict(), nothing, device)
Tape() = Tape(CPU())
Base.similar(tape::Tape) = Tape(AbstractOp[], tape.resultid, tape.derivs,
                                tape.fieldpaths, tape.compiled, tape.device)


function Base.show(io::IO, tape::Tape)
    println(io, "Tape")
    for op in tape.ops
        println(io, "  $op")
    end
end

Base.getindex(tape::Tape, i...) = getindex(tape.ops, i...)
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

Keyword params:

 * st::Dict - substitution table, used to replace symbols in ex with tape op ids
 * bcast::Bool - replace all function calls with corresponding broadcasting
"""
function record_expr!(tape::Tape, ex::Expr; st=Dict(), bcast=false)
    @assert Meta.isexpr(ex, :call) "Expression isn't a call"
    new_op_args = Vector{Int}(undef, length(ex.args) - 1)
    for (i, x) in enumerate(ex.args[2:end])
        if haskey(st, x)
            # op ID comes from substitution table
            new_op_args[i] = st[x]
        elseif Meta.isexpr(x, :call)
            # recursively record arg expression
            arg_id = record_expr!(tape, x; st=st, bcast=bcast)
            new_op_args[i] = arg_id
        else
            # treat as constant
            arg_id = record!(tape, Constant, x)
            new_op_args[i] = arg_id
        end
    end    
    fn = ex.args[1]
    fn = device_function(tape.device, fn)
    if bcast
        retval = broadcast(fn, [tape[id].val for id in new_op_args]...)
        fn_id = record!(tape, Constant, fn)
        return record!(tape, Call, retval, broadcast, [fn_id; new_op_args])
    elseif fn isa Symbol && fn in Set([:.+, :.*, :.-, :./, :.^])
        _fn = eval(Symbol(string(fn)[2:end]))
        retval = broadcast(_fn, [tape[id].val for id in new_op_args]...)
        fn_id = record!(tape, Constant, _fn)
        return record!(tape, Call, retval, broadcast, [fn_id; new_op_args])
    else
        retval = fn([tape[id].val for id in new_op_args]...)
        return record!(tape, Call, retval, fn, new_op_args)
    end
end


function record_expr!(tape::Tape, x::Symbol; st=Dict(), bcast=false)
    id = st[x]
    return record!(tape, Assign, id, tape[id].val)
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
Recover broadcast operation from Broadcast.broadcasted and Broadcast.materialize
"""
function recover_broadcast(tape::Tape)
    new_tape = copy_with(tape; ops=AbstractOp[])
    # TODO: seems like we don't need subs table any more
    # remove after squash_assigned is implemented
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


function squash_assigned(tape::Tape)
    tape = copy_with(tape; ops=deepcopy(tape.ops))
    # 1. compute subs table for chains of assignment operations
    # and replace equivalent op ids
    assign_st = Dict()
    for (id, op) in enumerate(tape)
        if op isa Assign
            # propagate replacement through known st
            src_id = op.src_id
            while haskey(assign_st, src_id)
                src_id = assign_st[src_id]
            end
            assign_st[id] = src_id
        end
    end
    replace_in_args!(tape, assign_st)
    tape.resultid = get(assign_st, tape.resultid, tape.resultid)
    # 2. create new ops w/o assignments
    new_tape = copy_with(tape; ops=AbstractOp[])
    reindex_st = Dict()
    for (id, op) in enumerate(tape)
        if !isa(op, Assign)
            # record any other operations as is
            new_id = length(new_tape) + 1
            push!(new_tape.ops, copy_with(op; id=new_id))
            if id != new_id
                reindex_st[id] = new_id
            end
        end
    end
    replace_in_args!(new_tape, reindex_st)
    new_tape.resultid = get(reindex_st, tape.resultid, tape.resultid)
    return new_tape
end


function simplify(tape::Tape)
    tape = recover_broadcast(tape)
    tape = squash_assigned(tape)
    tape = remove_unused(tape)
    return tape
end


########################################################################
#                              EXECUTION                               #
########################################################################


exec!(tape::Tape, op::Input) = ()
exec!(tape::Tape, op::Constant) = ()
exec!(tape::Tape, op::Assign) = (op.val = tape[op.src_id].val)
exec!(tape::Tape, op::Call) = (op.val = op.fn([tape[id].val for id in op.args]...))


function play!(tape::Tape, args...; use_compiled=true)
    for (i, val) in enumerate(args)
        @assert(tape[i] isa Input, "More arguments than the original function had")
        tape[i].val = val
    end
    if use_compiled && tape.compiled != nothing
        Base.invokelatest(tape.compiled)
    else
        for op in tape
            exec!(tape, op)
        end
    end
    return tape[tape.resultid].val
end
