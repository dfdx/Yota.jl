########################################################################
#                 REINDEXING & OP BINARIZATION                         #
########################################################################


function reindex(op::Call, st::Dict)
    new_args = [get(st, x, x) for x in op.args]
    new_id = get(st, op.id, op.id)
    return copy_with(op, args=new_args, id=new_id)
end

reindex(op::AbstractOp, st::Dict) = copy_with(op, id=get(st, op.id, op.id))


function reindex_fields!(tape::Tape, st::Dict)
    tape.resultid = get(st, tape.resultid, tape.resultid)
    # possibly we also need to reindex .derivs
end


"""
Transform tape so that all calls to *, /, + and - with multiple arguments were split into
binary form, e.g. transform:

*(a, b, c, d)

to

w1 = a * b
w2 = w1 * c
w3 = w2 * d
"""
function binarize_ops(tape::Tape)
    st = Dict()
    new_tape = copy_with(tape; ops=AbstractOp[])
    for (id, op) in enumerate(tape)
        # id == 11 && break
        # println(id)
        new_id = -1 # to be modified in both branches
        _op = reindex(op, st)   # we need both - original op and reindexed _op
        if op isa Call && op.fn in (+, -, *, /) && length(op.args) > 2
            # record first pair of arguments
            arg1_val, arg2_val = tape[op.args[1]].val, tape[op.args[2]].val
            new_val = op.fn(arg1_val, arg2_val)
            new_id = record!(new_tape, Call, new_val, op.fn, _op.args[1:2])
            # record the rest
            for i=3:length(op.args)
                # calculate new value based on current value and value of original ith argument
                new_val = op.fn(new_val, tape[op.args[i]].val)
                # record new binary op with the previous new_id and new argument id
                new_id = record!(new_tape, Call, new_val, op.fn, [new_id, _op.args[i]])
            end
            # st[id] = new_id
        else
            new_id = length(new_tape) + 1
            push!(new_tape, copy_with(_op; id=new_id))
        end
        if id != new_id
            st[id] = new_id
        end
    end
    reindex_fields!(new_tape, st)
    return new_tape
end


########################################################################
#                       TAPE TRANSFORMATIONS                           #
########################################################################

function find_deps(tape::Tape, ids::Vector{Int}; result=Set{Int}())
    ids = filter(id -> id != -1, ids)
    for id in ids
        if tape[id] isa Call
            args = tape[id].args
            push!(result, args...)
            find_deps(tape, args; result=result)
        end
    end
    return result
end

find_deps(tape::Tape, id::Int; result=Set{Int}()) = find_deps(tape, [id]; result=result)


function remove_unused(tape::Tape; keep=[tape.resultid])
    deps = find_deps(tape, keep); push!(deps, keep...)
    st = Dict()
    new_tape = copy_with(tape; ops=AbstractOp[])
    for (id, op) in enumerate(tape)
        if id in deps
            new_id = length(new_tape) + 1
            if id != new_id
                st[id] = new_id
            end
            new_op = reindex(op, st)
            push!(new_tape, new_op)
        end
    end
    reindex_fields!(new_tape, st)
    return new_tape
end


########################################################################
#                       PRE- & POST-PROCESSING                         #
########################################################################

"""
Apply a number of transformations to a tape after tracing and before calculating derivatives
"""
function preprocess(tape::Tape)
    tape = binarize_ops(tape)
    tape = remove_unused(tape)
    return tape
end


"""
Apply a number of transformations after calculating derivatives
"""
function postprocess(tape::Tape)
    return tape
end


########################################################################
#                          HASH & EQUALITY                             #
########################################################################

# note: hash and == don't use operation values and .compiled field
# so different runs of a tape doesn't affect its identity

Base.hash(op::Input, h::UInt) = hash(op.id, h)
Base.hash(op::Constant, h::UInt) = hash(op.id, hash(op.val, h))
Base.hash(op::Assign, h::UInt) = hash(op.id, hash(op.src_id, h))
Base.hash(op::Call, h::UInt) = hash(op.id, hash(op.fn, hash(op.args, h)))


function Base.hash(tape::Tape, h::UInt)
    h = hash(tape.resultid)
    h = hash(tape.derivs, h)
    h = hash(tape.fieldpaths, h)
    h = hash(tape.device, h)
    for op in tape
        h = hash(op, h)
    end
    return h
end


Base.:(==)(op1::Input, op2::Input) = (op1.id == op2.id)
Base.:(==)(op1::Constant, op2::Constant) = (op1.id == op2.id && op1.val == op2.val)
Base.:(==)(op1::Assign, op2::Assign) = (op1.id == op2.id && op1.src_id == op2.src_id)
Base.:(==)(op1::Call, op2::Call) = (op1.id == op2.id && op1.fn == op2.fn && op1.args == op2.args)

function Base.:(==)(tape1::Tape, tape2::Tape)
    return tape1.resultid == tape2.resultid &&
        tape1.derivs == tape2.derivs &&
        tape1.fieldpaths == tape2.fieldpaths &&
        tape1.device == tape2.device &&
        tape1.ops == tape2.ops
end


########################################################################
#                       __NEW__() & __GETPROPERTY__()                  #
########################################################################


function field_var_from_ctor_op(tape::Tape, ctor::Call, getprop_op::Call)
    @assert ctor.fn == __new__
    @assert getprop_op.fn == Base.getproperty
    T = typeof(ctor.val)
    flds = fieldnames(T)
    fld = tape[getprop_op.args[2]].val
    fld_arg_idx = findfirst(x -> x == fld, flds) + 1
    return tape[ctor.args[fld_arg_idx]]
end


"""
Given a tape and getproperty() operation, try to find a variable
that was used to create that field
"""
function find_field_source_var(tape::Tape, getprop_op::Call)
    parent = tape[getprop_op.args[1]]
    if parent isa Call && parent.fn == __new__
        # found constructor for this field, return variable that made up getprop_op's field
        return field_var_from_ctor_op(tape, parent, getprop_op)
    elseif parent isa Call && parent.fn == Base.getproperty
        # nested getproperty, find constructor for the current struct recursively
        ctor = find_field_source_var(tape, parent)
        if ctor != nothing
            return field_var_from_ctor_op(tape, ctor, getprop_op)
        else
            return nothing
        end
    else
        # can't find source field - give up and return nothing
        return nothing
    end
end


"""
Given a tape and getfield() operation, try to find a variable
that was used to create that field.
NOTE: This works only with getfield() on tuples
"""
function find_tuple_field_source_var(tape::Tape, getf_op::Call)
    ind_it_op = tape[getf_op.args[1]]
    @assert ind_it_op isa Call && ind_it_op.fn == Base.indexed_iterate
    tuple_op = tape[ind_it_op.args[1]]
    @assert tuple_op isa Call && tuple_op.fn == __tuple__
    tuple_idx = tape[ind_it_op.args[2]].val
    src_var = tape[tuple_op.args[tuple_idx]]
    return src_var
end
