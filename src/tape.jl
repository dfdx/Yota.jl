
abstract type AbstractOp end

"""
Variable represents a reference to an operation on a tape.
Variables can be used to index tape or keep reference to
a specific operation on the tape.

Variables can be free (only var ID is known) or bound (holds
reference to the operation on tape)
"""
mutable struct Variable
    _id::Union{<:Integer, Nothing}
    _op::Union{AbstractOp, Nothing}
end

Variable(id::Integer) = Variable(id, nothing)
Variable(op::AbstractOp) = Variable(nothing, op)

function Base.getproperty(v::Variable, p::Symbol)
    if p == :id
        if v._op !== nothing
            # variable bound to a specific operation on a tapea
            return v._op.id
        else
            # free variable with only ID
            return v._id
        end
    else
        return getfield(v, p)
    end
end

const V = Variable

Base.show(io::IO, v::Variable) = print(io, "%$(v.id)")

########################################################################
#                            OPERATIONS                                #
########################################################################

function Base.getproperty(op::AbstractOp, f::Symbol)
    if f == :typ
        return typeof(op.val)
    elseif f == :var
        return Variable(nothing, op)
    else
        getfield(op, f)
    end
end

# ## Input

mutable struct Input <: AbstractOp
    id::Int
    val::Any
end

Input(val::Any) = Input(0, val)

Base.show(io::IO, op::Input) = print(io, "inp %$(op.id)::$(op.typ)")


# ## Constant

# mutable struct Constant <: AbstractOp
#     id::Int
#     typ::Type
#     val
# end


# Constant(id::Int, val) = Constant(id, typeof(val), val)
# Base.show(io::IO, op::Constant) = print(io, "const %$(op.id) = $(op.val)::$(op.typ)")


# ## Assign

# mutable struct Assign <: AbstractOp
#     id::Int
#     src_id::Int
#     val
# end

# Base.show(io::IO, op::Assign) = print(io, "%$(op.id) = %$(op.src_id)::$(op.typ)")


## Call

mutable struct Call <: AbstractOp
    id::Int
    val::Any
    fn::Union{Function, Type, Variable}
    args::Vector{Any}   # vector of Variables or const values
end

function Base.show(io::IO, op::Call)
    arg_str = join(["$v" for v in op.args], ", ")
    print(io, "%$(op.id) = $(op.fn)($arg_str)::$(op.typ)")
end


# Call(fn::Union{Function, Type, Variable}, args...; val=nothing) =
#     Call(0, val, fn, [args...])

"""
Helper function to map a function only to Variable arguments of a Call
leaving constant values as is
"""
function map_vars(fn::Function, args::Union{Vector, Tuple})
    return map(v -> v isa Variable ? fn(v) : v, args)
end


"""
    mkcall(fn, args...; val=nothing)

Convenient constructor for Call operation. If val is nothing (default)
and call value can be calculated from (bound) variables and constants,
they are calculated. To prevent this behavior, set val to some neutral value.
"""
function mkcall(fn::Union{Function, Type, Variable}, args...; val=nothing)
    calculable = all(a -> !isa(a, Variable) || a._op !== nothing, args)
    if val === nothing && calculable
        args_ = map_vars(v -> v._op.val, args)
        val_ = fn(args_...)
    else
        val_ = val
    end
    return Call(0, val_, fn, [args...])
end



# """
# Copy struct `x` replacing specified fields with new values
# """
# function copy_with(x; kvs...)
#     T = typeof(x)
#     d = Dict(kvs)
#     flds = fieldnames(T)
#     new_flds = [get(d, f, getfield(x, f)) for f in flds]
#     return T(new_flds...)
# end


########################################################################
#                                 TAPE                                 #
########################################################################

const MaybeFunction = Union{Function, Nothing}

mutable struct Tape
    # linearized execution graph
    ops::Vector{<:AbstractOp}
    # id of result variable
    result::Variable
    # derivs[var_id] == grad_id
    derivs::Dict{Variable, Variable}
    # compiled tape or nothing
    compiled::MaybeFunction
    # device of the tape
    device::AbstractDevice
end

Tape(device::AbstractDevice) = Tape(AbstractOp[], Variable(0), Dict(), nothing, device)
Tape() = Tape(CPU())
Base.similar(tape::Tape) = Tape(AbstractOp[], tape.resultid, tape.derivs,
                                tape.compiled, tape.device)


function Base.show(io::IO, tape::Tape)
    println(io, "Tape")
    for op in tape.ops
        println(io, "  $op")
    end
end


inputs(tape::Tape) = [V(op) for op in tape.ops if op isa Input]
function inputs!(tape::Tape, vals...)
    @assert length(tape) == 0 "Can only set inputs to an empty tape"
    for val in vals
        push!(tape, Input(val))
    end
    return [V(op) for op in tape.ops[1:length(vals)]]
end

# Base.getindex(tape::Tape, i::Integer) = tape.ops[i]
Base.getindex(tape::Tape, v::Variable) = tape.ops[v.id]

# Base.setindex!(tape::Tape, op::AbstractOp, i::Integer) = (tape.ops[i] = op)

function Base.setindex!(tape::Tape, op::AbstractOp, v::Variable)
    op.id = v.id
    tape.ops[v.id] = op
    v._op = op   # bind to op, overriding v.id
end

Base.lastindex(tape::Tape) = lastindex(tape.ops)
Base.length(tape::Tape) = length(tape.ops)
Base.iterate(tape::Tape) = iterate(tape.ops)       # exclude inputs?
Base.iterate(tape::Tape, s) = iterate(tape.ops, s)

"""
    push!(tape::Tape, op::AbstractOp)

Push a new operation to the end of the tape.
"""
function Base.push!(tape::Tape, op::AbstractOp)
    new_id = length(tape) + 1
    op.id = new_id
    push!(tape.ops, op)
    return V(op)
end

"""
    insert(tape::Tape, idx::Integer, ops::AbstractOp...)

Insert new operations into tape starting from position idx.
"""
function Base.insert!(tape::Tape, idx::Integer, ops::AbstractOp...)
    num_new_ops = length(ops)
    old_ops = tape.ops
    new_ops = Vector{AbstractOp}(undef, length(tape) + num_new_ops)
    # copy old ops before insertion point
    for i=1:idx - 1
        new_ops[i] = old_ops[i]
    end
    # insert target ops, assign ids
    for i=1:num_new_ops
        id = idx + i - 1
        new_ops[id] = ops[i]
        new_ops[id].id = id
    end
    # insert the rest of old ops
    for i=idx:length(old_ops)
        id = i + num_new_ops
        new_ops[id] = old_ops[i]
        new_ops[id].id = id
    end
    tape.ops = new_ops
    # # note: making sure to reindex only ops after idx + num_new_ops
    # # otherwise we would affect newly inserted ops as well
    # reindex!(tape, st; from=idx + num_new_ops)
    return [V(op) for op in ops]
end


foo(x, y) = x * y

function test_it()
    tape = Tape()
    a1, a2, a3 = inputs!(tape, foo, 2.0, 5.0)
    r = push!(tape, mkcall(*, a2, a3))
    ops = [mkcall(+, a2, 1), mkcall(+, a3, 1)]
    v1, v2 = insert!(tape, 4, ops...)
    tape[r] = mkcall(*, v1, v2)
end



###############################################################################
#                                REINDEX                                      #
###############################################################################

# TODO: perhaps not needed with bound variables?

reindex!(op::Input, st::Dict) = (op.id = get(st, op.id, op.id))

function reindex!(op::Call, st::Dict)
    op.args = map_vars(v -> Variable(get(st, v.id, v.id)), op.args)
    op.id = get(st, op.id, op.id)
    return op
end

function reindex_fields!(tape::Tape, st::Dict)
    tape.resultid = get(st, tape.resultid, tape.resultid)
    tape.derivs = Dict(get(st, k, k) => get(st, v, v) for (k, v) in tape.derivs)
end

function reindex!(tape::Tape, st::Dict; from=1, to=length(tape))
    for id=from:to
        reindex!(tape[id], st)
    end
    reindex_fields!(tape, st)
    return tape
end






"""
Record a new operation (defined by its type and arguments for constructor)
to a tape and return ID of this new op.
"""
function record!(tape::Tape, optype::Type{<:AbstractOp}, args...)
    ret_id = length(tape) + 1
    push!(tape, optype(ret_id, args...))
    return ret_id
end

@deprecate record!(tape::Tape, optype::Type{<:AbstractOp}, args...) push!(tape, optype(args...))

"""
Parse a complex call expression and record corresponding operations to a tape.

Optionally takes substitution table (st parameter) to replace known symbols with
provided values.

Keyword params:

 * st::Dict - substitution table, used to replace symbols in ex with tape op ids
 * bcast::Bool - replace all function calls with corresponding broadcasting
"""
function record_expr!(tape::Tape, ex::Expr; st=Dict(), bcast=false)
    # special case: x => identity(x)
    if ex isa Symbol
        ex = :($identity($ex))
    end
    # special case: x[i] => getindex(x, i)
    if Meta.isexpr(ex, :ref)
        ex = rewrite(ex, :(_x[_i]), :($getindex(_x, _i)))
    end
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
            tape[i] = copy_with(op, args=new_args)
        end
    end
end


# """
# Recover broadcast operation from Broadcast.broadcasted and Broadcast.materialize
# """
# function recover_broadcast(tape::Tape)
#     new_tape = copy_with(tape; ops=AbstractOp[])
#     # TODO: seems like we don't need subs table any more
#     # remove after squash_assigned is implemented
#     st = Dict()
#     for (id, op) in enumerate(tape)
#         if op isa Call && op.fn === Broadcast.broadcasted
#             # replace Broadcast.broadcasted with just broadcast & materialize value
#             val = Broadcast.materialize(op.val)
#             new_id = length(new_tape) + 1
#             push!(new_tape.ops, copy_with(op; fn=broadcast, val=val))
#             st[id] = length(new_tape)
#         elseif op isa Call && op.fn === Broadcast.materialize
#             # materialize becomes just assignment
#             new_id = record!(new_tape, Assign, op.args[1], op.val)
#             st[id] = new_id
#         else
#             # record any other operations as is
#             new_id = length(new_tape) + 1
#             push!(new_tape.ops, copy_with(op; id=new_id))
#             st[id] = length(new_tape)
#         end
#     end
#     replace_in_args!(new_tape, st)
#     new_tape.resultid = get(st, tape.resultid, tape.resultid)
#     return new_tape
# end


# # function squash_assigned(tape::Tape)
# #     # note: after update to CuArrays v1.5.0 deepcopy(tape.ops) fails for Lilith.GRU if there's Assign op
# #     # I don't know the reason, but [deepcopy(op) for op in tape.ops] fixes it
# #     tape = copy_with(tape; ops=[deepcopy(op) for op in tape.ops])
# #     # 1. compute subs table for chains of assignment operations
# #     # and replace equivalent op ids
# #     assign_st = Dict()
# #     for (id, op) in enumerate(tape)
# #         if op isa Assign
# #             # propagate replacement through known st
# #             src_id = op.src_id
# #             while haskey(assign_st, src_id)
# #                 src_id = assign_st[src_id]
# #             end
# #             assign_st[id] = src_id
# #         end
# #     end
# #     replace_in_args!(tape, assign_st)
# #     tape.resultid = get(assign_st, tape.resultid, tape.resultid)
# #     # 2. create new ops w/o assignments
# #     new_tape = copy_with(tape; ops=AbstractOp[])
# #     reindex_st = Dict()
# #     for (id, op) in enumerate(tape)
# #         if !isa(op, Assign)
# #             # record any other operations as is
# #             new_id = length(new_tape) + 1
# #             push!(new_tape.ops, copy_with(op; id=new_id))
# #             if id != new_id
# #                 reindex_st[id] = new_id
# #             end
# #         end
# #     end
# #     replace_in_args!(new_tape, reindex_st)
# #     reindex_fields!(new_tape, reindex_st)
# #     return new_tape
# # end


# function squash_assigned(tape::Tape)
#     new_tape = copy_with(tape, ops=AbstractOp[])
#     st = Dict{Int, Int}()  # substitution table for op indices
#     for id=1:length(tape)
#         # id == 228 && break
#         op = reindex(tape[id], st)
#         if op isa Assign
#             # dig until not-assign argument is found
#             root = op
#             while root isa Assign
#                 root = new_tape[root.src_id]
#             end
#             st[op.id] = root.id   # replace this op id on _old_ tape with root id on the _new_ tape
#             # note: not pushing this op into new tape
#             # println("assign: $op; replacing $(op.id) => $(root.id)")
#         else
#             new_id = length(new_tape) + 1
#             new_op = copy_with(op, id=new_id)
#             push!(new_tape, new_op)
#             if new_id != id
#                 st[id] = new_id
#                 # println("subs $id => $new_id")
#             end
#         end
#     end
#     reindex_fields!(new_tape, st)
#     return new_tape
# end


# # function test_fuse_assigned()
# #     tape = Tape()
# #     record!(tape, Input, 3.0)
# #     record!(tape, Input, 2.0)
# #     record!(tape, Assign, 1, tape[1].val)
# #     record!(tape, Assign, 3, tape[3].val)
# #     record!(tape, Call, 5.0, +, [2, 4])
# #     record!(tape, Assign, 5, tape[5].val)
# #     tape.resultid = length(tape)
# # end



# # """
# # Unwind iterate() sequences into plain __getfield__ expressions.
# # unwind_iterate() doesn't remove unused elements for performance reasons,
# # so remove_unused() should be called after it.
# # """
# # function unwind_iterate(tape::Tape)
# #     tape = copy_with(tape)
# #     for op in tape
# #         if (op isa Call && op.fn in (getfield, __getfield__)
# #             && tape[op.args[1]] isa Call && tape[op.args[1]].fn == Base.iterate
# #             && tape[op.args[2]] isa Constant && tape[op.args[2]].val == 1)
# #             iterate_op = tape[op.args[1]]
# #             iterable_op = tape[iterate_op.args[1]]
# #             idx = length(iterate_op.args) > 1 ? tape[iterate_op.args[2]].val : 1
# #             if iterable_op.val isa Tuple || iterable_op.val isa Vector || iterable_op.val isa UnitRange
# #                 # 1. Replace iterable op with index in the original iterable
# #                 tape[iterate_op.id] = Constant(iterate_op.id, idx)
# #                 # 2. Replace __getfield__ on iterator with __getfield__ or getindex on original iterable
# #                 idx_id = iterate_op.id
# #                 obj_id = iterable_op.id
# #                 # TODO: in which other cases getindex is better than __getfield__?
# #                 get_op = iterable_op.val isa UnitRange ? getindex : __getfield__
# #                 tape[op.id] = Call(op.id, op.val, get_op, [obj_id, idx_id])
# #             end
# #         end
# #     end
# #     return tape
# # end


# function simplify(tape::Tape)
#     tape = recover_broadcast(tape)
#     tape = squash_assigned(tape)
#     # tape = unwind_iterate(tape)
#     tape = eliminate_common(tape)
#     tape = remove_unused(tape)
#     return tape
# end


########################################################################
#                              EXECUTION                               #
########################################################################


exec!(::Tape, ::Input) = ()
# exec!(::Tape, ::Constant) = ()
# exec!(tape::Tape, op::Assign) = (op.val = tape[op.src_id].val)

function exec!(tape::Tape, op::Call)
    # arg_vals = [v isa Variable ? tape[v.id].val : v for v in op.args]
    arg_vals = map_vars(v -> tape[v].val, op.args)
    op.val = op.fn(arg_vals...)
end


function play!(tape::Tape, args...; use_compiled=true, debug=false)
    for (i, val) in enumerate(args)
        @assert(tape[i] isa Input, "More arguments than the original function had")
        tape[i].val = val
    end
    if use_compiled && tape.compiled !== nothing
        Base.invokelatest(tape.compiled)
    else
        for op in tape
            if debug
                println(op)
            end
            exec!(tape, op)
        end
    end
    return tape[tape.resultid].val
end
