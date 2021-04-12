abstract type AbstractOp end

########################################################################
#                             VARIABLE                                 #
########################################################################

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

function Base.setproperty!(v::Variable, p::Symbol, x)
    if p == :id
        if v._op !== nothing
            # variable bound to a specific operation on a tapea
            v._op.id = x
        else
            # free variable with only ID
            v.id = x
        end
    else
        return setfield!(v, p, x)
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

## Input

mutable struct Input <: AbstractOp
    id::Int
    val::Any
end

Input(val::Any) = Input(0, val)

Base.show(io::IO, op::Input) = print(io, "inp %$(op.id)::$(op.typ)")


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
    calculable = all(
        a -> !isa(a, Variable) ||                      # not variable
        (a._op !== nothing && a._op.val !== nothing),  # bound variable
        args
    )
    if val === nothing && calculable
        args_ = map_vars(v -> v._op.val, args)
        val_ = fn(args_...)
    else
        val_ = val
    end
    return Call(0, val_, fn, [args...])
end


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

Base.getindex(tape::Tape, v::Variable) = tape.ops[v.id]

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
    insert!(tape::Tape, idx::Integer, ops::AbstractOp...)

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
    return [V(op) for op in ops]
end


"""
    replace!(tape, idx => [ops...]; rebind_to)

Replace operation at specified index with 1 or more other operations,
rebind variables in the reminder of the tape to ops[rebind_to].
"""
function Base.replace!(tape::Tape, idx_ops::Pair{<:Integer, <:Union{Tuple, Vector}};
                       rebind_to=length(idx_ops[2]))
    idx, ops = idx_ops
    tape[V(idx)] = ops[1]
    if idx < length(tape)
        insert!(tape, idx + 1, ops[2:end]...)
    else
        push!(ops[2:end]...)
    end
    st = Dict(idx => ops[rebind_to].id)
    rebind!(tape, st; from=idx + length(ops))
    return ops[rebind_to]
end


###############################################################################
#                                 REBIND                                      #
###############################################################################

"""
    rebind!(tape::Tape, op, st::Dict)
    rebind!(tape::Tape, st::Dict; from, to)

Rebind all variables according to substitution table. Example:

    tape = Tape()
    v1, v2 = inputs!(tape, nothing, 3.0, 5.0)
    v3 = push!(tape, mkcall(*, v1, 2))
    st = Dict(v1.id => v2.id)
    rebind!(tape, st)
    @assert tape[v3].args[1].id == v2.id

"""
function rebind!(tape::Tape, v::Variable, st::Dict)
    if haskey(st, v.id)
        # rebind to a new op
        v._op = tape[V(st[v.id])]
    end
end

rebind!(::Tape, ::Input, ::Dict) = ()

function rebind!(tape::Tape, op::Call, st::Dict)
    for v in op.args
        if v isa Variable
            rebind!(tape, v, st)
        end
    end
    return op
end

function rebind_fields!(tape::Tape, st::Dict)
    rebind!(tape, tape.result, st)
    for (v, dv) in tape.derivs
        rebind!(tape, v, st)
        rebind!(tape, dv, st)
    end
end

function rebind!(tape::Tape, st::Dict; from=1, to=length(tape))
    for id=from:to
        rebind!(tape, tape[V(id)], st)
    end
    rebind_fields!(tape, st)
    return tape
end


########################################################################
#                              EXECUTION                               #
########################################################################

exec!(::Tape, ::Input) = ()

function exec!(tape::Tape, op::Call)
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
    return tape[tape.result].val
end