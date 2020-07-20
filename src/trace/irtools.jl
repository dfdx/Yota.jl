# import MacroTools
import IRTools
import IRTools: IR, @dynamo, recurse!, xcall, self, @code_ir, insertafter!


# const PRIMITIVE_TYPES = Set([typeof(f) for f in PRIMITIVES])

add(x, y) = x + y
mul(x, y) = x * y
add_mul(x, y) = add(mul(x, y), y)

logistic(x::Real) = one(x) / (one(x) + exp(-x))
softplus(x::Real) = log(exp(x) + one(x))
logsigmoid(x::Real) = -softplus(-x)

function with_loop(x, n)
    r = 1
    for i=1:n
        r *= x
    end
    return r
end



################################################################################
#                                Frame                                         #
################################################################################

"""Frame of a call stack"""
mutable struct Frame
    ssa2tape::Dict{Int, Int}   # SSA var ID to Tape var ID
end

Frame(tape_ids...) = Frame(Dict(tid => ssa for (ssa, tid) in enumerate(tape_ids)))

function Base.show(io::IO, fr::Frame)
    map_str = join(["$k=>$v" for (k, v) in fr.ssa2tape], ",")
    print(io, "Frame($map_str)")
end


mutable struct IRTracer
    primitive_types::Set{Type}
    tape::Tape
    frames::Vector{Frame}
end

function IRTracer(;primitives=PRIMITIVES)
    primitive_types = Set([typeof(f) for f in primitives])
    tape = Tape()
    return IRTracer(primitive_types, tape, [])
end

Base.show(io::IO, t::IRTracer) = print(io, "IRTracer($(length(t.tape)))")

# push_frame!(t::IRTracer, ssa2tape::Dict{Int, Int}) = push!(t.frames, Frame(ssa2tape))
# pop_frame!(t::IRTracer) = pop!(t.frames)
# get_current_frame(t::IRTracer) = t.frames[end]


function ssa_args_to_tape_vars!(t::IRTracer, ssa_args)
    result = Vector{Int}(undef, length(ssa_args))
    for (i, arg) in enumerate(ssa_args)
        # println("result = $result; i = $i; arg = $arg")
        if arg isa Value
            arg_var = record!(t.tape, Constant, arg.val)
            result[i] = arg_var
        else
            # println("ssa2tape = $(t.frames[end].ssa2tape)")
            result[i] = t.frames[end].ssa2tape[arg]
        end
    end
    return result
end


function record!(t::IRTracer, ssa_id::Int, ssa_val::Any, optype::Type, fn, ssa_args...)
    tape_ids = ssa_args_to_tape_vars!(t, ssa_args)
    # record corresponding op to the tape
    ret_id = record!(t.tape, optype, ssa_val, fn, tape_ids)
    # update mapping from SSA var to tape var
    # note that in functions with loops this mapping may change over time
    t.frames[end].ssa2tape[ssa_id] = ret_id    
end


function push_frame!(t::IRTracer, ssa_args...)
    tape_ids = ssa_args_to_tape_vars!(t, ssa_args)
    frame = Frame(Dict(i + 1 => tape_id for (i, tape_id) in enumerate(tape_ids)))
    push!(t.frames, frame)
end


function pop_frame!(t::IRTracer, ssa_id::Int)
    pop!(t.frames)
    # create mapping from the current SSA ID to the last instruction on the tape
    t.frames[end].ssa2tape[ssa_id] = length(t.tape)
end


# simple wrapper for constant values
struct Value
    val::Any
end


# GlobalRefs should take into account true module of a function
# as well as current module. An alernative is to resolve function in compile time,
# but I'm not sure it's possible
const PRIMITIVE_GREFS = union(
    Set(GlobalRef(Base.parentmodule(p), Base.nameof(p)) for p in PRIMITIVES),
    Set(GlobalRef(@__MODULE__, Base.nameof(p)) for p in PRIMITIVES),
)



@dynamo function (t::IRTracer)(fargs...)
    ir = IR(fargs...)
    ir == nothing && return   # intrinsic functions
    for (v, st) in ir
        Meta.isexpr(st.expr, :call) || continue
        fn = st.expr.args[1]
        ssa_args = [v isa IRTools.Variable ? v.id : Value(v) for v in st.expr.args[2:end]]        
        # insertafter!(ir, v, Expr(:call, println, fn.mod, " ", fn.name))
        if fn in PRIMITIVE_GREFS
            record_ex = Expr(:call, record!, self, v.id, v, Call, fn, ssa_args...)
            r = insertafter!(ir, v, record_ex)           
        else
            ir[v] = Expr(:call, self, st.expr.args...)
            insert!(ir, v, Expr(:call, push_frame!, self, ssa_args...))
            insertafter!(ir, v, Expr(:call, pop_frame!, self, v.id))
        end
        # TODO: to support branches, we need to insert a call to a function
        # which re-maps target block args to the current tape vars corresponding
        # to branch arguments, e.g.:
        #
        #    br 2 (%5, 1)
        #  2: (%8, %9)
        #
        # we need to update state as ssa2tape[8] = ssa2tape[5].
        # If there's another entry point to block 2, another updating instruction
        # will be added too.
        #
        # The question is: is it possible to insert an instruction between branches or
        # should we just push to the end of block?
    end
    # println(ir)
    # println("")
    return ir
end


function main()
    t = IRTracer()
    fargs = (add_mul, 8.0, 9.0)
    record!(t.tape, Input, 8.0)
    record!(t.tape, Input, 9.0)
    push!(t.frames, Frame(2, 3))
    # @code_ir add(8.0, 9.0)
    ir = @code_ir t(add, 8.0, 9.0)
    t(add_mul, 8.0, 9.0)

    @code_lowered t(with_loop, 2.0, 4)
    @code_ir t(with_loop, 2.0, 4)
    @code_ir with_loop(2.0, 4)
    t(with_loop, 2.0, 4)


    @time t(add, 8.0, 9.0)
    @time ctrace(add, 8.0, 9.0)

    @time t(softplus, 8.0)
    @time ctrace(softplus, 8.0)

    @time t(logsigmoid, 8.0)
    @time ctrace(logsigmoid, 8.0)
end
