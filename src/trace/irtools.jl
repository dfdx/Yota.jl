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
    # SSA var ID to Tape var ID. Note that SSA ID refers to the _original_ IR,
    # not the transformed one. In fact, any unique name of SSA instructions
    # would fit, using SSA IDs is just convenient
    ssa2tape::Dict{Int, Int}
    # result ID - tape ID corresponding to latest return value
    # from this call frame
    resultid::Int
end

# Frame(tape_ids...) = Frame(Dict(tid => ssa for (ssa, tid) in enumerate(tape_ids)), -1)

function Base.show(io::IO, fr::Frame)
    map_str = join(["$k=>$v" for (k, v) in fr.ssa2tape], ",")
    print(io, "Frame($map_str, $(fr.resultid))")
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


"""Push a new call frame to tracer, setting function params accordingly"""
function push_frame!(t::IRTracer, ssa_args...)
    tape_ids = ssa_args_to_tape_vars!(t, ssa_args)
    frame = Frame(Dict(i + 1 => tape_id for (i, tape_id) in enumerate(tape_ids)), -1)
    push!(t.frames, frame)
end


"""Pop call frame from tracer"""
function pop_frame!(t::IRTracer, ssa_id::Int)
    frame = pop!(t.frames)
    # create mapping from the current SSA ID to the last instruction on the tape
    t.frames[end].ssa2tape[ssa_id] = (frame.resultid == -1 ?
                                      length(t.tape) :
                                      frame.resultid)
end


"""Set target branch parameters to vaiables corresponding to SSA args"""
function set_branch_params!(t::IRTracer, ssa_args, target_params)
    tape_vars = ssa_args_to_tape_vars!(t, ssa_args)
    ssa2tape = t.frames[end].ssa2tape
    for (v, p) in zip(tape_vars, target_params)
        ssa2tape[p] = v
    end
end


"""Set return variable for the current frame"""
function set_return!(t::IRTracer, ssa_arg)
    tape_var = ssa_args_to_tape_vars!(t, [ssa_arg])[1]
    t.frames[end].resultid = tape_var
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
    Set(GlobalRef(Base, x) for x in (:+, :*, :(:), :iterate, :not_int)),
    Set(GlobalRef(Core, x) for x in (:(===),)),
    Set(GlobalRef(@__MODULE__, x) for x in (:+, :*, :(:), :iterate, :not_int)),
    
);



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
    end
    # if a block ends with a branch, we map its parameters to tape IDs
    # which currently correspond to argument SSA IDs
    for block in IRTools.blocks(ir)
        for branch in IRTools.branches(block)
            if IRTools.isreturn(branch)
                ssa_arg_ = branch.args[1]
                ssa_arg = ssa_arg_ isa IRTools.Variable ? ssa_arg_.id : ssa_arg_
                push!(ir, Expr(:call, set_return!, self, ssa_arg))
            else
                ssa_args = [v isa IRTools.Variable ? v.id : Value(v) for v in branch.args]
                target_params = [v.id for v in ir.blocks[branch.block].args]
                push!(block, Expr(:call, set_branch_params!, self, ssa_args, target_params))
            end
        end
    end
    return ir
end


function irtrace(f, args...; optimize=true)
    t = IRTracer()
    for arg in args
        record!(t.tape, Input, arg)
    end
    push!(t.frames, Frame(Dict(i + 1 => i for i in 1:length(args)), -1))
    val = t(f, args...)
    t.tape.resultid = t.frames[1].resultid
    tape = t.tape
    if optimize 
        tape = simplify(tape)
    end
    return val, tape
end


function main()
    t = IRTracer()
    fargs = (add_mul, 8.0, 9.0)
    record!(t.tape, Input, 8.0)
    record!(t.tape, Input, 9.0)
    push!(t.frames, Frame(Dict(2 => 1, 3 => 2), -1))
    # @code_ir add(8.0, 9.0)
    ir = @code_ir t(add, 8.0, 9.0)
    t(add_mul, 8.0, 9.0)

    @code_lowered t(with_loop, 2.0, 4)
    @code_ir t(with_loop, 2.0, 4)
    ir = @code_ir with_loop(2.0, 4)

    t = IRTracer()
    record!(t.tape, Input, 2.0)
    record!(t.tape, Input, 4)
    push!(t.frames, Frame(Dict(2 => 1, 3 => 2), -1))
    ir = @code_ir with_loop(2.0, 4)
    @code_ir t(with_loop, 2.0, 4)
    t(with_loop, 2.0, 4)
    t.tape.resultid = t.frames[1].resultid
    simplify(t.tape)


    @time t(add, 8.0, 9.0)
    @time ctrace(add, 8.0, 9.0)

    @time t(softplus, 8.0)
    @time ctrace(softplus, 8.0)

    @time t(logsigmoid, 8.0)
    @time ctrace(logsigmoid, 8.0)
end
