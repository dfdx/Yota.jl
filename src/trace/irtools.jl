import IRTools
import IRTools: IR, @dynamo, self, insertafter!


################################################################################
#                                Utils                                         #
################################################################################

# simple wrapper to distinguish constant values from SSA IDs
struct Value
    val::Any
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

function Base.show(io::IO, fr::Frame)
    map_str = join(["$k=>$v" for (k, v) in fr.ssa2tape], ",")
    print(io, "Frame($map_str, $(fr.resultid))")
end



################################################################################
#                        IRTracer (defs and utils)                             #
################################################################################

mutable struct IRTracer
    tape::Tape
    frames::Vector{Frame}
end

function IRTracer(;primitives=PRIMITIVES)
    tape = Tape()
    return IRTracer(tape, [])
end

Base.show(io::IO, t::IRTracer) = print(io, "IRTracer($(length(t.tape)))")


function ssa_args_to_tape_vars!(t::IRTracer, ssa_args)
    result = Vector{Int}(undef, length(ssa_args))
    for (i, arg) in enumerate(ssa_args)
        # println("result = $result; i = $i; arg = $arg")
        if arg isa Value
            val = arg.val isa QuoteNode ? arg.val.value : arg.val
            arg_var = record!(t.tape, Constant, val)
            result[i] = arg_var
        else
            # println("ssa2tape = $(t.frames[end].ssa2tape)")
            result[i] = t.frames[end].ssa2tape[arg]
        end
    end
    return result
end


function record!(t::IRTracer, ssa_id::Int, ssa_val::Any, optype::Type, ssa_args...)
    if optype == Call
        fn = ssa_args[1]
        tape_ids = ssa_args_to_tape_vars!(t, ssa_args[2:end])
        # record corresponding op to the tape
        ret_id = record!(t.tape, optype, ssa_val, fn, tape_ids)
    elseif optype == Constant
        @assert length(ssa_args) == 1
        # @show (t.tape, optype, ssa_args[1])
        val = ssa_args[1]
        val = val isa QuoteNode ? val.value : val
        # println(val, " ", typeof(val))
        ret_id = record!(t.tape, optype, val)
    else
        throw(ArgumentError("Cannot record to tracer's tape operation of type $optype"))
    end
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


################################################################################
#                        IRTracer (body) + irtrace()                           #
################################################################################

function rewrite_special_cases!(ir::IR)
    for (v, st) in ir
        if Meta.isexpr(st.expr, :new)
            ir[v] = Expr(:call, GlobalRef(@__MODULE__, :__new__), st.expr.args...)
        end
    end
end


function trace_statements!(ir::IR)
    for (v, st) in ir
        ex = st.expr
        if Meta.isexpr(ex, :call)
            # record primitive and recurse into non-primitive calls
            fn_gref = ex.args[1]
            fn = getproperty(fn_gref.mod, fn_gref.name)
            ssa_args = [v isa IRTools.Variable ? v.id : Value(v) for v in ex.args[2:end]]
            # insertafter!(ir, v, Expr(:call, println, fn.mod, " ", fn.name))
            # println(fn.mod, " ", fn.name)
            if fn in PRIMITIVES
                record_ex = Expr(:call, record!, self, v.id, v, Call, fn, ssa_args...)
                r = insertafter!(ir, v, record_ex)
            else
                ir[v] = Expr(:call, self, ex.args...)
                insert!(ir, v, Expr(:call, push_frame!, self, ssa_args...))
                insertafter!(ir, v, Expr(:call, pop_frame!, self, v.id))
            end
        elseif ex isa GlobalRef
            # resolve GlobalRef's and record as constants
            val = getproperty(ex.mod, ex.name)
            insertafter!(ir, v, Expr(:call, record!, self, v.id, v, Constant, val))
        end
    end
end


function trace_branches!(ir::IR)
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
end


@dynamo function (t::IRTracer)(fargs...)
    ir = IR(fargs...)
    ir == nothing && return   # intrinsic functions
    rewrite_special_cases!(ir)
    trace_statements!(ir)
    trace_branches!(ir)
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


# TODO:
# * new()
