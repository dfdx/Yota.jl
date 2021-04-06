import IRTools: IR, @dynamo, self, insertafter!, var, xcall, Variable


################################################################################
#                                Frame                                         #
################################################################################

"""Frame of a call stack"""
mutable struct Frame
    # function + arguments (for introspection only)
    fargs
    # source var ID to tape (target) var
    src2tape::Dict{Int, Int}
    # result ID - tape ID corresponding to latest return value
    # from this call frame
    resultid::Int
end

function Base.show(io::IO, fr::Frame)
    map_str = join(["$k=>$v" for (k, v) in fr.src2tape], ",")
    print(io, "Frame($map_str, $(fr.resultid))")
end


################################################################################
#                        IRTracer (defs and utils)                             #
################################################################################

mutable struct IRTracer
    primitives::TypeTrie
    tape::IR
    frames::Vector{Frame}
end


function IRTracer(f, args::Tuple, primitives::TypeTrie)
    tape = IR()
    for _=1:length(args) + 1
        IRTools.argument!(tape)
    end
    frame = Frame((f, args...), Dict(i => i for i in 1:length(args) + 1), -1)
    return IRTracer(primitives, tape, [frame])
end



# function IRTracer(;primitives=PRIMITIVES)
#     tape = IR()  # create from meta instead?
#     return IRTracer(primitives, tape, [])
# end

Base.show(io::IO, t::IRTracer) = print(io, "IRTracer()")


# promote_const_value(x::QuoteNode) = x.value
# promote_const_value(x::GlobalRef) = getproperty(x.mod, x.name)
# promote_const_value(x) = x


# function ssa_args_to_tape_vars!(t::IRTracer, arg_defs::Union{Vector, Tuple})
#     result = Vector{Int}(undef, length(arg_defs))
#     for (i, arg) in enumerate(arg_defs)
#         if arg isa IRTools.Variable
#             result[i] = t.frames[end].src2tape[arg.id]
#         else
#             val = promote_const_value(arg)
#             arg_var = record!(t.tape, Constant, val)
#             result[i] = arg_var
#         end
#     end
#     return result
# end

function source2tape(t::IRTracer, src_vars)
    return [v isa Variable ? Variable(t.frames[end].src2tape[v.id]) : v
            for v in src_vars]
end


"""Push a new call frame to tracer, setting function params accordingly"""
function push_frame!(t::IRTracer, farg_defs, fargs)
    tape_vars = source2tape(t, farg_defs)
    frame = Frame(
        fargs,
        Dict(i => v.id for (i, v) in enumerate(tape_vars) if v isa Variable),
         -1)
    push!(t.frames, frame)
end


"""Pop call frame from tracer"""
function pop_frame!(t::IRTracer, res_sid::Int)
    frame = pop!(t.frames)
    # create mapping from the current SSA ID to the last instruction on the tape
    t.frames[end].src2tape[res_sid] =
        (frame.resultid == -1 ? length(t.tape) : frame.resultid)
end


"""Set target branch parameters to variables corresponding to SSA args"""
function set_branch_params!(t::IRTracer, ssa_args, target_params)
    tape_vars = source2tape(t, ssa_args)
    src2tape = t.frames[end].src2tape
    for (v, p) in zip(tape_vars, target_params)
        if v isa Variable
            src2tape[p] = v.id
        end
    end
end


"""Set return variable for the current frame"""
function set_return!(t::IRTracer, arg_sid_ref)
    # global STATE = (t, arg_sid_ref)
    tape_var = source2tape(t, [arg_sid_ref[]])[1]
    t.frames[end].resultid = tape_var.id
end


################################################################################
#                        IRTracer (body) + irtrace()                           #
################################################################################

# function rewrite_special_cases!(ir::IR)
#     for (v, st) in ir
#         if Meta.isexpr(st.expr, :new)
#             ir[v] = Expr(:call, __new__, st.expr.args...)
#         end
#     end
# end


"""
Record function call onto a tape or recurse into it.

Params:
-------
* t::IRTracer - current tracer
* src_id::Int - source ID of the operation
* src_fargs - IR variables of the operation
* fargs - values of the operation
"""
function record_or_recurse!(t::IRTracer, src_id::Int, src_fargs, fargs...)
    fn, args = fargs[1], fargs[2:end]
    global STATE = (t, src_id, src_fargs, fargs)
    if map(typeof, fargs) in t.primitives
        res = fn(args...)
        # tape_ids = ssa_args_to_tape_vars!(t, farg_defs[2:end])
        # tape_ids = t.frames[end].src2tape[]

        tape_fargs = source2tape(t, src_fargs)
        # record corresponding op to the tape
        tape_var = push!(t.tape, xcall(tape_fargs...))
        # res_tid = record!(t.tape, Call, res, fn, tape_ids)

        # update mapping from SSA var to tape var
        # note that in functions with loops this mapping may change over time
        t.frames[end].src2tape[src_id] = tape_var.id
    else
        push_frame!(t, src_fargs, fargs)
        res = t(fn, args...)
        pop_frame!(t, src_id)
    end
    return res
end


# function record_const!(t::IRTracer, res_sid, val)
#     val = val isa QuoteNode ? val.value : val
#     res_tid = record!(t.tape, Constant, val)
#     t.frames[end].src2tape[res_sid] = res_tid
#     return val
# end


function trace_branches!(ir::IR)
    # if a block ends with a branch, we map its parameters to tape IDs
    # which currently correspond to argument SSA IDs
    for block in IRTools.blocks(ir)
        for branch in IRTools.branches(block)
            if IRTools.isreturn(branch)
                ret_v = branch.args[1]
                push!(ir, Expr(:call, set_return!, self, Ref(ret_v)))
            else
                ssa_args = branch.args
                target_params = [v.id for v in ir.blocks[branch.block].args]
                push!(block, Expr(:call, set_branch_params!, self, ssa_args, target_params))
            end
        end
    end
end


@dynamo function (t::IRTracer)(fargs...)
    ir = IR(fargs...)
    ir === nothing && return   # intrinsic functions
    # rewrite_special_cases!(ir)
    for (v, st) in ir
        ex = st.expr
        # note the difference:
        # * `ex.args` is an array and thus will be passed to a function as is,
        # including definitions of IRTools.Variable;
        # * `ex.args...` is top-level to this expression and thus all Variable's
        # will be replaced with actual values during runtime
        if Meta.isexpr(ex, :call)
            ir[v] = IRTools.xcall(record_or_recurse!, self, v.id, ex.args, ex.args...)
        else
            # e.g. GlobalRef
            # note: using insertafter!() due to
            # https://github.com/FluxML/IRTools.jl/issues/78

            # insertafter!(ir, v, IRTools.xcall(record_const!, self, v.id, v))
        end
    end
    trace_branches!(ir)
    return ir
end


function trace(f, args...; primitives=PRIMITIVES)
    # init tracer
    t = IRTracer(f, args, primitives)
    # recursively trace function call
    val = t(f, args...)
    # t.tape.resultid = t.frames[1].resultid
    tape = t.tape
    return val, tape
end