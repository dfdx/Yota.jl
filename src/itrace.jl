import JuliaInterpreter
import JuliaInterpreter: enter_call, step_expr!, next_call!, @lookup, Frame

include("core.jl")



# function itrace(f, args...; primitives=PRIMITIVES, optimize=true)
# end


getexpr(fr::Frame, pc::Int) = fr.framecode.src.code[pc]
current_expr(fr::Frame) = getexpr(fr, fr.pc)



# if f in primitives
#         # args = with_tagged_properties(ctx, tape, args)    # only if f() is getproperty()
#         args = with_free_args_as_constants(ctx, tape, args)
#         arg_ids = [metadata(x, ctx) for x in args]
#         arg_ids = Int[id isa Cassette.NoMetaData ? -1 : id for id in arg_ids]
#         # execute call
#         retval = fallback(ctx, f, [untag(x, ctx) for x in args]...)
#         # record to the tape and tag with a newly created ID
#         ret_id = record!(tape, Call, retval, f, arg_ids)
#         retval = tag(retval, ctx, ret_id)
#     elseif canrecurse(ctx, f, args...)
#         retval = Cassette.recurse(ctx, f, args...)
#     else
#         retval = fallback(ctx, f, args...)
# end

# iscall(ex) =  || ()


# """
# Split JuliaInterpreter call expression into a tuple of 3 elements:

#  * function to be called
#  * args to this function
#  * vars on the tape corresponding to these args
# """
# function split_int_call(tape::Tape, fr::Frame, ex)
#     arr = Meta.isexpr(ex, :(=)) ? ex.args[2].args : ex.args
#     cf = @lookup(fr, arr[1])
#     cargs = [@lookup(fr, a) for a in arr[2:end]]
#     cvars = 
#     if 
#         f_args = 
#     else
#         f_args = [@lookup(fr, a) for a in ]
#     end
#     return f_args[1], f_args[2:end]
# end



function itrace!(f, tape::Tape, argvars...; primitives)
    args, vars = zip(argvars...)
    fr = enter_call(f, args...)
    frame_vars = Dict{Any, Int}(JuliaInterpreter.SlotNumber(i + 1) => v for (i, v) in enumerate(vars))    
    ex = current_expr(fr)
    while !Meta.isexpr(ex, :return)
        if Meta.isexpr(ex, :call) || (Meta.isexpr(ex, :(=)) && Meta.isexpr(ex.args[2], :call))
            arr = Meta.isexpr(ex, :(=)) ? ex.args[2].args : ex.args  # TODO: move or simplify
            # cf, cargs, cvars = function, args and vars of the current expression
            cf = @lookup(fr, arr[1])
            cargs = [@lookup(fr, a) for a in arr[2:end]]
            cvars = [frame_vars[wa] for wa in arr[2:end]]            
            if cf in primitives
                # we will map result to this location (SlotNumber or SSAValue)
                loc = Meta.isexpr(ex, :(=)) ? ex.args[1] : JuliaInterpreter.SSAValue(fr.pc)  # TODO: check 
                retval = next_call!(fr)
                ret_id = record!(tape, Call, retval, cf, cvars)                
                frame_vars[loc] = ret_id  # for slots, may overwrite old mapping
            else
                # TODO: handle recursive call
                itrace!(cf, tape, zip(cargs, cvars)...; primitives=primitives)
            end
        end
        ex = current_expr(fr)
    end
    # TODO: handle return
end



bar(x) = 2.0x + 1.0

function foo(x)
    y = bar(x)
    z = exp(y)
end


function itrace(f, args...; primitives=PRIMITIVES, optimize=true)
    tape = Tape(guess_device(args))
    argvars = Vector(undef, length(args))
    for (i, arg) in enumerate(args)        
        id = record!(tape, Input, arg)
        argvars[i] = (arg, id)
    end
    itrace!(f, tape, argvars...; primitives=primitives)
end


# NEXT STEPS:
# add bar() to primitives and finish non-recursive path of itrace!

function main()
    f = foo
    args = (4.0,)
    primitives = PRIMITIVES
    _itrace(f, args, ; primitives=primitives)
end
