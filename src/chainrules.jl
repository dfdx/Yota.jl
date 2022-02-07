import ChainRulesCore: rrule, no_rrule
import ChainRulesCore: rrule_via_ad, RuleConfig, NoForwardsMode, HasReverseMode
import Umlaut: make_name, Input, to_expr


###############################################################################
#                              Primitives                                     #
###############################################################################


struct ChainRulesCtx end

function isprimitive(::ChainRulesCtx, f, args...)
    Ts = [a isa DataType ? Type{a} : typeof(a) for a in (f, args...)]
    Core.Compiler.return_type(rrule, (YotaRuleConfig, Ts...,)) !== Nothing && return true
    if is_kwfunc(Ts[1])
        Ts_kwrrule = (Any, typeof(Core.kwfunc(f)), YotaRuleConfig, Ts[2:end]...,)
        Core.Compiler.return_type(Core.kwfunc(rrule), Ts_kwrrule) !== Nothing && return true
    end
    return false
end


###############################################################################
#                              RuleConfig                                     #
###############################################################################

struct YotaRuleConfig <: RuleConfig{Union{NoForwardsMode,HasReverseMode}} end


function to_rrule_expr(tape::Tape)
    fn_name = gensym("rrule_$(tape[V(1)].val)")
    header = Expr(:call, fn_name)
    for v in inputs(tape)
        op = tape[v]
        push!(header.args, Expr(:(::), make_name(op), op.typ))
    end
    body = Expr(:block)
    # generate transformed forward pass
    seed_id = tape.meta[:seed].id
    for op in tape.ops[1:seed_id - 1]
        op isa Input && continue
        ex = to_expr(op)
        if ex isa Vector
            push!(body.args, ex...)
        else
            push!(body.args, ex)
        end
    end
    # generate pullback
    pb_name = gensym("pullback_$(tape[V(1)].val)")
    pb_ex = :(function $pb_name(dy) end)
    pb_body = pb_ex.args[2]
    empty!(pb_body.args)  # clean from useless linenumber nodes
    push!(pb_body.args, Expr(:(=), make_name(tape.meta[:seed].id), :dy))
    for op in tape.ops[seed_id + 1:length(tape) - 2]
        op isa Input && continue
        ex = to_expr(op)
        if ex isa Vector
            push!(pb_body.args, ex...)
        else
            push!(pb_body.args, ex)
        end
    end
    push!(body.args, pb_ex)
    # generate return
    result_name = make_name(tape[tape.result].args[1].id)
    push!(body.args, Expr(:tuple, result_name, pb_name))
    fn_ex = Expr(:function, header, body)
    return fn_ex
end


"""
    make_rrule(tape::Tape)
    make_rrule(f, args...)

Generate a function equivalent to (but not extending) ChainRulesCore.rrule(),
i.e. returning the primal value and the pullback.


Examples:
=========

    foo(x) = 2x + 1
    rr = make_rrule(foo, 2.0)
    val, pb = rr(foo, 3.0)
    pb(1.0)

"""
make_rrule(tape::Tape) = Base.eval(@__MODULE__, to_rrule_expr(tape))
make_rrule(f, args...) = make_rrule(gradtape(f, args...))


const GENERATED_RRULE_CACHE = Dict()

function ChainRulesCore.rrule_via_ad(::YotaRuleConfig, f, args...)
    res = rrule(f, args...)
    !isnothing(res) && return res
    sig = map(typeof, (f, args...))
    if haskey(GENERATED_RRULE_CACHE, sig)
        rr = GENERATED_RRULE_CACHE[sig]
        # return Base.invokelatest(rr, f, args...)
        val, pb = Base.invokelatest(rr, f, args...)
        return val, dy -> Base.invokelatest(pb, dy)
    else
        rr = make_rrule(f, args...)
        GENERATED_RRULE_CACHE[sig] = rr
        # return Base.invokelatest(rr, f, args...)
        val, pb = Base.invokelatest(rr, f, args...)
        return val, dy -> Base.invokelatest(pb, dy)
    end
end