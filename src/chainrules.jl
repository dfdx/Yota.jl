import ChainRulesCore: rrule, no_rrule
import ChainRulesCore: rrule_via_ad, RuleConfig, NoForwardsMode, HasReverseMode
import Umlaut: make_name, Input, to_expr


###############################################################################
#                              RuleConfig                                     #
###############################################################################

"""
    YotaRuleConfig()

ChainRules.RuleConfig passed to all `rrule`s in Yota.
Extends RuleConfig{Union{NoForwardsMode,HasReverseMode}}.
"""
struct YotaRuleConfig <: RuleConfig{Union{NoForwardsMode,HasReverseMode}} end


###############################################################################
#                              rrule_via_ad                                   #
###############################################################################


function to_rrule_expr(tape::Tape)
    # TODO (maybe): add YotaRuleConfig() as the first argument for consistency
    fn_name = :(ChainRulesCore.rrule)
    header = Expr(:call, fn_name)
    push!(header.args, Expr(:(::), :config, YotaRuleConfig))
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
make_rrule!(tape::Tape) = Base.eval(@__MODULE__, to_rrule_expr(tape))

function make_rrule!(f, args...)
    arg_str = join(["::$(typeof(a))" for a in args], ", ")
    @debug "Generating new rrule for $(f)($arg_str)"
    tape = gradtape(f, args...; seed=:auto, ctx=GradCtx())
    make_rrule!(tape)
end


"""
    rrule_via_ad(::YotaRuleConfig, f, args...)

Generate `rrule` using Yota.
"""
function ChainRulesCore.rrule_via_ad(cfg::YotaRuleConfig, f, args...)
    # global RVA_STATE = (f, args)
    arg_type_str = join(["::$(typeof(a))" for a in args], ", ")
    @debug "Running rrule_via_ad() for $f($arg_type_str)"
    res = rrule(cfg, f, args...)
    if !isnothing(res)
        y, pb = res
        return y, pb
    end
    @debug "No rrule in older world ages, falling back to invokelatest"
    res = Base.invokelatest(rrule, cfg, f, args...)
    if !isnothing(res)
        y, pb_ = res
        # note: returned pullback is still in future, so we re-wrap it into invokelatest too
        pb = dy -> Base.invokelatest(pb_, dy)
        return y, pb
    end
    @debug "No rrule in the latest world age, compiling a new one"
    make_rrule!(f, args...)
    res = Base.invokelatest(rrule, cfg, f, args...)
    y, pb_ = res
    pb = dy -> Base.invokelatest(pb_, dy)
    return y, pb
end