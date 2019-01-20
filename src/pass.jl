# TODO: this is the beginning of a custom compiler pass for Cassette
# The plan is to use it later to overload type constructors
# One possible way to implement it is to replace `Expr(:new, typ, ...)` with
# `Expr(:call, __new__, typ, ...)` and overload this function
# Overdubbing mechanism should do the rest (including tagging)
# `replace_match!` might be enough to implement it
# But we will come back to it when the main tracer is in a more
# mature state

include("tape.jl")

using Cassette
using Core: CodeInfo, SlotNumber, SSAValue

Cassette.@context TraceCtx

# function Cassette.overdub(ctx::TraceCtx, f, callback, args...)
#     if Cassette.canrecurse(ctx, f, args...)
#         _ctx = Cassette.similarcontext(ctx, metadata = callback)
#         return Cassette.recurse(_ctx, f, args...) # return result, callback
#     else
#         return Cassette.fallback(ctx, f, args...), callback
#     end
# end

# function Cassette.overdub(ctx::TraceCtx, ::typeof(println), callback, args...)
#     return nothing, () -> (callback(); println(args...))
# end


function insert_statements_at!(code, codelocs, pos, stmts)
    count = length(stmts)
    Cassette.insert_statements!(code, codelocs,
                                (stmt, i) -> i == pos ? count : nothing,
                                (stmt, i) -> stmts)
end

function record_on_tape(::Type{<:TraceCtx}, reflection::Cassette.Reflection)
    ir = reflection.code_info
    tapeslotname = gensym("callback")
    push!(ir.slotnames, tapeslotname)
    push!(ir.slotflags, 0x00)
    tapeslot = SlotNumber(length(ir.slotnames))
    getmetadata = Expr(:call, Expr(:nooverdub, GlobalRef(Core, :getfield)), Expr(:contextslot), QuoteNode(:metadata))

    # insert the initial `tapeslot` assignment into the IR.
    # Cassette.insert_statements!(ir.code, ir.codelocs,
    #                             (stmt, i) -> i == 1 ? 2 : nothing,
    #                             (stmt, i) -> [Expr(:(=), tapeslot, getmetadata), stmt])
    tapeslot_expr = Expr(:(=), tapeslot, getmetadata)
    insert_statements_at!(ir.code, ir.codelocs, 1, [tapeslot_expr, ir.code[1]])


    # # replace all calls of the form `f(args...)` with `f(callback, args...)`, taking care to
    # # properly handle Core._apply calls and destructure the returned `(result, callback)`
    # # into the appropriate statements
    # Cassette.insert_statements!(ir.code, ir.codelocs,
    #                             (stmt, i) -> begin
    #                                 i > 1 || return nothing # don't slice the callback assignment
    #                                 stmt = Base.Meta.isexpr(stmt, :(=)) ? stmt.args[2] : stmt
    #                                 if Base.Meta.isexpr(stmt, :call)
    #                                     isapply = Cassette.is_ir_element(stmt.args[1], GlobalRef(Core, :_apply), ir.code)
    #                                     return 3 + isapply
    #                                 end
    #                                 return nothing
    #                             end,
    #                             (stmt, i) -> begin
    #                                 items = Any[]
    #                                 callstmt = Base.Meta.isexpr(stmt, :(=)) ? stmt.args[2] : stmt
    #                                 callssa = SSAValue(i)
    #                                 if Cassette.is_ir_element(callstmt.args[1], GlobalRef(Core, :_apply), ir.code)
    #                                     push!(items, Expr(:call, Expr(:nooverdub, GlobalRef(Core, :tuple)), callbackslot))
    #                                     push!(items, Expr(:call, callstmt.args[1], callstmt.args[2], SSAValue(i), callstmt.args[3:end]...))
    #                                     callssa = SSAValue(i + 1)
    #                                 else
    #                                     push!(items, Expr(:call, callstmt.args[1], callbackslot, callstmt.args[2:end]...))
    #                                 end
    #                                 push!(items, Expr(:(=), callbackslot, Expr(:call, Expr(:nooverdub, GlobalRef(Core, :getfield)), callssa, 2)))
    #                                 result = Expr(:call, Expr(:nooverdub, GlobalRef(Core, :getfield)), callssa, 1)
    #                                 if Base.Meta.isexpr(stmt, :(=))
    #                                     result = Expr(:(=), stmt.args[1], result)
    #                                 end
    #                                 push!(items, result)
    #                                 return items
    #                             end)

    # # replace return statements of the form `return x` with `return (x, callback)`
    # Cassette.insert_statements!(ir.code, ir.codelocs,
    #                               (stmt, i) -> Base.Meta.isexpr(stmt, :return) ? 2 : nothing,
    #                               (stmt, i) -> begin
    #                                   return [
    #                                       Expr(:call, Expr(:nooverdub, GlobalRef(Core, :tuple)), stmt.args[1], callbackslot)
    #                                       Expr(:return, SSAValue(i))
    #                                   ]
    #                               end)
    return ir
end

const record_on_tape_pass = Cassette.@pass record_on_tape




########### test ###########


function add(a, b)
    println("I'm about to add $a + $b")
    c = a + b
    println("c = $c")
    return c
end



function main()
    a = rand(3)
    b = rand(3)

    add(a, b)

    tape = Tape()
    ctx = TraceCtx(pass=record_on_tape_pass, metadata=tape)
    result = Cassette.@overdub(ctx, add(a, b))
    # callback()

end
