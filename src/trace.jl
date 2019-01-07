using Cassette
using Cassette: Tagged, tag, untag, istagged, metadata, hasmetadata,
                enabletagging, @overdub, overdub, canrecurse, similarcontext, fallback


Cassette.@context TraceCtx

# allow assiciation of Int values with TraceCtx
# Cassette.metadatatype(::Type{<:TraceCtx}, ::Type{<:Number}) = Int
Cassette.metadatatype(::Type{<:TraceCtx}, ::DataType) = Int
# Cassette.metadatatype(::Type{<:TraceCtx}, ::Type) = Any
Cassette.hastagging(::Type{<:TraceCtx}) = true



function trace(f, args...)
    println(args)
    # create tape
    tape = Tape()
    ctx = enabletagging(TraceCtx(metadata=tape), f)
    tagged_args = Vector(undef, length(args))
    for (i, x) in enumerate(args)
        id = record!(tape, Input, typeof(x))
        tagged_args[i] = tag(x, ctx, i)
    end
    # trace f with tagged arguments
    val = overdub(ctx, f, tagged_args...)
    # TODO: ret_id = metadata(val, ctx)   -- ID of returned value, write as Return(ret_id) to tape?
    return val, tape
end


const PRIMITIVES = Set([*, /, +, -, Base.getproperty, println, sin, cos, broadcast])


function with_free_args_as_constants(ctx::TraceCtx, tape::Tape, args)
    new_args = []
    for x in args
        if istagged(x, ctx)
            push!(new_args, x)
        else
            id = record!(tape, Constant, x)
            x = tag(x, ctx, id)
            push!(new_args, x)
        end
    end
    return new_args
end


function Cassette.overdub(ctx::TraceCtx, f, args...)
    args_str = join([a isa Nothing ? "nothing" : a for a in args], ", ")
    @info("@@@ $f($args_str)")
    tape = ctx.metadata
    if f in PRIMITIVES
        args = with_free_args_as_constants(ctx, tape, args)
        arg_ids = [metadata(x, ctx) for x in args]
        arg_ids = Int[id isa Cassette.NoMetaData ? -1 : id for id in arg_ids]
        # execute call
        retval = fallback(ctx, f, [untag(x, ctx) for x in args]...)
        # record to the tape and tag with a newly created ID
        retval_str = repr(retval)
        @info("~~~ $retval_str <- $f($args_str)")
        ret_id = record!(tape, Call, typeof(retval), f, arg_ids)
        retval = tag(retval, ctx, ret_id)
    elseif canrecurse(ctx, f, args...)
        retval = Cassette.recurse(ctx, f, args...)
        # @assert !istagged(retval, ctx) "Return value isn't tagged: $retval <- $f($(args...))"
    else
        # error("Non-tracable non-primitive function: $f($args...)")
        retval = fallback(ctx, f, args...)
        # retval = tag(retval, ctx, length(ctx.metadata.ops) + 1)
    end
    return retval
end
