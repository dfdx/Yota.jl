function rrule_types(rr::Method)
    sig = rr.sig
    while sig isa UnionAll
        sig = sig.body
    end
    return tuple(collect(sig.parameters[2:end])...)
end


function rrule_signatures()
    rrules = methods(rrule).ms
    return [rrule_types(rr) for rr in rrules]
end


struct FunctionResolver{T}
    signatures::Dict{Any, Vector{Pair{DataType, T}}}
    FunctionResolver{T}() where T = new{T}(Dict())
end

function FunctionResolver{T}(pairs::Vector{Pair{S, T} where S}) where T
    rsv = FunctionResolver{T}()
    for (sig, val) in pairs
        rsv[sig] = val
    end
    order!(rsv)
    return rsv
end

Base.show(io::IO, rsv::FunctionResolver) = print(io, "FunctionResolver($(length(rsv.signatures)))")


function Base.setindex!(rsv::FunctionResolver{T}, val::T, sig::Tuple) where T
    fn_typ = sig[1]
    tuple_sig = Tuple{sig...}
    if !haskey(rsv.signatures, fn_typ)
        rsv.signatures[fn_typ] = Pair[]
    end
    push!(rsv.signatures[fn_typ], tuple_sig => val)
    return val
end

function Base.getindex(rsv::FunctionResolver{T}, sig::Tuple) where T
    fn_typ = sig[1]
    if haskey(rsv.signatures, fn_typ)
        tuple_sig = Tuple{sig...}
        for (TT, val) in rsv.signatures[fn_typ]
            if tuple_sig <: TT
                return val
            end
        end
    end
    return nothing
end

function order!(rsv::FunctionResolver)
    for (fn_typ, sigs) in rsv.signatures
        sort!(sigs, lt=(p1, p2) -> p1[1] <: p2[1])
    end
end

Base.haskey(rsv::FunctionResolver, sig::Tuple) = (rsv[sig] !== nothing)
Base.in(sig::Tuple, rsv::FunctionResolver{Bool}) = haskey(rsv, sig)
Base.empty!(rsv::FunctionResolver) = empty!(rsv.signatures)


function test_it()
    rsv = FunctionResolver{Symbol}()
    rsv[(typeof(sin), Vararg)] = :Vararg
    rsv[(typeof(sin), Float64)] = :Float64
    rsv[(typeof(sin), Real)] = :Real
    rsv[(typeof(sin), Number)] = :Number
    order!(rsv)

    @test rsv[(typeof(sin), Float64)] == :Float64
    @test rsv[(typeof(sin), Float32)] == :Real
    @test rsv[(typeof(sin), Float64, Float64)] == :Vararg

    # non-matching signature
    rsv[(typeof(cos), Number)] = :CosineNumber
    @test rsv[(typeof(cos), String)] === nothing
end


const CHAIN_RULE_PRIMITIVES = Ref(FunctionResolver{Bool}())


function update_chain_rules!()
    P = FunctionResolver{Bool}([sig => true for sig in rrule_signatures()])
    delete!(P.signatures, Any)
    CHAIN_RULE_PRIMITIVES[] = P
end


is_chainrules_primitive(sig) = sig in CHAIN_RULE_PRIMITIVES[]
