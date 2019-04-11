# isstruct(::Type{T}) where T = !isbitstype(T) && !(T <: AbstractArray)
# isstruct(obj) = !isbits(obj) && !isa(obj, AbstractArray)
"Check if an object is of a struct type, i.e. not a number or array"
isstruct(::Type{T}) where T = !isempty(fieldnames(T))
isstruct(obj) = isstruct(typeof(obj))


if !isdefined(@__MODULE__, :__EXPRESSION_HASHES__)
    __EXPRESSION_HASHES__ = Set{AbstractString}()
end

"""
If loaded twice without changes, evaluate expression only for the first time.
"""
macro runonce(expr)
    h = string(expr)
    return esc(quote
        if !in($h, __EXPRESSION_HASHES__)
            push!(__EXPRESSION_HASHES__, $h)
            $expr
        end
    end)
end
