"Check if an object is of a struct type, i.e. not a number or array"
isstruct(::Type{T}) where T = !isbits(T) && !(T <: AbstractArray)
isstruct(obj) = !isbits(obj) && !isa(obj, AbstractArray)
