# isstruct(::Type{T}) where T = !isbitstype(T) && !(T <: AbstractArray)
# isstruct(obj) = !isbits(obj) && !isa(obj, AbstractArray)
"Check if an object is of a struct type, i.e. not a number or array"
isstruct(::Type{T}) where T = !isempty(fieldnames(T))
isstruct(obj) = isstruct(typeof(obj))
