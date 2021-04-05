const UType = Union{Type, Tuple, TypeVar}

"""
Trie of types. This data structure is useful for matching function
signatures in runtime.

Example:
========

tt = TypeTrie()
push!(tt, (typeof(sin), Number))
push!(tt, (typeof(sin), Missing))
push!(tt, (typeof(cos), Number))

(typeof(sin), Number) in tt   # ==> true
# also checks for supertypes
(typeof(sin), Float64) in tt  # ==> true
(typeof(sin), String) in tt   # ==> false
"""
mutable struct TypeTrie
    children::Dict{UType, TypeTrie}
    is_key::Bool
end

TypeTrie() = TypeTrie(Dict{UType, TypeTrie}(), false)


function Base.push!(s::TypeTrie, key)
    if isempty(key)
        s.is_key = true
    else
        head, tail = key[1], key[2:end]
        if !haskey(s.children, head)
            s.children[head] = TypeTrie()
        end
        push!(s.children[head], tail)
    end
end


"""
Check if the key (tuple of types) is in the trie.
Also checks all supertypes of each type in the key.
"""
function Base.in(key, s::TypeTrie)
    if isempty(key)
        return s.is_key
    else
        head, tail = key[1], key[2:end]
        while !haskey(s.children, head) && head !== Any
            # check supertypes of the current key head up to Any
            head = supertype(head)
        end
        # now head is either in s.chidlren or is non-matched Any
        if haskey(s.children, head)
            return tail in s.children[head]
        elseif haskey(s.children, Vararg)  # didn't match concrete types
            # TODO: find the most specific Vararg{T, N}
            return s.children[Vararg].is_key
        else
            return false
        end
    end
end


function print_trie(s::TypeTrie; indent=0)
    indent_str = join(["  " for _=1:indent + 1])
    if s.is_key
        println("Trie (key)")
    else
        println("Trie")
    end
    for (h, c) in s.children
        print("$(indent_str)$(h) ==> ")
        print_trie(c; indent=indent + 1)
    end
end