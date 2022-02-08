var documenterSearchIndex = {"docs":
[{"location":"reference/","page":"Reference","title":"Reference","text":"CurrentModule = Yota","category":"page"},{"location":"reference/#Public-API","page":"Reference","title":"Public API","text":"","category":"section"},{"location":"reference/#Tracing","page":"Reference","title":"Tracing","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"trace\nisprimitive\nrecord_primitive!\nBaseCtx\n__new__","category":"page"},{"location":"reference/#Umlaut.record_primitive!","page":"Reference","title":"Umlaut.record_primitive!","text":"record_primitive!(tape::Tape{GradCtx}, v_fargs...)\n\nReplace ChainRules primitives f(args...) with a sequence:\n\nrr = push!(tape, mkcall(rrule, f, args...))   # i.e. rrule(f, args...)\nval = push!(tape, mkcall(getfield, rr, 1)     # extract value\npb = push!(tape, mkcall(getfield, rr, 2)      # extract pullback\n\n\n\n\n\n","category":"function"},{"location":"reference/#Variables","page":"Reference","title":"Variables","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Variable\nbound\nrebind!\nrebind_context!","category":"page"},{"location":"reference/#Tape-structure","page":"Reference","title":"Tape structure","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Tape\nAbstractOp\nInput\nConstant\nCall\nLoop\ninputs\ninputs!\nmkcall","category":"page"},{"location":"reference/#Tape-transformations","page":"Reference","title":"Tape transformations","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"push!\ninsert!\nreplace!\ndeleteat!\nprimitivize!","category":"page"},{"location":"reference/#Tape-execution","page":"Reference","title":"Tape execution","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"play!\ncompile\nto_expr","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"","category":"page"},{"location":"#Yota.jl","page":"Main","title":"Yota.jl","text":"","category":"section"},{"location":"","page":"Main","title":"Main","text":"Umlaut.jl is a code tracer for the Julia programming language. It lets you trace the function execution, recording all primitive operations onto a linearized tape. Here's a quick example:","category":"page"},{"location":"","page":"Main","title":"Main","text":"using Umlaut     # hide\ninc(x) = x + 1\nmul(x, y) = x * y\ninc_double(x) = mul(inc(x), inc(x))\n\nval, tape = trace(inc_double, 2.0)","category":"page"},{"location":"","page":"Main","title":"Main","text":"The tape can then be analyzed, modified and even compiled back to a normal function. See the following sections for details.","category":"page"},{"location":"","page":"Main","title":"Main","text":"note: Note\nUmlaut.jl was started as a fork of Ghost.jl trying to overcome some of its limitations, but eventually the codebase has diverged so much that the new package was born. Although the two have pretty similar API, there are several notable differences. See Migration from Ghost for details.","category":"page"}]
}