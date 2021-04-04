## general tape transformations


"""
Trace function execution, apply transformation on the resulting tape and
recompile function back. Returns function as is if transformation didn't result
in any changes (as determined by the second return value from `tform`). This way
`transform()` can be safely applied to any function and changes only the ones
for which `tform` is applicable.

Params:

 * tform::Function - transfomrmation function of type (Tape) -> (Tape, Bool)
   where second return value shows whether
"""
function transform(tform::Function, f, args...)
    _, tape = trace(f, args...)
    new_tape, changed = tform(tape)
    return changed ? compile(new_tape; bind=false) : f
end
