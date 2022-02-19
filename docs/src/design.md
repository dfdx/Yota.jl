# Build your own AD

The design of Yota is pretty simple, and the core of the package can be easily reproduced. In this section we discuss some of the key points in Yota implementation by creating a new AD system from scratch.

## Theory

Automatic differentiation is based on two things:

* set of "primivitve" functions such as `+`, `sin`, `sqrt`, etc. with known symbolic derivatives
* [chain rule](https://en.wikipedia.org/wiki/Chain_rule) that describes how to combine them together to obtain the derivative of a complex function

Reverse-mode AD works in two steps:

1. Forward pass, i.e. going from inputs to outputs and evaluating primitives in order.
2. Reverse pass, i.e. going from outputs to inputs and evaluating derivatives.

The reverse pass always starts with a "seed" - a derivative of the output variable w.r.t. itself (and thus equal to `1`). [This answer](https://stats.stackexchange.com/a/235758/3305) provides a complete step-by-step example of a (manual) reverse-mode differentiation for scalars.

While derivatives of scalar-valued functions w.r.t. scalar inputs are also scalars (i.e. have the same "size"), derivative of a vector-valued function `f: Rⁿ → Rᵐ` w.r.t. a vector input is called Jacobian and has size `Rⁿ × Rᵐ`. For large inputs and outputs it's a pretty huge matrix, so in practice it's never calculated as is. Instead, so-called vector-jacobian product (VJP) is used to propagate gradients to function inputs. For example:

```
x = ...        # input
y = f(x)       # both - x and y - are vectors
z = g(y)       # output variabel, must be scalar

dz/dz = 1               # seed
dz/dy = vjp_g(dz/dz)
dz/dx = vjp_f(dz/dy)
```

Here `vjp_f()` essentially calculates `dz/dy * J_f`, but more efficiently.

Sometimes forward and reverse pass share a piece of computation. In this case forward and backward pass can be combined into a single function, returning:

* primal value, i.e. output `y` of the function `f(x)`
* pullback function that takes the gradient w.r.t. to the function output (`dz/dy`) and returns gradients w.r.t. its arguments (`dz/dx`)

See [JAX docs](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#vector-jacobian-products-vjps-aka-reverse-mode-autodiff) for a mathematically robust explanation of VJP and pullbacks.

## Practice

Yota uses two packages under the hood:

* [Umlaut](https://github.com/dfdx/Umlaut.jl) - code tracer that records execution as a list of primitive calls (`Tape`)
* [ChainRules](https://github.com/JuliaDiff/ChainRules.jl) - collection of VJP rules (`rrule`)

The idea is pretty simple. To differentiate a function call `f(args...)` we:

1. Recursively trace function execution to obtain a tape with only primitives.
2. Replace all primitive calls `y = g(xs...)` with `y, pb = rrule(g, xs....)`.
3. Add the seed to the tape.
4. Go backwards, at each step invoking pullbacks to propagate function output gradient to its arguments, i.e. `dxs = pb(dy)`.

We will use the following simple function as an example:

```julia
g(a, b) = a * b
h(a) = sin(a)

f(x1, x2) = g(x1, x2) + h(x1)
args = (2.0, 3.0)
```

### Trace the function

First of all, we need to let Umlaut know what functions to treat as primitives. By default, Umlaut records everything in Julia's built-in modules such as `Base` and `Core`, but we also want to record functions for which `rrule` is defined. To do so, we need to create a new tracing context and overload `Umlaut.isprimitive` method for it:

```julia
using Umlaut
using ChainRules: rrule

struct GradCtx
    pullbacks::Dict{Variable,Variable}    # ignore for now
    derivs::Dict{Variable,Variable}       # ignore for now
end
GradCtx() = GradCtx(Dict(), Dict())

function Umlaut.isprimitive(::GradCtx, f, args...)
    Ts = [a isa DataType ? Type{a} : typeof(a) for a in (f, args...)]
    # return type of rrule(f, args...) will be nothing only if there's no
    # rrule for this function signature
    if Core.Compiler.return_type(rrule, Ts) !== Nothing
        return true
    else
        return false
    end
end
```

Now we can check if the function call is traced correctly:

```julia
val, tape = trace(f, args...; ctx=GradCtx())

# output
(6.909297426825682, Tape{GradCtx}
  inp %1::typeof(f)
  inp %2::Float64
  inp %3::Float64
  %4 = *(%2, %3)::Float64
  %5 = sin(%2)::Float64
  %6 = +(%4, %5)::Float64
)
```

Woohoo! Despite nested calls to `h()` and `g()`, Umlaut correctly recorded only primirives to the tape.

### Replace f(args...) with rrule(f, args...)

Umlaut has a function `replace!()` that can do the replacement. But in this case we will calculate both `f(args...)` and `rrule(f, args...)`, which is a double work. Instead we will do a slightly better thing and implement the replacement right during the tracing. To do so, we need to overload `Umlaut.record_primitive!()` as follows:


```julia
const V = Umlaut.Variable

"""
    record_primitive!(tape::Tape{GradCtx}, v_fargs...)

Replace ChainRules primitives `f(args...)` with a sequence:

    rr = push!(tape, mkcall(rrule, f, args...))   # i.e. rrule(f, args...)
    val = push!(tape, mkcall(getfield, rr, 1)     # extract value
    pb = push!(tape, mkcall(getfield, rr, 2)      # extract pullback
"""
function Umlaut.record_primitive!(tape::Tape{GradCtx}, v_fargs...)
    # v_xxx refer to instances of Umlaut.Variable or constants
    # e.g. v_fargs = [+, V(1), 2.0]
    v_f, v_args... = v_fargs
    # fargs are actial values of a function and arguments
    f, args... = [v isa V ? tape[v].val : v for v in v_fargs]
    # record rrule(v_f, v_args...)
    v_rr = push!(tape, mkcall(rrule, v_f, v_args...))
    # get the output and the pullback as separate operations on the tape
    v_val = push!(tape, mkcall(getfield, v_rr, 1))
    v_pb = push!(tape, mkcall(getfield, v_rr, 2))
    # store the mapping from the value var to pullback var
    # in the tape's context
    tape.c.pullbacks[v_val] = v_pb
    # return value var, the same as if we recorded just f(args...)
    return v_val
end
```

Just as the docstring explains it, for each primitive call `f(args...)` instead of recording it directly we record a sequence of three calls - one for `rrule(f, args...)` and two to destructure its result. We also save the mapping `value -> pullback` to the `GradCtx.pullbacks` field, which we prudently added to the context object.

Let's see what we get now:

```julia
val, tape = trace(f, args...; ctx=GradCtx())

# output
(6.909297426825682, Tape{GradCtx}
  inp %1::typeof(f)
  inp %2::Float64
  inp %3::Float64
  %4 = rrule(*, %2, %3)::Tuple{Float64, ChainRules.var"#times_pullback2#1214"{Float64, Float64}}
  %5 = getfield(%4, 1)::Float64
  %6 = getfield(%4, 2)::ChainRules.var"#times_pullback2#1214"{Float64, Float64}
  %7 = rrule(sin, %2)::Tuple{Float64, ChainRules.var"#sin_pullback#1175"{Float64}}
  %8 = getfield(%7, 1)::Float64
  %9 = getfield(%7, 2)::ChainRules.var"#sin_pullback#1175"{Float64}
  %10 = rrule(+, %5, %8)::Tuple{Float64, ChainRules.var"#+_pullback#1203"{Bool, Bool, ChainRulesCore.ProjectTo{Float64, NamedTuple{(), Tuple{}}}, ChainRulesCore.ProjectTo{Float64, NamedTuple{(), Tuple{}}}}}
  %11 = getfield(%10, 1)::Float64
  %12 = getfield(%10, 2)::ChainRules.var"#+_pullback#1203"{Bool, Bool, ChainRulesCore.ProjectTo{Float64, NamedTuple{(), Tuple{}}}, ChainRulesCore.ProjectTo{Float64, NamedTuple{(), Tuple{}}}}
)
```
That's pretty verbose. Let's make it bit more readable:

```julia
Umlaut.show_format!(:compact)   # switch back with Umlaut.show_format!(:plain)

println(tape)

# output
Tape{GradCtx}
  inp %1::typeof(f)
  inp %2::Float64
  inp %3::Float64
  %5, %6 = [%4] = rrule(*, %2, %3)
  %8, %9 = [%7] = rrule(sin, %2)
  %11, %12 = [%10] = rrule(+, %5, %8)
```

### Add seed to the tape

As discussed earlier, seed is the derivative of the ouput variable w.r.t. itself, which is 1. We can record this constant to the tape as follows:

```julia
dy = push!(tape, Constant(1))
```

We also hold mapping from a (primal) variable on the tape to its derivative variable in the `GradCtx.derivs` field. The very first pair we must add is from the tape result to the seed:

```julia
tape.c.derivs[tape.result] = dy
```

### Go backwards, propagate derivatives

During the reverse pass we go backwards from the tape result variable (`%11` in our case) to its inputs, at each step calling the pullback and updating the derivatives of a function arguments:


```julia
# y is the output variable of the current Call operation
# i.e. we assume call `y = fn(xs...)`
function step_back!(tape::Tape, y::Variable)
    # dy - output of the tape result w.r.t. y
    dy = tape.c.derivs[y]
    # extract pullback for y
    pb = tape.c.pullbacks[y]
    # record a call pb(dy)
    dxs = push!(tape, mkcall(pb, dy))
    # we don't actually need to go through call to `rrule` itself
    # instead we propage derivs to original f(args...)
    rr = tape[y].args[1]
    y_fargs = tape[rr].args
    # for each argument in f(args...) (including `f` itself)
    for (i, x) in enumerate(y_fargs)
        # if it's not a constant
        if x isa V
            # set the derivative var as getfield(xs, i)
            dx = push!(tape, mkcall(getfield, dxs, i))
            # WARNING: this is simplified version,
            # won't work for multipath derivatives!
            tape.c.derivs[bound(tape, x)] = bound(tape, dx)
        end
    end
end
```

Say, we are analyzing the last function call - `%11, %12 = [%10] = rrule(+, %5, %8)`:

* `y` points to `%11`
* `dy` points to our seed - `%13`
* the pullback `pb` is stored in `%12`
* we record `dxs = pb(dy)`
* and update derivatives for each argument in the original call `+(%5, %8)`

Note that bold warning at the end: for illustrative purposes we _set_ the mapping from the variable to its derivative, but in practice the same variable may influence the result in more than one way (e.g. `x1` influences the result via both - `g()` and `h()`), and thus we need to _set or add_ to the derivative. Let's fix it:

```julia
import ChainRules: Tangent


getderiv(tape::Tape, v::Variable) = get(tape.c.derivs, bound(tape, v), nothing)
setderiv!(tape::Tape, x::Variable, dx::Variable) = (
    tape.c.derivs[bound(tape, x)] = bound(tape, dx)
)
hasderiv(tape::Tape, v::Variable) = getderiv(tape, v) !== nothing


function set_or_add_deriv!(tape::Tape, x::Variable, dx::Variable)
    if !hasderiv(tape, x)
        setderiv!(tape, x, dx)
    else
        old_dx = getderiv(tape, x)
        if tape[dx].val isa Tangent || tape[old_dx].val isa Tangent
            new_dx = push!(tape, mkcall(+, dx, old_dx))
        else
            new_dx = push!(tape, mkcall(broadcast, +, dx, old_dx))
        end
        setderiv!(tape, x, new_dx)
    end
end


function step_back!(tape::Tape, y::Variable)
    dy = tape.c.derivs[y]
    pb = tape.c.pullbacks[y]
    dxs = push!(tape, mkcall(pb, dy))
    rr = tape[y].args[1]
    y_fargs = tape[rr].args
    for (i, x) in enumerate(y_fargs)
        if x isa V
            dx = push!(tape, mkcall(getfield, dxs, i))
            # this line has changed
            set_or_add_deriv!(tape, x, dx)
        end
    end
end
```

The backward pass itself is simple:

```julia
function back!(tape::Tape; seed=1)
    # z - final variable (usually a loss)
    # y - resulting variable of current op
    # x - dependencies of y
    # dy - derivative of z w.r.t. y
    z = tape.result
    # set seed and the first derivative
    dy = push!(tape, Constant(seed))
    tape.c.derivs[z] = dy
    # the reverse pass, literally
    for i=length(tape)-1:-1:1
        y = bound(tape, V(i))
        op = tape[y]
        # note: skipping rrule() and pullbacks
        if op isa Call && op.fn != rrule && !in(y, values(tape.c.pullbacks))
            step_back!(tape, y)
        end
    end
end

back!(tape)

tape.c.derivs

# output
Dict{Variable, Variable} with 5 entries:
  %8  => %16
  %3  => %21
  %2  => %20
  %5  => %15
  %11 => %13
```

`tape.c.derivs` now contains a map from all primal variables to their derivative variables. These include derivatives w.r.t. function inputs `%2` and `%3`. For example:

```julia
# bound variables for function inputs
_, x1, x2 = inputs(tape)

# derivaitves w.r.t. these inputs
dx1 = tape.c.derivs[x1]
dx2 = tape.c.derivs[x2]

# values of these derivatives
tape[dx1].val
tape[dx2].val
```

Finally, let's add a bit of post-processing and wrap it up into a single convenient function:

```julia
import ChainRules: ZeroTangent, unthunk


function grad(f, args...)
    _, tape = trace(f, args...; ctx=GradCtx())
    back!(tape)
    # add a tuple of (val, (gradients...))
    deriv_vars = [hasderiv(tape, v) ? getderiv(tape, v) : ZeroTangent() for v in inputs(tape)]
    deriv_tuple = push!(tape, mkcall(tuple, deriv_vars...))
    # unthunk results
    deriv_tuple_unthunked = push!(tape, mkcall(map, unthunk, deriv_tuple))
    new_result = push!(tape, mkcall(tuple, tape.result, deriv_tuple_unthunked))
    # set result
    tape.result = new_result
    return tape[tape.result].val
end

grad(f, 2.0, 3.0)   # (6.909297426825682, (ZeroTangent(), 2.5838531634528574, 2.0))

# verify using numerical differentiation
import Yota.ngradient

ngradient(f, 2.0, 3.0)  # (2.583853163452614, 2.0000000000001696)
```

And we are done!

### What's not covered

Although the code above is a valid AD system suitable for most functions, it doesn't cover many corner cases, including:

* functions with keyword arguments
* non-differentiable paths
* object constructors
* some (often purposely) missing [`rrule`s](https://github.com/dfdx/Yota.jl/blob/0103f6883df49b8bed256deaea41a0a0b65b0a2d/src/rulesets.jl)
* custom seeds
* tape caching, etc.