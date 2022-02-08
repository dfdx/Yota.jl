```@meta
CurrentModule = Yota
```

## Public API

### Tracing

```@docs
trace
isprimitive
record_primitive!
BaseCtx
__new__
```

### Variables

```@docs
Variable
bound
rebind!
rebind_context!
```

### Tape structure

```@docs
Tape
AbstractOp
Input
Constant
Call
Loop
inputs
inputs!
mkcall
```

### Tape transformations

```@docs
push!
insert!
replace!
deleteat!
primitivize!
```

## Tape execution

```@docs
play!
compile
to_expr
```

## Index

```@index
```