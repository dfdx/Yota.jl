# Cookbook

Value and gradient:

```@example
using Yota      # hide

f(x, y) = x^2 + sqrt(y)
val, g = grad(f, 2.0, 3.0)
_, dx, dy = g
```

Gradient tape (useful for further processing):

```@example
using Yota                # hide
f(x, y) = x^2 + sqrt(y)   # hide

tape = gradtape(f, 2.0, 3.0)
```

VJP, value and gradient:

```@example
using Yota                # hide

h(w, b, x) = w * x .+ b

w, b, x = rand(3, 4), rand(3), rand(4, 5)
val, g = grad(h, w, b, x; seed=ones(3, 5))
```

VJP, value and pullback:

```@example
using Yota                # hide
import Yota: YotaRuleConfig, rrule_via_ad

h(w, b, x) = w * x .+ b

w, b, x = rand(3, 4), rand(3), rand(4, 5)
val, pb = rrule_via_ad(YotaRuleConfig(), h, w, b, x)
pb(ones(3, 5))
```

Reset gradient cache:

```@example
import Yota    # hide

Yota.reset!()
```
