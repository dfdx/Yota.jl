# Yötä

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://dfdx.github.io/Yota.jl/dev)
[![Test](https://github.com/dfdx/Yota.jl/actions/workflows/test.yml/badge.svg)](https://github.com/dfdx/Yota.jl/actions/workflows/test.yml)

Yota.jl is a package for reverse-mode automatic differentiation in Julia. The main features are:

* optimized for large inputs and conventional deep learning
* tracer-based with a hackable computational graph (tape)
* supports [ChainRules](https://github.com/JuliaDiff/ChainRules.jl) API
