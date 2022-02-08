using Documenter
using Yota

makedocs(
    sitename = "Yota",
    format = Documenter.HTML(),
    modules = [Yota],
    pages = [
        "Main" => "index.md",
        "Reference" => "reference.md",
    ],
)

deploydocs(
    repo = "github.com/dfdx/Yota.jl.git",
    devbranch = "main",
)