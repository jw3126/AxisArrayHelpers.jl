using AxisArrayHelpers
using Documenter

makedocs(;
    modules=[AxisArrayHelpers],
    authors="Jan Weidner <jw3126@gmail.com> and contributors",
    repo="https://github.com/jw3126/AxisArrayHelpers.jl/blob/{commit}{path}#L{line}",
    sitename="AxisArrayHelpers.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jw3126.github.io/AxisArrayHelpers.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jw3126/AxisArrayHelpers.jl",
)
