using Documenter, DiskArrays
using DiskArrays.TestTypes

makedocs(;
    modules=[DiskArrays],
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true"),
    authors="Fabian Gans",
    sitename="DiskArrays.jl",
    pages=Any["index.md", "Backend System" => "backend.md"],
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(; repo="github.com/JuliaIO/DiskArrays.jl.git")
