# Iteration is column-major, so `Base.Generator` and `collect` place disk array values correctly.
# The macro remains so that packages calling it keep loading.

macro implement_generator(t)
    Base.depwarn(
        "`@implement_generator` is deprecated and does nothing: generators work on disk arrays without it.",
        Symbol("@implement_generator"),
    )
    return nothing
end
