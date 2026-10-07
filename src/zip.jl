# Iteration is column-major, so `Base.Zip` pairs disk array values correctly.
# These macros remain so that packages calling them keep loading.

macro implement_zip(t)
    Base.depwarn(
        "`@implement_zip` is deprecated and does nothing: `zip` works on disk arrays without it.",
        Symbol("@implement_zip"),
    )
    return nothing
end

macro implement_diskarray_skip_zip(t)
    Base.depwarn(
        "`@implement_diskarray_skip_zip` is deprecated, use `@implement_diskarray`.",
        Symbol("@implement_diskarray_skip_zip"),
    )
    return :(@implement_diskarray $(esc(t)))
end
