import CUDA
import Adapt
"""
    struct TextureEx end

Executes the computation using CUDA Texture arrays.
Computations are inaccurate, but extremly fast.
"""
struct TextureEx end

struct AxesProduct{T,N,A} <: AbstractArray{T,N}
    axes::A
    function AxesProduct(axes::NTuple{N,AbstractVector}) where {N}
        T = Tuple{map(eltype, axes)...}
        A = typeof(axes)
        return new{T,N,A}(axes)
    end
end
Base.size(o::AxesProduct) = map(length, o.axes)
Adapt.@adapt_structure AxesProduct

Base.@propagate_inbounds function Base.getindex(o::AxesProduct, I::Integer...)
    map(getindex, o.axes, Tuple(I))
end

Base.IndexStyle(::Type{<:AxesProduct}) = IndexCartesian()

resolve_address_mode(::LI.Replicate) = CUDA.ADDRESS_MODE_CLAMP
resolve_address_mode(::LI.Reflect)   = CUDA.ADDRESS_MODE_MIRROR

@noinline function check_data_to(data_to::NamedTuple)
    for ax in Tuple(data_to.axes)
        @argcheck ax isa AbstractRange
        @argcheck eltype(ax) === Float32
        @argcheck length(ax) >= 2
        @argcheck issorted(ax)
    end
end

function _pullback_namedtuple(f, axes_from, data_to, ::TextureEx; kw...)
    out_vals = CUDA.CuArray{Float32}(undef, length.(Tuple(axes_from)))
    out = (axes=axes_from, values=out_vals)
    _pullback_namedtuple!(f, out, data_to, TextureEx(); kw...)
end

function _pullback_namedtuple!(f, out, data_to, ::TextureEx;
                               extrapolate=LI.Replicate())
    check_data_to(data_to)
    @argcheck extrapolate === LI.Replicate() # others are not handled properly by CUDA.jl
    AC.check_consistency(out)
    address_mode = resolve_address_mode(extrapolate)
    vals_tex = CUDA.CuTexture(CUDA.CuTextureArray(data_to.values);
        address_mode,
        interpolation=CUDA.LinearInterpolation()
    )
    pts = AxesProduct(Tuple(out.axes))
    axes_to = Tuple(data_to.axes)
    broadcast!(out.values, pts, Ref(vals_tex)) do pt, tex
        pt = Tuple(f(pt))
        pt = indexcoords(pt, axes_to)
        tex[pt...]
    end
    out
end

indexcoords(pt, axes) = map(indexcoord, pt, axes)
function indexcoord(x, r)
    (x - first(r)) / step(r) + firstindex(r)
end
