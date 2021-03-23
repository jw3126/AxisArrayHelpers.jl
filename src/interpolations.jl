import Interpolations
const ITP = Interpolations

using ArgCheck
using CoordinateTransformations
using ConstructionBase: setproperties
using StaticArrays
using AxisArrayConversion: to
using LinearAlgebra

export create_interpolate
export pullback, pullback!

struct DefaultOnOutside end

function _create_interpolate(data, scheme, onoutside::DefaultOnOutside)
    obj = AC.to(NamedTuple, data)
    axes = Tuple(obj.axes)
    itp = ITP.interpolate(axes, obj.values, scheme)
    return itp
end

function _create_interpolate(data, scheme, onoutside)
    itp = _create_interpolate(data, scheme, DefaultOnOutside())
    return ITP.extrapolate(itp, onoutside)
end

function create_interpolate(data;
        scheme = ITP.Gridded(ITP.Linear()),
        onoutside = DefaultOnOutside(),
    )
    return _create_interpolate(data, scheme, onoutside)
end

function pullback(f, axes, data; kw...)
    AC.check_consistency(data)
    data_nt    = AC.to(NamedTuple, data)
    axes_nt    = AC.name_axes(axes)
    out_nt = _pullback_namedtuple(f, axes_nt, data_nt; kw...)
    T      = AC.roottype(typeof(data))
    out    = AC.to(T, out_nt)
    return out
end

function _pullback_namedtuple(f, axes::NamedTuple, data::NamedTuple; kw...)
    dims = map(length, Tuple(axes))
    T = float(eltype(data.values))
    vals = similar(data.values, T, dims)
    out  = (axes=axes, values=vals)
    return _pullback_namedtuple!(f, out, data; kw...)
end

function pullback!(f, out, data; kw...)
    AC.check_consistency(out)
    AC.check_consistency(data)
    out_nt  = AC.to(NamedTuple, out)
    data_nt = AC.to(NamedTuple, data)
    _pullback_namedtuple!(f, out_nt, data_nt; kw...)
    T = AC.roottype(typeof(out))
    return AC.to(T, out_nt)
end

@noinline function _pullback_namedtuple!(f, out::NamedTuple, data::NamedTuple; kw...)
    itp = create_interpolate(data; kw...)
    Threads.@threads for ci in CartesianIndices(out.values)
        pt0 = SVector(map(getindex, Tuple(out.axes), Tuple(ci)))
        pt = f(pt0)
        out.values[ci] = apply_interpolate(itp, pt)
    end
    return out
end

function apply_interpolate(f, pt::SVector{1})
    # needed for inference
    x, = pt
    f(x)
end
function apply_interpolate(f, pt::SVector{2})
    x,y = pt
    f(x,y)
end
function apply_interpolate(f, pt::SVector{3})
    x,y,z = pt
    f(x,y,z)
end
function apply_interpolate(f, pt::SVector{4})
    w,x,y,z = pt
    f(w,x,y,z)
end
function apply_interpolate(f, pt)
    f(pt...)
end

function pullback_axes(f, axes)
    pullback_axes_tuple(f, Tuple(axes))
end

function pullback_axes_tuple(trans::Translation, axes::NTuple{N,Any}) where {N}
    @argcheck length(trans.translation) == length(axes)
    map(axes, NTuple{N}(trans.translation)) do ax, x
        ax .- x
    end
end

function norm0(arr; atol=0, rtol=sqrt(eps(float(eltype(arr)))))
    m = norm(arr, Inf)
    ret = 0
    thresh = max(atol, rtol*m)
    for x in arr
        if norm(x) >= thresh
            ret += 1
        end
    end
    ret
end

function pullback_axes_tuple(am::AffineMap, axes)
    axes = pullback_axes_tuple(Translation(am.translation), axes)
    axes = pullback_axes_tuple(LinearMap(am.linear), axes)
end

function pullback_axes_tuple(am::LinearMap, axes)
    sp = splitperm(am.linear)
    isp = inv_splitperm(sp)
    forward_axes_splitperm(isp, axes)
end

function splitperm(A::AbstractMatrix)
    @argcheck size(A,1) == size(A,2)
    N = size(A,1)
    @argcheck norm0(A) == N
    colmaxima = NTuple{N}(findmax(abs.(A), dims=(1,))[2])
    perm = map(colmaxima) do ci
        Tuple(ci)[1]
    end

    scalings = map(colmaxima) do ci
        A[ci]
    end
    (;perm, scalings)
end

function forward_axes_splitperm(sp, axes)
    axes = axes .* sp.scalings
    axes = getindex.(Ref(axes), sp.perm)
    map(axes) do ax
        first(ax) > last(ax) ? reverse(ax) : ax
    end
end

function matrix_from_splitperm(sp)
    N = length(sp.perm)
    T = eltype(sp.scalings)
    ret = zeros(T, N,N)
    for (i,j, s) in zip(1:N,sp.perm, sp.scalings)
        ret[j,i] = s
    end
    ret
end

function inv_splitperm(sp)
    perm = invperm(sp.perm)
    scalings = map(sp.scalings) do s
        1/s
    end
    scalings = getindex.(Ref(scalings), sp.perm)
    (;perm, scalings)
end

################################################################################
##### common_ground
################################################################################
function common_grid(grids)
    n = length(first(grids))
    axes_values = map(ntuple(identity, n)) do i
        axes_i = map(grids) do grid
            @argcheck length(grid) == n
            grid[i]
        end
        return common_axis(axes_i)
    end
    grid1 = first(grids)
    if grid1 isa Tuple
        axes_values
    else
        NamedTuple{propertynames(grid1)}(axes_values)
    end
end

function common_axis(axis)
    x_lo, x_hi = common_axis_bounds(axis)
    ret = filter(first(axis)) do x
        x_lo < x < x_hi
    end
    pushfirst!(ret, x_lo)
    push!(ret, x_hi)
    return ret
end

function common_axis_bounds(axs) x_lo = maximum(minimum, axs)
    x_hi = minimum(maximum, axs)
    return x_lo, x_hi
end

"""
    new_axis_arrays = common_ground(axis_arrays)

Take a collection of axis array like objects, with possibly distinct but overlapping
axes and shrink them such that their axes coincide.
"""
function common_ground(arrs)
    nts = map(arr -> AxisArrayConversion.to(NamedTuple, arr), arrs)
    axs = common_grid(map(nt -> nt.axes, nts))

    map(arrs, nts) do orig, nt
        nt_common = pullback(identity, axs, nt)
        T = AxisArrayConversion.roottype(typeof(orig))
        AxisArrayConversion.to(T, nt_common)
    end
end

function flip_decreasing_axes(nt::NamedTuple)
    flip_needs = map(nt.axes) do ax
        last(ax) < first(ax)
    end
    if any(flip_needs)
        axes_new = map(nt.axes, flip_needs) do ax, needs_flip
            needs_flip ? reverse(ax) : ax
        end
        inds_new = map(Base.axes(nt.values), flip_needs) do inds, needs_flip
            needs_flip ? reverse(inds) : inds
        end
        vals_new = nt.values[inds_new...]
        return (axes=axes_new, values=vals_new)
    else
        return nt
    end
end

function flip_decreasing_axes(arr)
    nt = AxisArrayConversion.to(NamedTuple, arr)
    nt_out = flip_decreasing_axes(nt)
    T = AxisArrayConversion.roottype(typeof(arr))
    AxisArrayConversion.to(T, nt_out)
end
