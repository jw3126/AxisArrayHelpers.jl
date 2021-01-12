import Interpolations
const ITP = Interpolations

using ConstructionBase: setproperties
using StaticArrays
export create_interpolate
using AxisArrayConversion: to

function create_interpolate(data;
        scheme = ITP.Gridded(ITP.Linear())
    )
    obj = AC.to(NamedTuple, data)
    itp = ITP.interpolate(Tuple(obj.axes), obj.values, scheme)
    return itp
end

function create_interpolate(obj, onoutside; kw...)
    itp = create_interpolate(obj; kw...)
    return ITP.extrapolate(itp, onoutside)
end

function pullback(f, axes, data; kw...)
    AC.check_consistency(data)
    arr    = AC.to(NamedTuple, data)
    axs    = AC.names_axes(axes)
    nt_out = _pullback_namedtuple(f, axes, data; kw...)
    T      = AC.roottype(typeof(data))
    out    = AC.to(T, nt_out)
    return out
end

function _pullback_namedtuple(f, axes::NamedTuple, arr::NamedTuple; kw...)
    dims = map(length, axes)
    vals = similar(data.values, dims)
    out = setproperties(data, (axes=axes, values=vals))
    return _pullback_namedtuple!(f, out, arr; kw...)
end

function pullback!(f, out, data; kw...)
    AC.check_consistency(out)
    AC.check_consistency(data)
    out_nt =AC.to(NamedTuple, out)
    data_nt = AC.to(NamedTuple, data)
    _pullback_namedtuple!(f, nt_out, data_nt; kw...)
    return AC.to(AC.roottype(out), out_nt)
end

function _pullback_namedtuple!(f, out::NamedTuple, arr::NamedTuple; onoutside, kw...)
    itp = create_interpolate(data, onoutside; kw...)
    Threads.@threads for ci in CartesianIndices(out.values)
        pt0 = SVector(map(getindex, out.axes, Tuple(ci)))
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
