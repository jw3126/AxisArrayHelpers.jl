module TestInterpolations
using Test
using AxisArrayHelpers
const AH = AxisArrayHelpers
import AxisKeys
const AK = AxisKeys
import Interpolations
const ITP = Interpolations
using AxisArrayConversion
const AC = AxisArrayConversion
using CoordinateTransformations


@testset "create_interpolate" begin
    data = (axes=(1:3,), values=[10,11,13])
    itp = @inferred create_interpolate(data)
    @test itp(1  ) ≈ 10
    @test itp(1.2) ≈ 10.2
    @test itp(2.2) ≈ 11.4
    @test_throws BoundsError itp(0.9)

    datas = [
        AH.to(AK.KeyedArray, data),
        AH.to(Array, data),
        AH.to(NamedTuple, data),
    ]

    for data in datas
        itp = @inferred create_interpolate(data)
        @test itp(1  ) ≈ 10
        @test itp(1.2) ≈ 10.2
        @test itp(2.2) ≈ 11.4
        @test_throws BoundsError itp(0.9)
    end

    data = (axes=(-2:2.0, ), values=[0,1,0,0,10.0])
    itp = @inferred create_interpolate(data, onoutside=ITP.Flat())
    @test itp(-2) === 0.0
    @test itp(-100) === 0.0
    @test itp(100) === 10.0

    data = (axes=(-2.0:2, ), values=[0,1,0,0,10.0])
    itp = @inferred create_interpolate(data, scheme=ITP.Gridded(ITP.Constant()))
    @test itp(-1.4) === 1.0
    @test itp(1.9) === 10.0
end

@testset "pullback" begin
    data = (axes=(1:3,), values=[2,0,10])
    pb = @inferred pullback(identity, (x=1:3,), data)
    AC.check_consistency(pb)
    @test pb.axes === (x=1:3,)
    @test pb.values == data.values

    data = (axes=(a=1:3,), values=[2,0,10.0])
    out = deepcopy(data)
    out.values .= NaN
    pb = @inferred pullback!(identity, out, data)
    @test pb === out
    @test out == data

    data = (axes=(a=1:3,), values=[2,0,10])
    pb_axes = (a=(1/2)*(1:3),)
    #pb_axes = (a=[2,2],)
    pb = @inferred pullback(pt -> 2pt, pb_axes, data, onoutside=ITP.Flat())
    AC.check_consistency(pb)
    @test pb.axes === pb_axes
    @test pb.values == data.values
end

@testset "pullback_axes" begin
    #A = [0 5 0; 1 0 0; 0 0 -1]
    #axes = (1:10, 2:5, 4:8)
    #N = length(axes)

    #sp = splitperm(A)
    #A2 = matrix_from_splitperm(sp)
    #@test A ≈ A2
    #Ainv = matrix_from_splitperm(inv_splitperm(sp))
    #@test A * Ainv ≈ LinearAlgebra.I

    #function permute(v, perm)
    #    permute!(copy(v), collect(perm))
    #end

    #v = randn(3)
    using Test
    @test (0:9, 0:18)      === AH.pullback_axes(Translation([1,2]), (x=1:10, y=2:20))
    @test_throws ArgumentError AH.pullback_axes(Translation([1,2,3]), (x=1:10, y=2:20))

    @test (2:2:20.0, 1:10.0)  === AH.pullback_axes(LinearMap([0 1; 1 0]), (1:10, 2:2:20))
    @test (0.5:0.5:5, 1:5.0)  === AH.pullback_axes(LinearMap([2 0; 0 1]), (1:10, 1:5))
    @test (-5:1:-1.0, 1:1:5.0) == AH.pullback_axes(LinearMap([-1 0; 0 1]), (1:5, 1:5))

    for lin in [
            LinearMap([1 0 0; 0 1 0; 0 0 1]),
            LinearMap([randn() 0 0; 0 randn() 0; 0 0 randn()]),
            LinearMap([0 1 0; -1 0 0; 0 0 2]),
            LinearMap([0 1 0; 0 0 1; 1 0 0]),
            LinearMap([0 randn() 0; 0 0 randn(); randn() 0 0]),
        ]
        trans = Translation(randn(3))
        trafo = trans ∘ lin
        axs = (1:5, -2:2:4, 1:10)
        axs_pb  = AH.pullback_axes(trafo, axs)
        axs_pb1 = AH.pullback_axes(trans, axs)
        axs_pb2 = AH.pullback_axes(lin, axs_pb1)
        @test all(length.(axs_pb2) .== length.(axs_pb))
        @test all(axs_pb2 .≈ axs_pb)
    end
end

end#module
