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
    pb = pullback(identity, (x=1:3,), data)
    AC.check_consistency(pb)
    @test pb.axes === (x=1:3,)
    @test pb.values == data.values

end

end#module
