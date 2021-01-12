module TestInterpolations
using Test
using AxisArrayHelpers
const AH = AxisArrayHelpers
import AxisKeys
const AK = AxisKeys


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

end

end#module
