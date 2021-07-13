module TestCUDA

import LinearInterpolations as LI
import AxisArrayHelpers as AH
using CUDA
using Test

@testset "TextureEx" begin
    @testset "1d" begin
        axes_from = (CuVector(Float32[1,2,3,3.1, 5]),)
        axes_to = (1f0:4f0,)
        vals_to = CuVector(Float32[10,20,30,40])
        data_to = (axes=axes_to, values=vals_to)
        res_tex = AH.pullback(identity, axes_from, data_to, executor=AH.TextureEx(), extrapolate=LI.Replicate())
        @test only(res_tex.axes) == only(axes_from)
        @test res_tex.values isa CUDA.CuVector
        @test Vector(res_tex.values) ≈ [10,20,30,31,40] rtol=1e-3
    end
    @testset "2d" begin
        axes_from = (CuVector(Float32[1,2,3,3.1, 4, 5]),)
        axes_to = (-1f0:2:1f0, 1f0:4f0)
        vals_to = CuMatrix(
            Float32[
             10  20  30  40;
            100 200 300 400
        ])
        function f(pt)
            x, = pt
            if isodd(round(Int, x))
                (-1f0, x)
            else
                (1f0, x)
            end
        end

        data_to = (axes=axes_to, values=vals_to)
        res_tex = AH.pullback(f, axes_from, data_to, executor=AH.TextureEx(), extrapolate=LI.Replicate())
        @test Tuple(res_tex.axes) == axes_from
        @test res_tex.values isa CUDA.CuVector
        @test Vector(res_tex.values) ≈ [10,200,30,31,400, 40] rtol=1e-3
    end

    @testset "3d" begin
        function f(pt)
            x,y,z=pt
            z+sin(x), y+cos(z), x+1f-3*sqrt(abs(y*z))
        end
        for _ in 1:10
            axes_from = (-10f0:2:10f0, -10f0:1:12f0, -12f0:3:9f0)
            axes_to = (-8:1f0:8, -9:2f0:11f0, -20:1f0:10)
            vals_to = randn(Float32, map(length, axes_to))
            data_to = (axes=axes_to, values=vals_to)
            res_cpu = AH.pullback(f, axes_from, data_to, executor=AH.ThreadsEx(), extrapolate=LI.Replicate())
            res_tex = AH.pullback(f, axes_from, data_to, executor=AH.TextureEx(), extrapolate=LI.Replicate())
            @test res_cpu.axes === res_tex.axes
            @test res_tex.values isa CUDA.CuArray
            @test res_cpu.values ≈ Array(res_tex.values) rtol=5e-3
        end
    end
end

end#module
