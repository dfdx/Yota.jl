
@testset "functionresolver" begin
    rsv = Yota.FunctionResolver{Symbol}()
    rsv[(typeof(sin), Vararg)] = :Vararg
    rsv[(typeof(sin), Float64)] = :Float64
    rsv[(typeof(sin), Real)] = :Real
    rsv[(typeof(sin), Number)] = :Number
    Yota.order!(rsv)

    @test rsv[(typeof(sin), Float64)] == :Float64
    @test rsv[(typeof(sin), Float32)] == :Real
    @test rsv[(typeof(sin), Float64, Float64)] == :Vararg

    # non-matching signature
    rsv[(typeof(cos), Number)] = :CosineNumber
    @test rsv[(typeof(cos), String)] === nothing
end