import Yota.TypeTrie


@testset "TypeTrie" begin
    trie = TypeTrie()
    push!(trie, (typeof(sin), Number))
    @test (typeof(sin), Number) in trie
    @test (typeof(sin), Float64) in trie

    push!(trie, (typeof(Base.broadcasted), Vararg))
    @test (typeof(Base.broadcasted), typeof(sin), Number) in trie
end