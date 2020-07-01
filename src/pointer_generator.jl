include("linear.jl")

struct PointerGenerator
    projection::Linear          # decoderhiddens(B*Ty, Hy) -> vocabprobs(B,Ty, vocabsize)
    linearpointer::Linear       # decoderhiddens(B*Ty, Hy) -> pointerprobs(B,Ty, 3): p_copy_source,p_copy_target, p_generate
end

# The `PointerGenerator` constructor takes the following arguments:
# * `hidden`: size of the hidden vectors of the decoder
# * `vocab_size`: number of decoder tokens vocabsize
function PointerGenerator(hidden::Int, vocabsize::Int)
    projection = Linear(hidden, vocabsize)
    linear_pointer = Linear(hidden, 3)
    PointerGenerator(projection, linear_pointer) 
end

#=
@testset "Testing PointerGenerator" begin
    H = 512
    pg = PointerGenerator(H, vocabsize, 1e-20)
    @test size(pg.projection.w) == (vocabsize, H)
end
=#
