# ### Embed
#
# `Embed` is a layer that takes an integer or an array of integers as input, uses them as
# column indices to lookup embeddings in its parameter matrix `w`, and returns these columns
# packed into an array. If the input size is `(X1,X2,...)`, the output size will be
# `(C,X1,X2,...)` where C is the columns size of `w` (which Julia will automagically
# accomplish if you use the right indexing expression). Please review [Array
# indexing](https://docs.julialang.org/en/v1/manual/arrays/#man-array-indexing-1) and the
# Knet `param` function to implement this layer.

using Knet, Test, LinearAlgebra
struct Embed; w; end

function Embed(vocabsize::Int, embedsize::Int)
    Embed(param(embedsize,vocabsize))
end

function (l::Embed)(x)
    l.w[:,x]
end

#=
#input -> T x B
#output -> embedsize x T x B
@info "Testing Embed"
vocabsize = 100; embedsize = 10; T=7; B=64;
Knet.seed!(1)
embed = Embed(vocabsize, embedsize)
input = rand(1:100, T, B)
output = embed(input)
@test size(output) == (embedsize, T, B)
=#