using Knet
include("vocab.jl")
include("embed.jl")

# Define a 1D convolutional layer:
struct Conv; w; b; f; p; end
(c::Conv)(x) = c.f.(maximum(conv4(c.w, dropout(x,c.p)) .+ c.b, dims=2))
Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=tanh;pdrop=0) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f, pdrop)

struct CharCNNEncoder
    convlayers
    embed_size           
    vocab                # To convert input int tokens to original string tokens 
    charvocab            # To convert the characters in string tokens to character tokens
    charembed            # To get char embeddings
    tokenizer
    ngram_sizes
end

function CharCNNEncoder(embed_size::Int, numfilters::Int, vocab, charvocab, ngram_sizes=[2, 3, 4, 5], conv_activation = tanh, tokenizer=x->split(x,""))
    convlayers = [Conv(embed_size, size, 1, numfilters) for size in ngram_sizes]
    charembed = Embed(charvocab.vocabsize, embed_size)
    return CharCNNEncoder(convlayers, embed_size, vocab, charvocab, charembed, tokenizer, ngram_sizes)
end

function (ce::CharCNNEncoder)(tokens)
    # input: tokens: [numtokens, batchsize]
    # output: numfilters*len(convlayers), numtokens, batchsize
        
    # Retreive original string tokens from the index tokens
    tokens_str = [ce.vocab.idx_to_token[idx] for idx in tokens]      # numtokens, batchsize
    
    # Convert string tokens to character tokens
    max_len = maximum([length(token) for token in tokens_str])
    # We can't have less tokens than the largest filter width
    max_len = max(max_len, maximum(ce.ngram_sizes))
    
    tokens_char = []
    padtoken = ce.charvocab.token_to_idx[ce.charvocab.padtoken]
    for str in tokens_str
        cur_char_tokens = [padtoken for i in 1:max_len]
        for (idx, charidx) in enumerate(vocab_indexsequence(ce.charvocab,ce.tokenizer(str)))
            cur_char_tokens[idx] = charidx
        end
        push!(tokens_char, cur_char_tokens)
    end
    
    tokens_char = hcat(tokens_char...)         # max_len , (numtokens*batchsize)
    
    token_embeds = ce.charembed(tokens_char)   # embed_size, max_len, (numtokens*batchsize)
    
    token_embeds = reshape(token_embeds, ce.embed_size, max_len, 1, :) # embed_size, max_len, 1, (numtokens*batchsize)
    
    outputs = []
    
    for layer in ce.convlayers
        convout = layer(token_embeds)            # 1, 1, numfilters, (numtokens*batchsize)
        convout = reshape(convout, :, size(token_embeds, 4))
        push!(outputs, convout)
    end
    
    out = cat(outputs..., dims=1)                # numfilters*len(convlayers), (numtokens*batchsize)

    reshape(out, :, size(tokens)...)             # numfilters*len(convlayers), numtokens, batchsize
end



ipsum="""Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Adipiscing enim eu turpis egestas pretium aenean pharetra magna ac. Varius sit amet mattis vulputate enim nulla. Fringilla phasellus faucibus scelerisque eleifend donec pretium vulputate sapien. At consectetur lorem donec massa. Consectetur libero id faucibus nisl tincidunt eget. Neque sodales ut etiam sit amet nisl. Mauris pharetra et ultrices neque. Blandit volutpat maecenas volutpat blandit aliquam etiam erat. Donec ac odio tempor orci. Amet consectetur adipiscing elit duis tristique sollicitudin nibh sit. Luctus venenatis lectus magna fringilla urna. Dignissim convallis aenean et tortor at. Felis eget nunc lobortis mattis aliquam faucibus purus in massa. Est ante in nibh mauris cursus mattis molestie a iaculis. Sit amet massa vitae tortor condimentum lacinia. Morbi leo urna molestie at elementum eu facilisis sed odio. Nunc congue nisi vitae suscipit tellus mauris a diam maecenas. Id interdum velit laoreet id donec ultrices tincidunt arcu non.

A cras semper auctor neque vitae tempus quam pellentesque. Non tellus orci ac auctor. Nunc aliquet bibendum enim facilisis gravida neque convallis a. Pulvinar sapien et ligula ullamcorper malesuada proin libero. Nunc sed augue lacus viverra vitae congue eu. Eu scelerisque felis imperdiet proin. Facilisi nullam vehicula ipsum a. Sed elementum tempus egestas sed sed risus pretium quam. Vivamus arcu felis bibendum ut. Ac turpis egestas integer eget aliquet nibh praesent. Nisl nunc mi ipsum faucibus vitae aliquet nec ullamcorper sit. Accumsan sit amet nulla facilisi morbi tempus iaculis urna id. Erat pellentesque adipiscing commodo elit at imperdiet dui accumsan."""


@testset "Testing CharCNN Encoder" begin
    ngram_sizes=[2, 3]
    srcvocab = Vocab(split(ipsum))
	srccharvocab = Vocab(split(ipsum, ""))
	# Embed size: 24, numfilters=5
	cenc = CharCNNEncoder(24, 5, srcvocab, srccharvocab, ngram_sizes)
	# Number of tokens to use
	N = 23
	inptoks = [srcvocab.token_to_idx[tok] for tok in split(ipsum)[1:N]]

	@show size(cenc(inptoks))
	@show 5*length(ngram_sizes), N  
	@test size(cenc(inptoks)) == (5*length(ngram_sizes), N)  
end