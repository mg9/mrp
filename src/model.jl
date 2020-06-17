include("debug.jl")
include("data.jl")
include("embed.jl")
include("linear.jl")

# ## S2S: Sequence to sequence model with attention
#
# A sequence to sequence encoder-decoder model with attention for text to linearized-AMR translation
# the `memory` layer computes keys and values from the encoder,
# the `attention` layer computes the attention vector for the decoder.

struct Memory; w; end

struct Attention; wquery; wattn; scale; end

struct S2S
    #srcembed::Embed       # encinput(B,Tx) -> srcembed(Ex,B,Tx)
    encoder::RNN          # srcembed(Ex,B,Tx) -> enccell(Dx*H,B,Tx)
    memory::Memory        # enccell(Dx*H,B,Tx) -> keys(H,Tx,B), vals(Dx*H,Tx,B)
    #tgtembed::Embed       # decinput(B,Ty) -> tgtembed(Ey,B,Ty)
    decoder::RNN          # tgtembed(Ey,B,Ty) . attnvec(H,B,Ty)[t-1] = (Ey+H,B,Ty) -> deccell(H,B,Ty)
    attention::Attention  # deccell(H,B,Ty), keys(H,Tx,B), vals(Dx*H,Tx,B) -> attnvec(H,B,Ty)
    #projection::Linear    # attnvec(H,B,Ty) -> proj(Vy,B,Ty)
    dropout::Real         # dropout probability
    #srcvocab::Vocab       # source language vocabulary
    #tgtvocab::Vocab       # target language vocabulary
end


BERT_EMB_DIM = 768
TOKEN_EMB_DIM = 300 #glove
POS_EMB_DIM = 100
MUSTCOPY_EMB_DIM = 50
CNN_EMB_DIM = 100
ENCODER_TOKEN_VOCAB_SIZE = 18002
ENCODER_CHAR_VOCAB_SIZE = 125
ENCODER_POSTAG_VOCAB_SIZE = 51
DECODER_TOKEN_VOCAB_SIZE = 12202
DECODER_CHAR_VOCAB_SIZE = 87
MUSTCOPY_VOCAB_SIZE = 3
Ex = BERT_EMB_DIM + TOKEN_EMB_DIM + POS_EMB_DIM + MUSTCOPY_EMB_DIM + CNN_EMB_DIM


# ## prepare_encode_input
#
# Concats embed of various features
# returns (encoder_inputs, encoder_mask) 
# encoder_inputs -> Ex x B x T 
# encoder_mask -> B x T
function prepare_encode_input(bert_tokens, token_subword_index, tokens, pos_tags, must_copy_tags, chars, mask)
    encoder_inputs = []

    CNN_EMB_DIM = 100; DROPOUT = 0.33
    Tx, B = size(tokens)
    # (pytorch loads are 0-indexed, however knet 1-indexed)
    must_copy_tags = must_copy_tags .+1
    tokens = tokens .+1
    pos_tags = pos_tags .+1

    if true #use_bert
        bert_embeddings = KnetArray{Float32}(zeros(BERT_EMB_DIM, Tx , B))
        push!(encoder_inputs, bert_embeddings)
    end

    embed = Embed(ENCODER_TOKEN_VOCAB_SIZE, TOKEN_EMB_DIM)
    token_embeddings = embed(tokens)
    push!(encoder_inputs, token_embeddings)

    embed = Embed(ENCODER_POSTAG_VOCAB_SIZE, POS_EMB_DIM)
    pos_tag_embeddings = embed(pos_tags)
    push!(encoder_inputs, pos_tag_embeddings)

    if true #use_must_copy_embedding
        embed = Embed(MUSTCOPY_VOCAB_SIZE, MUSTCOPY_EMB_DIM)
        must_copy_tag_embeddings = embed(must_copy_tags)
        push!(encoder_inputs, must_copy_tag_embeddings)
    end

    if true #use_char_cnn
        embed = Embed(ENCODER_CHAR_VOCAB_SIZE, CNN_EMB_DIM)
        # chars: num_chars, Tx, B
        char_embeddings = embed(chars) # -> CNN_EMB_DIM , num_chars , Tx, B 
        CNN_EMB_DIM, num_chars, T, B = size(char_embeddings) # -> CNN_EMB_DIM , num_chars , Tx , B
        char_embeddings = reshape(char_embeddings, :,num_chars, B*T) # -> CNN_EMB_DIM, num_chars , (B x T)

        # Send it to charCNN which we ignore for now
        #char_cnn_output = self.encoder_char_cnn(char_embeddings, None)
        #char_cnn_output = char_cnn_output.view(batch_size, Tx, -1)

        char_cnn_output = KnetArray{Float32}(zeros(CNN_EMB_DIM , Tx, B))
        push!(encoder_inputs, char_cnn_output)
    end
    encoder_inputs = vcat(encoder_inputs...)
    encoder_inputs = dropout(encoder_inputs, DROPOUT)
    encoder_inputs = permutedims(encoder_inputs, [1, 3, 2])
    encoder_mask = permutedims(mask, [2, 1])
    return encoder_inputs, encoder_mask
end

#-
@testset "Testing prepare_encode_input" begin
    encoder_inputs, decoder_inputs, generator_inputs, parser_inputs = prepare_batch_input(batch)
    enc_inputs, encoder_mask = prepare_encode_input(
                encoder_inputs["bert_token"],
                encoder_inputs["token_subword_index"],
                encoder_inputs["token"],
                encoder_inputs["pos_tag"],
                encoder_inputs["must_copy_tag"],
                encoder_inputs["char"],
                encoder_inputs["mask"])
    Tx,B = size(encoder_inputs["token"])
    @test size(enc_inputs) == (Ex, B, Tx)
    @test size(encoder_mask) == (B, Tx)
end



function S2S(hidden::Int, srcembsz::Int, tgtembsz::Int; layers=1, bidirectional=false, dropout=0)
    @assert !bidirectional || iseven(layers) "layers should be even for bidirectional models"
    #srcvocsz,tgtvocsz = length.((srcvocab.i2w,tgtvocab.i2w))
    #srcembed = Embed(srcvocsz, srcembsz)
    encoderlayers = (bidirectional ? layers ÷ 2 : layers)
    encoder = RNN(srcembsz, hidden; dropout=dropout, numLayers=encoderlayers, bidirectional=bidirectional)
    
    #tgtembed = Embed(tgtvocsz, tgtembsz)
    decoderinput = tgtembsz + hidden
    decoder = RNN(decoderinput, hidden; dropout=dropout, numLayers=layers)
    #projection = Linear(hidden,tgtvocsz)
    memory = bidirectional ? Memory(param(hidden,2hidden)) : Memory(1)
    wquery = 1
    scale = param(1)
    wattn = bidirectional ? param(hidden,3hidden) : param(hidden,2hidden)
    attn = Attention(wquery,wattn,scale)
    S2S(encoder,memory, decoder, attn, dropout) 
end

#-
H, Ex, Ey, L, Dx,Pdrop = 512, Ex, 10, 2, 2, 0.33
m = S2S(H,Ex,Ey; layers=L,bidirectional=(Dx==2),dropout=Pdrop)
@testset "Testing S2S constructor" begin
    #@test size(m.srcembed.w) == (Ex,Vx)
    #@test size(m.tgtembed.w) == (Ey,Vy)
    @test m.encoder.inputSize == Ex
    @test m.decoder.inputSize == Ey + H
    @test m.encoder.hiddenSize == m.decoder.hiddenSize == H
    @test m.encoder.direction == Dx-1
    @test m.encoder.numLayers == (Dx == 2 ? L÷2 : L)
    @test m.decoder.numLayers == L
    @test m.encoder.dropout == m.decoder.dropout == Pdrop
    #@test size(m.projection.w) == (Vy,H)
    @test size(m.memory.w) == (Dx == 2 ? (H,2H) : ())
    @test m.attention.wquery == 1
    @test size(m.attention.wattn) == (Dx == 2 ? (H,3H) : (H,2H))
    @test size(m.attention.scale) == (1,)
    #@test m.srcvocab === dtrn.src.vocab
    #@test m.tgtvocab === dtrn.tgt.vocab
end


# ## Part 2. Memory
#
# The memory layer turns the output of the encoder to a pair of tensors that will be used as
# keys and values for the attention mechanism. Remember that the encoder RNN output has size
# `(H*D,B,Tx)` where `H` is the hidden size, `D` is 1 for unidirectional, 2 for
# bidirectional, `B` is the batchsize, and `Tx` is the sequence length. It will be
# convenient to store these values in batch major form for the attention mechanism, so
# *values* in memory will be a permuted copy of the encoder output with size `(H*D,Tx,B)`
# (see `@doc permutedims`). The *keys* in the memory need to have the same first dimension
# as the *queries* (i.e. the decoder hidden states). So *values* will be transformed into
# *keys* of size `(H,B,Tx)` with `keys = m.w * values` where `m::Memory` is the memory
# layer. Note that you will have to do some reshaping to 2-D and back to 3-D for matrix
# multiplications. Also note that `m.w` may be a scalar such as `1` e.g. when `D=1` and we
# want keys and values to be identical.


function (m::Memory)(x)
    ## Your code here
    mmul(w,x) = (w == 1 ? x : w == 0 ? 0 : reshape(w * reshape(x,size(x,1),:), (:, size(x)[2:end]...)))
    vals = permutedims(x, (1,3,2))
    keys = mmul(m.w, vals)
    return (keys, vals)
    ## Your code here
end

# You can use the following helper function for scaling and linear transformations of 3-D tensors:
mmul(w,x) = (w == 1 ? x : w == 0 ? 0 : reshape(w * reshape(x,size(x,1),:), (:, size(x)[2:end]...)))

#-
@testset "Testing memory" begin
    H,D,B,Tx =  m.encoder.hiddenSize, m.encoder.direction+1, 64, 7
    x = KnetArray(randn(Float32,H*D,B,Tx))
    k,v = m.memory(x)
    @test v == permutedims(x,(1,3,2))
    @test k == mmul(m.memory.w, v)
end


# ## Part 3. Encoder
#
# `encode()` takes a model `s` and encoder_inputs and encoder_mask?`. It passes the input
# through `s.srcembed` and `s.encoder` layers with the `s.encoder` RNN hidden states
# initialized to `0` in the beginning, and copied to the `s.decoder` RNN at the end. 
# The encoder output is passed to the `s.memory` layer which returns a `(keys,values)` pair. `encode()` returns
# this pair to be used later by the attention mechanism.

function encode(s::S2S, encoder_inputs, encoder_mask) 
    z = encoder_inputs              #; @size z (Ex, B, Tx)      
    s.encoder.h, s.encoder.c = 0, 0 #; #@sizes(s); (B,Tx) = size(src)
    z = s.encoder(z)                #; @size z (Hx*Dx,B,Tx)
    s.decoder.h = s.encoder.h       #; @size s.encoder.h (Hy,B,Ly)
    s.decoder.c = s.encoder.c       #; @size s.encoder.c (Hy,B,Ly)
    z = s.memory(z)                 #; if z != (); @size z[1] (Hy,Tx,B); @size z[2] (Hx*Dx,Tx,B); end # z:(keys,values)
    return z
end



#-
@testset "Testing encoder" begin
    encoder_inputs, decoder_inputs, generator_inputs, parser_inputs = prepare_batch_input(batch)
    Tx,B = size(encoder_inputs["token"])
    enc_inputs, encoder_mask = prepare_encode_input(
                encoder_inputs["bert_token"],
                encoder_inputs["token_subword_index"],
                encoder_inputs["token"],
                encoder_inputs["pos_tag"],
                encoder_inputs["must_copy_tag"],
                encoder_inputs["char"],
                encoder_inputs["mask"])

    key1,val1 = encode(m, enc_inputs, encoder_mask)
    H,D,B, Tx = m.encoder.hiddenSize, m.encoder.direction+1, B, Tx
    @test size(key1) == (H,Tx,B)
    @test size(val1) == (H*D,Tx,B)
    @test (m.decoder.h,m.decoder.c) === (m.encoder.h,m.encoder.c)
    #@test norm(key1) ≈ 1214.4755f0
    #@test norm(val1) ≈ 191.10411f0
    #@test norm(pretrained.decoder.h) ≈ 48.536964f0
    #@test norm(pretrained.decoder.c) ≈ 391.69028f0
end



#=
# [batch, num_tokens, encoder_output_size]
encoder_outputs = self.encoder(encoder_inputs, mask)
encoder_outputs = self.encoder_output_dropout(encoder_outputs)

# A tuple of (state, memory) with shape [num_layers, batch, encoder_output_size]
encoder_final_states = self.encoder._states
self.encoder.reset_states()

return dict(
    memory_bank=encoder_outputs,
    final_states=encoder_final_states
)
=#



function batchloss(batch)
    encoder_inputs, decoder_inputs, generator_inputs, parser_inputs = prepare_batch_input(batch)
    encoder_inputs = prepare_encode_input(
        encoder_inputs["bert_token"],
        encoder_inputs["token_subword_index"],
        encoder_inputs["token"],
        encoder_inputs["pos_tag"],
        encoder_inputs["must_copy_tag"],
        encoder_inputs["char"],
        encoder_inputs["mask"]
    )
    ## TODO....
    decode_for_training()
    generator()
    generator_compute_loss()
    graph_decode()
end


