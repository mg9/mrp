using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random, CuArrays, IterTools

include("debug.jl")
include("data.jl")
include("embed.jl")

# ## S2S: Sequence to sequence model with attention
#
# A sequence to sequence encoder-decoder model with attention for text to linearized-AMR translation
# the `memory` layer computes keys and values from the encoder,
# the `attention` layer computes the attention vector for the decoder.

struct Memory; w; end

struct Attention; wquery; wattn; scale; end

mutable struct S2S
    encoder::RNN             # srcembed(Ex,B,Tx) -> enccell(Dx*H,B,Tx)
    srcmemory::Memory        # enccell(Dx*H,B,Tx) -> keys(H,Tx,B), vals(Dx*H,Tx,B)
    decoder::RNN             # tgtembed(Ey,B,Ty) . attnvec(H,B,Ty)[t-1] = (Ey+H,B,Ty) -> deccell(H,B,Ty)
    #tgtmemory::Memory        # enccell(Dy*Hy,B,Ty) -> keys(H,Ty,B), vals(Dy*H,Ty,B)
    srcattention::Attention  # deccell(H,B,Ty), keys(H,Tx,B), vals(Dx*H,Tx,B) -> attnvec(H,B,Ty)
    #tgtattention::Attention  # deccell(H,B,Ty), keys(H,Ty,B), vals(Dy*H,Ty,B) -> attnvec(H,B,Ty)
    dropout::Real            # dropout probability
end


# ## Part 1. Model constructor
#
# The `S2S` constructor takes the following arguments:
# * `hidden`: size of the hidden vectors for both the encoder and the decoder
# * `layers=1`: number of layers
# * `bidirectional=false`: whether the encoder is bidirectional
# * `dropout=0`: dropout probability
#
# Hints:
# * If the encoder is bidirectional `layers` must be even and the encoder should have `layers÷2` layers.
# * The decoder will use "input feeding", i.e. it will concatenate its previous output to its input. Therefore the input size for the decoder should be `tgtembsz+hidden`.
# * Only `numLayers`, `dropout`, and `bidirectional` keyword arguments should be used for RNNs, leave everything else default.
# * The memory parameter `w` is used to convert encoder states to keys. If the encoder is bidirectional initialize it to a `(hidden,2*hidden)` parameter, otherwise set it to the constant 1.
# * The attention parameter `wquery` is used to transform the query, set it to the constant 1 for this project.
# * The attention parameter `scale` is used to scale the attention scores before softmax, set it to a parameter of size 1.
# * The attention parameter `wattn` is used to transform the concatenation of the decoder output and the context vector to the attention vector. It should be a parameter of size `(hidden,2*hidden)` if unidirectional, `(hidden,3*hidden)` if bidirectional.


function S2S(hidden::Int, srcembsz::Int, tgtembsz::Int; layers=1, bidirectional=false, dropout=0)
    @assert !bidirectional || iseven(layers) "layers should be even for bidirectional models"
    encoderlayers = (bidirectional ? layers ÷ 2 : layers)
    encoder = RNN(srcembsz, hidden; dropout=dropout, numLayers=encoderlayers, bidirectional=bidirectional)
   
    decoderinput = tgtembsz + hidden
    decoder = RNN(decoderinput, hidden; dropout=dropout, numLayers=layers)
    
    srcmemory = bidirectional ? Memory(param(hidden,2hidden)) : Memory(1)
    tgtmemory = Memory(param(hidden,hidden))
   
    wquery = 1
    srcscale = param(1)
    #tgtscale = param(1)

    srcwattn = bidirectional ? param(hidden,3hidden) : param(hidden,2hidden)
    #tgtwattn = param(hidden, 2hidden)

    srcattn = Attention(wquery, srcwattn,srcscale)
    #tgtattn = Attention(wquery, tgtwattn, tgtscale)

    #S2S(encoder, srcmemory, decoder, tgtmemory, srcattn, tgtattn, dropout) 
    S2S(encoder, srcmemory, decoder, srcattn, dropout) 
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
    mmul(w,x) = (w == 1 ? x : w == 0 ? 0 : reshape(w * reshape(x,size(x,1),:), (:, size(x)[2:end]...)))
    vals = permutedims(x, (1,3,2))
    keys = mmul(m.w, vals)
    return (keys, vals)
end

# You can use the following helper function for scaling and linear transformations of 3-D tensors:
mmul(w,x) = (w == 1 ? x : w == 0 ? 0 : reshape(w * reshape(x,size(x,1),:), (:, size(x)[2:end]...)))



# ## Part 3. Encoder
#
# First prepare encode inputs
# Concats embed of various features
function prepare_encode_input(inputs)
    ## inputs dict ->  (Ex, B, Tx), (B,Tx) 
    bert_tokens, token_subword_index, tokens, pos_tags, must_copy_tags, chars, mask = inputs["bert_token"], inputs["token_subword_index"], inputs["token"], inputs["pos_tag"], inputs["must_copy_tag"], inputs["char"], inputs["mask"]
    encoder_inputs = []
    CNN_EMB_DIM = 100; DROPOUT = 0.33
    Tx, B = size(tokens)

    # (pytorch loads are 0-indexed, however knet 1-indexed)
    must_copy_tags = must_copy_tags .+1
    tokens = tokens .+1
    pos_tags = pos_tags .+1

    # Bert embeddings
    if true #use_bert
        bert_embeddings = KnetArray{Float32}(zeros(BERT_EMB_DIM, Tx , B))
        push!(encoder_inputs, bert_embeddings)
    end

    # Token embeddings
    embed = Embed(ENCODER_TOKEN_VOCAB_SIZE, TOKEN_EMB_DIM)
    token_embeddings = embed(tokens)
    push!(encoder_inputs, token_embeddings)

    # Postag embeddings
    embed = Embed(POSTAG_VOCAB_SIZE, POS_EMB_DIM)
    pos_tag_embeddings = embed(pos_tags)
    push!(encoder_inputs, pos_tag_embeddings)

    # Mustcopy embeddings
    if true #use_must_copy_embedding
        embed = Embed(MUSTCOPY_VOCAB_SIZE, MUSTCOPY_EMB_DIM)
        must_copy_tag_embeddings = embed(must_copy_tags)
        push!(encoder_inputs, must_copy_tag_embeddings)
    end

    # CharCNN embeddings
    if true #use_char_cnn
        embed = Embed(ENCODER_CHAR_VOCAB_SIZE, CNN_EMB_DIM)
        # chars: num_chars, Tx, B
        char_embeddings = embed(chars) # -> CNN_EMB_DIM , num_chars , Tx, B 
        CNN_EMB_DIM, num_chars, T, B = size(char_embeddings) # -> CNN_EMB_DIM , num_chars , Tx , B
        char_embeddings = reshape(char_embeddings, :,num_chars, B*T) # -> CNN_EMB_DIM, num_chars , (B x T)
        #Send it to charCNN which we ignore for now
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


# `encode()` takes a model `s` and encoder_inputs and encoder_mask?`. It passes the input
# through `s.srcembed` and `s.encoder` layers with the `s.encoder` RNN hidden states
# initialized to `0` in the beginning, and copied to the `s.decoder` RNN at the end. 
# The encoder output is passed to the `s.memory` layer which returns a `(keys,values)` pair. `encode()` returns
# this pair to be used later by the attention mechanism.

function encode(s::S2S, z, encoder_mask) 
    ##  (Ex,B,Tx), (B,Tx) -> ((Hx,Tx,B), (Hx*D,Tx,B))
                                    #; @size z (Ex, B, Tx)      
    s.encoder.h, s.encoder.c = 0, 0 #; #@sizes(s); (B,Tx) = size(src)
    z = s.encoder(z)                #; @size z (Hx*Dx,B,Tx)
    s.decoder.h = s.encoder.h       #; @size s.encoder.h (Hy,B,Ly)
    s.decoder.c = s.encoder.c       #; @size s.encoder.c (Hy,B,Ly)
    z = s.srcmemory(z)              #; if z != (); @size z[1] (Hy,Tx,B); @size z[2] (Hx*Dx,Tx,B); end # z:(keys,values)
    return z
end



# ## Part 4. Attention
#
# The attention layer takes `cell`: the decoder output, and `mem`: a pair of (keys,vals)
# from the encoder, and computes and returns the attention vector. First `a.wquery` is used
# to linearly transform the cell to the query tensor. The query tensor is reshaped and/or
# permuted as appropriate and multiplied with the keys tensor to compute the attention
# scores. Please see `@doc bmm` for the batched matrix multiply operation used for this
# step. The attention scores are scaled using `a.scale` and normalized along the time
# dimension using `softmax`. After the appropriate reshape and/or permutation, the scores
# are multiplied with the `vals` tensor (using `bmm` again) to compute the context
# tensor.

# After the appropriate reshape and/or permutation the context vector is
# concatenated with the cell and linearly transformed to the attention vector using
# `a.wattn`. Please see the paper and code examples for details.
#
# Note: the paper mentions a final `tanh` transform, however the final version of the
# reference code does not use `tanh` and gets better results. Therefore we will skip `tanh`.

function (a::Attention)(cell, mem)
    ## (Hy,B,Ty/1), ((Hy,Tx,B), (Hx*Dx,Tx,B)) -> (Hy,B,Ty/1)
    mmul(w,x) = (w == 1 ? x : w == 0 ? 0 : reshape(w * reshape(x,size(x,1),:), (:, size(x)[2:end]...)))
    qtrans(q) = (size(q,3)==1 ? reshape(q,(1,size(q,1),:)) : permutedims(q,(3,1,2)))
    atrans(a) = (size(a,1)==1 ? reshape(a,(size(a,2),1,:)) : permutedims(a,(2,1,3)))
    ctrans(c) = (size(c,2)==1 ? reshape(c,(size(c,1),:,1)) : permutedims(c,(1,3,2)))
    keys, vals = mem                         ; (Hy,B,Ty)=size(cell);(HxDx,Tx,B)=size(vals);@size keys (Hy,Tx,B)
    query = mmul(a.wquery, cell)             ; @size query (Hy,B,Ty)
    query = qtrans(query)                    ; @size query (Ty,Hy,B)
    alignments = bmm(query, keys)            ; @size alignments (Ty,Tx,B) 
    alignments = a.scale .* alignments       ; @size alignments (Ty,Tx,B) 
    alignments = softmax(alignments, dims=2) # normalize along the Tx dimension
    alignments = atrans(alignments)          ; @size alignments (Tx,Ty,B)
    context = bmm(vals, alignments)          ; @size context (HxDx,Ty,B)
    context = ctrans(context)                ; @size context (HxDx,B,Ty)
    attnvec = mmul(a.wattn, vcat(cell,context)) ; @size attnvec (Hy,B,Ty)
    return attnvec, alignments
    ## return tanh.(attnvec)
    ## doc says tanh, implementation does not have it, no tanh does better:
    ## https://github.com/tensorflow/nmt/issues/57
end




# ## Part 5. Decoder
#
# `decode()` takes a model `s`, a linearized-AMR `decoder_input`, the memory from the
# encoder `mem` and the decoder output from the previous time step `prev`. After the input
# is passed through the embedding layer, it is concatenated with `prev` (this is called
# input feeding). The resulting tensor is passed through `s.decoder`. Finally the
# `s.attention` layer takes the decoder output and the encoder memory to compute the
# "attention vector" which is returned by `decode()`.

#
# First prepare decode inputs
# Concats embed of various features
function prepare_decode_input(enc_inputs, dec_inputs, parser_inputs)
    ## enc_inputs, dec_inputs, parser_inputs dicts ->  (Ey, B, Ty)
    tokens, pos_tags, chars, corefs, mask, tgt_mask = dec_inputs["token"], dec_inputs["pos_tag"], dec_inputs["char"], dec_inputs["coref"], enc_inputs["mask"], parser_inputs["mask"]
    decoder_inputs = []
    CNN_EMB_DIM = 100; DROPOUT = 0.33
    Ty, B = size(tokens)

    # (pytorch loads are 0-indexed, however knet 1-indexed)
    corefs = corefs .+1

    # [batch, num_tokens, embedding_size]   
    # Token embeddings
    embed = Embed(DECODER_TOKEN_VOCAB_SIZE, TOKEN_EMB_DIM)
    token_embeddings = embed(tokens)
    push!(decoder_inputs, token_embeddings)

    # Postag embeddings
    embed = Embed(POSTAG_VOCAB_SIZE, POS_EMB_DIM)
    pos_tag_embeddings = embed(pos_tags)
    push!(decoder_inputs, pos_tag_embeddings)

    # Coref embeddings
    embed = Embed(DECODER_COREF_VOCAB_SIZE, COREF_EMB_DIM)
    coref_embeddings = embed(corefs)
    push!(decoder_inputs, coref_embeddings)

    # CharCNN embeddings
    if true #use_char_cnn
        embed = Embed(DECODER_CHAR_VOCAB_SIZE, CNN_EMB_DIM)
        # chars: num_chars, Tx, B
        char_embeddings = embed(chars) # -> CNN_EMB_DIM , num_chars , Ty, B 
        CNN_EMB_DIM, num_chars, Ty, B = size(char_embeddings) # -> CNN_EMB_DIM , num_chars , Ty , B
        char_embeddings = reshape(char_embeddings, :,num_chars, B*Ty) # -> CNN_EMB_DIM, num_chars , (B, T)
        # Send it to charCNN which we ignore for now
        #char_cnn_output = self.encoder_char_cnn(char_embeddings, None)
        #char_cnn_output = char_cnn_output.view(batch_size, Ty, -1)
        char_cnn_output = KnetArray{Float32}(zeros(CNN_EMB_DIM , Ty, B))
        push!(decoder_inputs, char_cnn_output)
    end

    decoder_inputs = vcat(decoder_inputs...)
    decoder_inputs = dropout(decoder_inputs, DROPOUT)
    decoder_inputs = permutedims(decoder_inputs, [1, 3, 2])
    return decoder_inputs
end


function decode(s::S2S, input, srcmem, prev)
    # input:    (Ey,B,1) 
    # srcmem:   ((Hy,Tx,B),(Hx*Dx,Tx,B))
    # prev:     (Hy,B,1) 
    # -> z(hiddens):      (Hy,B,1)
    # -> srcalignments:   (Tx,Ty,B)
    # -> tgtalignments:   (Ty,Ty,B)
    
    z = input
    z = s.decoder(vcat(convert(_atype,z), prev))                    # -> (H,B,1)               
    src_attn_vector, srcalignments = s.srcattention(z, srcmem)      # -> (H,B,1), (Tx,1,B) 
    #tgtmem = s.tgtmemory(src_attn_vector)
    #tgt_attn_vector, tgtalignments = s.tgtattention(z, tgtmem)        
    return z, src_attn_vector, srcalignments #, tgtalignments
end
