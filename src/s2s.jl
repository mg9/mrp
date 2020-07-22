using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random, CuArrays, IterTools

include("debug.jl")
include("embed.jl")

_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})


# ## S2S: Sequence to sequence model with attention
#
# A sequence to sequence encoder-decoder model with attention for text to linearized-AMR translation
# the `memory` layer computes keys and values from the encoder,
# the `attention` layer computes the attention vector for the decoder.

struct Memory; w; end

struct Attention; wquery; wattn; scale; end

struct S2S
    srctokenembed::Embed        # encinput(B,Tx) -> srcembed(Ex,B,Tx)
    srcpostagembed::Embed
    encoder::RNN                
    srcmemory::Memory      
    tgtmemory::Memory           
    tgttokenembed::Embed       
    tgtpostagembed::Embed   
    tgtcorefembed::Embed    
    decoder::RNN               
    srcattention::Attention  
    tgtattention::Attention     
    dropout::Real   
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
    srctokenembed  = Embed(ENCODER_VOCAB_SIZE, TOKEN_EMB_DIM)
    srcpostagembed = Embed(POSTAG_VOCAB_SIZE, POS_EMB_DIM)

    tgttokenembed  = Embed(DECODER_VOCAB_SIZE, TOKEN_EMB_DIM)
    tgtpostagembed = Embed(POSTAG_VOCAB_SIZE, POS_EMB_DIM)
    tgtcorefembed  = Embed(DECODER_COREF_VOCAB_SIZE, COREF_EMB_DIM)

    encoderlayers = (bidirectional ? layers ÷ 2 : layers)
    encoder = RNN(srcembsz, hidden; dropout=dropout, numLayers=encoderlayers, bidirectional=bidirectional)
    decoderinput = tgtembsz + hidden
    decoder = RNN(decoderinput, hidden; dropout=dropout, numLayers=layers)
    
    srcmemory = bidirectional ? Memory(param(hidden,2hidden)) : Memory(1)
    tgtmemory =  Memory(1)
    wquery = 1
    
    srcscale = param(1)
    tgtscale = param(1)
   
    srcwattn = bidirectional ? param(hidden,3hidden) : param(hidden,2hidden)
    tgtwattn = param(hidden,3hidden)

    srcattn = Attention(wquery, srcwattn, srcscale)
    tgtattn = Attention(wquery, tgtwattn, tgtscale)

    S2S(srctokenembed, srcpostagembed, encoder, srcmemory, tgtmemory, tgttokenembed, tgtpostagembed, tgtcorefembed, decoder, srcattn, tgtattn, dropout) 
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

# `encode()` takes a model `s` and encoder_inputs`. It passes the input
# through `s.srcembed` and `s.encoder` layers with the `s.encoder` RNN hidden states
# initialized to `0` in the beginning, and copied to the `s.decoder` RNN at the end. 
# The encoder output is passed to the `s.memory` layer which returns a `(keys,values)` pair. `encode()` returns
# this pair to be used later by the attention mechanism.

function encode(s::S2S, tokens, srcpostags) 
    s.encoder.h, s.encoder.c = 0, 0    #; @sizes(s); (B,Tx) = size(tokens)
    z = cat(s.srcpostagembed(srcpostags), s.srctokenembed(tokens),dims=1)
    z = s.encoder(z)                   #; @size z (Hx*Dx,B,Tx)
    s.decoder.h = s.encoder.h          #; @size s.encoder.h (Hy,B,s.decoder.numLayers)
    s.decoder.c = s.encoder.c          #; @size s.encoder.c (Hy,B,s.decoder.numLayers)
    z = s.srcmemory(z)                 #; if z != (); @size z[1] (Hy,Tx,B); @size z[2] (Hx*2,Tx,B); end # z:(keys,values)
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

function (a::Attention)(cell, mem; tgt=false, t=nothing, srcattnvector=nothing)
    ## (Hy,B,Ty/1), ((Hy,Tx,B), (Hx*Dx,Tx,B)) -> (Hy,B,Ty/1)
    mmul(w,x) = (w == 1 ? x : w == 0 ? 0 : reshape(w * reshape(x,size(x,1),:), (:, size(x)[2:end]...)))
    qtrans(q) = (size(q,3)==1 ? reshape(q,(1,size(q,1),:)) : permutedims(q,(3,1,2)))
    atrans(a) = (size(a,1)==1 ? reshape(a,(size(a,2),1,:)) : permutedims(a,(2,1,3)))
    ctrans(c) = (size(c,2)==1 ? reshape(c,(size(c,1),:,1)) : permutedims(c,(1,3,2)))
    keys, vals = mem                         #; (Hy,B,Ty)=size(cell);(HxDx,Tx,B)=size(vals);@size keys (Hy,Tx,B)
    query = mmul(a.wquery, cell)             #; @size query (Hy,B,Ty)
    query = qtrans(query)                    #; @size query (Ty,Hy,B)
    alignments = bmm(query, keys)            #; @size alignments (Ty,Tx,B) 
    alignments = a.scale .* alignments       #; @size alignments (Ty,Tx,B) 

    if tgt && !isnothing(t) 
        if t>1 && t < size(alignments,2)
            mask = _atype(ones(1, size(alignments,2), size(alignments,3)))
            mask[:,t+1:end,:] .= -Inf # only allow to attend earlier positions and itself.
            alignments = alignments .+ mask
        end
    end
    alignments = softmax(alignments, dims=2) # normalize along the Tx dimension
    alignments = atrans(alignments)          #; @size alignments (Tx,Ty,B)
    context = bmm(vals, alignments)          #; @size context (HxDx,Ty,B)
    context = ctrans(context)                #; @size context (HxDx,B,Ty)

    if !isnothing(srcattnvector)
        attnvec = mmul(a.wattn, vcat(cell,context, srcattnvector)) #; @size attnvec (Hy,B,Ty)
    else
        attnvec = mmul(a.wattn, vcat(cell,context)) 
    end
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

function decode(s::S2S, tokens, tgtpostags, corefs, srcmem, prev, t, prevkeys, prevvals)
    B,Ty = size(corefs)

    z = cat(s.tgttokenembed(tokens), 
            s.tgtpostagembed(tgtpostags),
            s.tgtcorefembed(corefs),dims=1)   #; (B,Ty) = size(tokens); @sizes(s); @size z (Ey,B,Ty)
    z = vcat(z, prev)                                                       #; @size z (Ey+Hy,B,1)
    z = s.decoder(z)                                                        #; @size z (Hy,B,1)
    srcattnvector, srcalignments = s.srcattention(z, srcmem)                #; @size srcattnvector (Hy,B,1); @size srcalignments (Tx,1,B)
    
    keys,vals = s.tgtmemory(z) 
    if !isnothing(prevkeys) && !isnothing(prevvals)
        keys = cat(prevkeys, keys, dims=2)
        vals = cat(prevvals, vals, dims=2)
    end
    tgtmem=(keys,vals)
    tgtattnvector, tgtalignments = s.tgtattention(z, tgtmem, tgt=true, t=t, srcattnvector=srcattnvector)                #; @size srcattnvector (Hy,B,1); @size srcalignments (Tx,1,B)
    return z, srcattnvector, srcalignments, tgtattnvector, tgtalignments,  keys, vals
end

