include("s2s.jl")
include("pointer_generator.jl")
include("deep_biaffine_graph_decoder.jl")
include("debug.jl")

struct BaseModel
    s::S2S             
    p::PointerGenerator
    g::DeepBiaffineGraphDecoder
end


# ## Model constructor
#
# The `BaseModel` constructor takes the following arguments:
# * `H`: size of the hidden vectors for both the s2s encoder and the decoder
# * `Ex`: s2s encoder embed size of 
# * `Ey`: s2s decoder embed size of
# * `L`: number of layers
# * `bidirectional=true`: whether the s2s encoder is bidirectional 
# * `dropout=0`: dropout probability
# * `vocabsize`: number of s2s decoder tokens vocabsize
# * `edgenode_hiddensize`: number of edgenode_hiddensize;  transform representations into a space for edge node heads and edge node modifiers
# * `edgelabel_hiddensize`: number of edgelabel_hiddensize; transform representations into a space for edge label heads and edge label modifiers
# * `num_labels`: number of head tags
function BaseModel(H, Ex, Ey, L, vocabsize, edgenode_hidden_size, edgelabel_hidden_size, num_labels; bidirectional=true, dropout=0)
    s = S2S(H,Ex,Ey; layers=L, bidirectional=bidirectional, dropout=Pdrop)
    p = PointerGenerator(H, vocabsize)
    g = DeepBiaffineGraphDecoder(H, edgenode_hidden_size, edgelabel_hidden_size, num_labels)
    BaseModel(s, p, g)
end


function (m::BaseModel)(srcpostags, tgtpostags, corefs, srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, parsermask, edgeheads, edgelabels,encodervocab_pad_idx, decodervocab_pad_idx ; force_copy=true)

    B, Tx = size(srctokens); Ty = size(tgttokens,2); Hx=Hy = H; Dx =2
    @size srcattentionmaps (Tx,Tx+2,B); @size tgtattentionmaps (Ty,Ty+1,B); 
    srcmem = encode(m.s, srctokens, srcpostags) ; if srcmem != (); @size srcmem[1] (Hy,Tx,B); @size srcmem[2] (Hx*Dx,Tx,B); end
    #tgt2 = mask!(tgttokens[:,2:end], s.tgteos)    ; @size tgt2 (B,Ty-1)
    tgt2 = tgttokens[:,2:end]  ; @size tgt2 (B,Ty-1)
    sumloss, pgeneratorloss, graphloss  = 0, 0, 0
    cell = fill!(similar(srcmem[1], size(srcmem[1],1), size(srcmem[1],3), 1), 0)
    hiddens = []

    # Coverage vector for the source vocabulary, accumulated
    coverage = convert(_atype,zeros(Tx,1, B))
    # List of coverages at each timestamp if needed
    # coverage_records = [coverage]
    tgtkeys=tgtvals =nothing
    for t in 1:size(tgt2,2)
        input,output = (@view tgttokens[:,t:t]), (@view tgt2[:,t:t])
        # TODO: Input masks are not considered in the decoding part
        postag = (@view tgtpostags[:,t:t])
        _corefs = (@view corefs[:,t:t]) ;@size _corefs (B,1)

        cell, srcattnvector, srcattentions, tgtattnvector, tgtattentions, tgtkeys, tgtvals = decode(m.s, input, postag, _corefs, srcmem, cell, t, tgtkeys, tgtvals)          ; @size cell (Hy,B,1); @size srcattnvector (Hy,B,1);  @size srcattentions (Tx,1,B);
        push!(hiddens, cell)
        srcattnvector = reshape(srcattnvector, Hy,B)                                ; @size srcattnvector (Hy,B);
        dif = Ty-size(tgtattentions,1) 
        if dif>0
            pad = zeros(dif, size(tgtattentions,2), B)
            tgtattentions = vcat(tgtattentions, _atype(pad))
        end
        @size tgtattentions (Ty,1,B);
        sumloss += m.p(srcattnvector, srcattentions, tgtattentions, srcattentionmaps, tgtattentionmaps,  generatetargets[t:t,:], srccopytargets[t:t,:] , tgtcopytargets[t:t,:], decodervocab_pad_idx, coverage)
    end
    hiddens = cat(hiddens...,dims=3) ;@size hiddens (Hy,B,Ty-1)
    graphloss = m.g(hiddens, parsermask, edgeheads, edgelabels)
    sumloss += graphloss
    return  sumloss
end