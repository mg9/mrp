include("s2s.jl")
include("pointer_generator.jl")
include("deep_biaffine_graph_decoder.jl")
include("debug.jl")
include("data.jl")

# Model configs
TOKEN_EMB_DIM = 300 #glove
POS_EMB_DIM = 100
COREF_EMB_DIM = 50
CHARCNN_EMB_DIM = 100 # The CharCNN output size, this is equal to length(Ngram_sizes) * Num_Filters
CHARCNN_EMB_SIZE = 100 # This is for the internal character embedding size
CHARCNN_NUM_FILTERS = 100
CHARCNN_NGRAM_SIZES = [3]
ENCODER_VOCAB_SIZE = 18420 
DECODER_VOCAB_SIZE = 13286 
POSTAG_VOCAB_SIZE = 52
DECODER_COREF_VOCAB_SIZE = 500
Ex = TOKEN_EMB_DIM + POS_EMB_DIM # + CHARCNN_EMB_DIM
Ey = TOKEN_EMB_DIM + POS_EMB_DIM + COREF_EMB_DIM  #+ CHARCNN_EMB_DIM + COREF_EMB_DIM 
H, L, Pdrop = 512, 2, 0.33
edgenode_hiddensize = 256
edgelabel_hiddensize = 128
num_edgelabels = 144
epochs = 50

## Not used yet
BERT_EMB_DIM = 768
CNN_EMB_DIM = 100
MUSTCOPY_EMB_DIM = 50
MUSTCOPY_VOCAB_SIZE = 3
ENCODER_CHAR_VOCAB_SIZE = 125
DECODER_CHAR_VOCAB_SIZE = 87


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
function BaseModel(H, Ex, Ey, L, vocabsize, edgenode_hidden_size, edgelabel_hidden_size, num_labels, srcvocab, srccharvocab, tgtvocab, tgtcharvocab; bidirectional=true, dropout=0)
    s = S2S(H,Ex,srcvocab,srccharvocab,Ey,tgtvocab,tgtcharvocab; layers=L, bidirectional=bidirectional, dropout=Pdrop)
    p = PointerGenerator(H, vocabsize)
    g = DeepBiaffineGraphDecoder(H, edgenode_hidden_size, edgelabel_hidden_size, num_labels)
    BaseModel(s, p, g)
end


function (m::BaseModel)(b::AMRBatch)

    B, Tx = size(b.srctokens); Ty = size(b.tgttokens,2); Hx=Hy = H; Dx =2
    @size b.srcattentionmaps (Tx,Tx+2,B); @size b.tgtattentionmaps (Ty,Ty+1,B); 
    srcmem = encode(m.s, b.srctokens, b.srcpostags) ; if srcmem != (); @size srcmem[1] (Hy,Tx,B); @size srcmem[2] (Hx*Dx,Tx,B); end
    tgt2 = b.tgttokens[:,2:end]  ; @size tgt2 (B,Ty-1)
    sumloss, pgeneratorloss, graphloss  = 0, 0, 0
    hiddens = []
    srcattentionvectors = []
    srcattentions = []
    tgtattentions = []

    # Coverage vector for the source vocabulary, accumulated
    coverage = convert(_atype,zeros(Tx,1, B))
    # List of coverages at each timestamp if needed
    # coverage_records = [coverage]
    tgtkeys=tgtvals =nothing
    cell = fill!(similar(srcmem[1], size(srcmem[1],1), size(srcmem[1],3), 1), 0)

    for t in 1:size(tgt2,2)
        input,output = (@view b.tgttokens[:,t:t]), (@view tgt2[:,t:t])
        # TODO: Input masks are not considered in the decoding part
        _postag = (@view b.tgtpostags[:,t:t])
        _corefs = (@view b.corefs[:,t:t]) ;@size _corefs (B,1)
        cell, srcattnvector, srcattention, tgtattnvector, tgtattention, tgtkeys, tgtvals = decode(m.s, input, _postag, _corefs, srcmem, cell, t, tgtkeys, tgtvals)          ; @size cell (Hy,B,1); @size srcattnvector (Hy,B,1);  @size srcattention (Tx,1,B);
        dif = Ty-size(tgtattention,1) 
        if dif>0
            pad = zeros(dif, size(tgtattention,2), B)
            tgtattention = vcat(tgtattention, _atype(pad))
        end
        @size tgtattention (Ty,1,B);

        push!(hiddens, cell)
        push!(srcattentionvectors, srcattnvector)
        push!(tgtattentions, tgtattention)
        push!(srcattentions, srcattention)
    end

    # Cat, reshape etc. 
    srcattentionvectors = cat(srcattentionvectors...,dims=3) ;@size srcattentionvectors (Hy,B,Ty-1)
    _srcattentionvectors = reshape(permutedims(srcattentionvectors, [1,3,2]), Hy,:)
    tgtattentions = cat(tgtattentions...,dims=2) ;@size tgtattentions (Ty,Ty-1,B)
    srcattentions = cat(srcattentions...,dims=2) ;@size srcattentions (Tx,Ty-1,B)
   
    # Add pointer generator loss
    sumloss += model.p(_srcattentionvectors, srcattentions, tgtattentions, b.srcattentionmaps, b.tgtattentionmaps,  b.generatetargets, b.srccopytargets , b.tgtcopytargets, b.decodervocab_pad_idx, nothing)

    # Add graph decoder loss, send mask and vectors with excluding BOS
    sumloss += m.g(srcattentionvectors[:,:,2:end], b.parsermask[:,2:end], b.edgeheads, b.edgelabels)
    return sumloss
end