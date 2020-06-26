include("s2s.jl")
include("pointer_generator.jl")
include("deep_biaffine_graph_decoder.jl")

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
    p = PointerGenerator(H, vocabsize, 1e-20)
    g = DeepBiaffineGraphDecoder(H, edgenode_hidden_size, edgelabel_hidden_size, num_labels)
    BaseModel(s, p, g)
end


function (bm::BaseModel)(src, src_mask, tgt, src_attention_maps, tgt_attention_maps, generate_targets, src_copy_targets, tgt_copy_targets)
    # src:                (Ex, B, Tx) 
    # src_mask:           (B, Tx)
    # tgt:                (Ey, B, Ty)
    # src_attention_maps: (src_dynamic_vocabsize, Tx, B)
    # tgt_attention_maps: (tgt_dynamic_vocabsize, Ty, B)
    # generate_targets:   (Ty,B)
    # src_copy_targets:   (Ty,B)
    # tgt_copy_targets:   (Ty,B)

    # -> cell:             (H, B, 1)
    # -> src_attn_vector:  (H, B, 1)
    # -> srcalignments:    (13,1,B)
    # -> tgtalignments:    (1, 1, B)


    #TODO fix tgtalignments
    (Ex, B, Tx), Ty =size(src), size(tgt,3)
    tgtalignments = convert(_atype,zeros(Ty,1,B))

    generatorloss = 0.0
    mem = encode(bm.s, src, src_mask) # -> ((H, Tx, B), (H*Dx, Tx, B))
    cell = fill!(similar(mem[1], size(mem[1],1), size(mem[1],3), 1), 0)
    for t in 1:size(tgt,3)
        input = (@view tgt[:,:,t:t])   # -> (Ey,B,1)
        cell, src_attn_vector, srcalignments = decode(bm.s, input, mem, cell) 
        generatorloss += bm.p(src_attn_vector, srcalignments, src_attention_maps, tgtalignments, tgt_attention_maps,
                                                                            generate_targets, src_copy_targets, tgt_copy_targets)
    end
    return generatorloss
end



#=
## DeepBiaffineGraphDecoder
edge_heads = parser_inputs["edge_heads"]     
edgelabels = parser_inputs["edge_labels"]   
corefs = parser_inputs["corefs"]      #TODO: this doesnt come correct      
parser_mask = parser_inputs["mask"] 
graph_decoder_loss = bm.g(hiddens, edge_heads, edgelabels, corefs, parser_mask)

loss = pointergenerator_loss + graph_decoder_loss
return loss
=#