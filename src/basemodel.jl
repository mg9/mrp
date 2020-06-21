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


function (bm::BaseModel)(batch)
    ## S2S
    enc_inputs, dec_inputs, generator_inputs, parser_inputs = prepare_batch_input(batch)
    encoder_input, encoder_mask = prepare_encode_input(enc_inputs)
    mem = encode(bm.s, encoder_input, encoder_mask)
    (key,val) = mem
    decoder_input = prepare_decode_input(enc_inputs, dec_inputs, parser_inputs)
    cell = randn!(similar(key, size(key,1), size(key,3), 2))      # -> (H,B,Ty)
    hiddens, src_attn_vector, srcalignments, tgt_attn_vector, tgtalignments = decode(bm.s, decoder_input, (key,val), cell)

    ## PointerGenerator
    src_attentions = srcalignments
    tgt_attentions = tgtalignments
    generate_targets = generator_inputs["vocab_targets"]
    src_copy_targets = generator_inputs["copy_targets"]
    tgt_copy_targets = generator_inputs["coref_targets"]
    src_attention_maps = generator_inputs["copy_attention_maps"]
    tgt_attention_maps = generator_inputs["coref_attention_maps"]
    pointergenerator_loss = bm.p(hiddens, src_attentions, src_attention_maps, tgt_attentions, tgt_attention_maps,
                                                                        generate_targets, src_copy_targets, tgt_copy_targets)

    ## DeepBiaffineGraphDecoder
    edge_heads = parser_inputs["edge_heads"]     
    edgelabels = parser_inputs["edge_labels"]   
    corefs = parser_inputs["corefs"]            
    mask = parser_inputs["mask"] 
    graph_decoder_loss = bm.g(hiddens, edge_heads, edgelabels, corefs, mask)
    
    loss = pointergenerator_loss + graph_decoder_loss
    return loss
end


