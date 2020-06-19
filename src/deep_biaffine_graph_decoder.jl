include("s2s.jl")
include("linear.jl")
include("biaffine_attention.jl")
include("bilinear.jl")

_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})


struct DeepBiaffineGraphDecoder
    edgenode_h_linear::Linear       # decoderhiddens(Hy, B, Ty) ->  (edgenode_hiddensize,  B, Ty)
    edgenode_m_linear::Linear       # decoderhiddens(Hy, B, Ty) ->  (edgenode_hiddensize,  B, Ty)
    edgelabel_h_linear::Linear      # decoderhiddens(Hy, B, Ty) ->  (edgelabel_hiddensize, B, Ty)
    edgelabel_m_linear::Linear      # decoderhiddens(Hy, B, Ty) ->  (edgelabel_hiddensize, B, Ty)
    biaffine_attention::BiaffineAttention
    edge_label_bilinear::BiLinear
end


# ## Part 1. Model constructor
#
# The `DeepBiaffineGraphDecoder` constructor takes the following arguments:
# * `hidden`: size of the hidden vectors of the decoder
# * `edgenode_hiddensize`: number of edgenode_hiddensize;  transform representations into a space for edge node heads and edge node modifiers
# * `edgelabel_hiddensize`: number of edgelabel_hiddensize; transform representations into a space for edge label heads and edge label modifiers
# * `num_labels`: number of head tags
function DeepBiaffineGraphDecoder(inputsize::Int, edgenode_hiddensize::Int, edgelabel_hiddensize::Int, num_labels::Int)
    edgenode_h_linear  = Linear(inputsize, edgenode_hiddensize)
    edgenode_m_linear  = Linear(inputsize, edgenode_hiddensize)
    edgelabel_h_linear = Linear(inputsize, edgelabel_hiddensize)
    edgelabel_m_linear = Linear(inputsize, edgelabel_hiddensize)
    # TODO: dropout. encode_dropout = torch.nn.Dropout2d(p=dropout)
    biaffine_attention  = BiaffineAttention(edge_node_hidden_size, edge_node_hidden_size)
    edge_label_bilinear = BiLinear(edge_label_hidden_size, edge_label_hidden_size, num_labels)
    DeepBiaffineGraphDecoder(edgenode_h_linear, edgenode_m_linear, edgelabel_h_linear, edgelabel_m_linear, biaffine_attention, edge_label_bilinear)
end


enc_inputs, dec_inputs, generator_inputs , parser_inputs = prepare_batch_input(batch)
encoder_input, encoder_mask = prepare_encode_input(enc_inputs)
mem = encode(m, encoder_input, encoder_mask)
decoder_input = prepare_decode_input(enc_inputs, dec_inputs, parser_inputs)
cell = fill!(similar(mem[1], size(mem[1],1), size(mem[1],3), 2), 0) #TODO: give input 1by1, cell should be (Hy, B, 1)
hiddens, src_attn_vector, srcalignments, tgt_attn_vector, tgtalignments = decode(m, decoder_input, mem, cell)
edge_heads = parser_inputs["edge_heads"]     
edge_labels = parser_inputs["edge_labels"]   
corefs = parser_inputs["corefs"]            
mask = parser_inputs["mask"]                 
# size.([hiddens, edge_heads, edge_labels, corefs, mask])
edge_node_hidden_size = 256
edge_label_hidden_size = 128
num_labels = 141# TODO: Fix this
Hy, B, Ty = size(hiddens)
g = DeepBiaffineGraphDecoder(Hy, edge_node_hidden_size, edge_label_hidden_size, num_labels)



function (g::DeepBiaffineGraphDecoder)(hiddens, edge_heads, edge_labels, corefs, mask)
    # hiddens:      (Hy, B, Ty) memory_bank?
    # edge_heads:   (1(ty-1?), B)
    # edge_labels:  (1(ty-1?), B)
    # corefs:       (2(ty?), B)
    # mask:         (2(ty?), B)
    input_size = size(hiddens, 1)
    num_nodes  = size(mask, 1)
    hiddens, edge_heads, edge_labels, corefs, mask = add_head_sentinel(hiddens, edge_heads, edge_labels, corefs, mask)
    (edge_node_h, edge_node_m), (edge_label_h, edge_label_m) = encode(g, hiddens)
    edge_node_scores = get_edge_node_scores(g, edge_node_h, edge_node_m, mask)

end

## Add a dummy ROOT at the beginning of each node sequence.
#
function add_head_sentinel(hiddens, edge_heads, edge_labels, corefs, mask)
    # hiddens:      (Hy, B, Ty) memory_bank?
    # edge_heads:   (1(ty-1?), B)
    # edge_labels:  (1(ty-1?), B)
    # corefs:       (2(ty?), B)
    # mask:         (2(ty?), B)
    Hy, B, _ = size(hiddens)
    head_sentinel = param(Hy, 1, 1)                # -> (Hy, 1, 1) does this meet randn requirement?
    dummy = convert(_atype, zeros(1, B, 1))
    head_sentinel = head_sentinel .+ dummy         # -> (Hy, B, 1)
    hiddens = cat(head_sentinel, hiddens, dims=3)  # -> (Hy, B, Ty+1)
    if !isnothing(edge_heads)
        edge_heads = vcat(edge_heads,  zeros(1,B))
    end
    if !isnothing(edge_labels)
        edge_labels = vcat(edge_labels,  zeros(1,B))
    end
    if !isnothing(corefs)
        corefs = vcat(corefs,  zeros(1,B))
    end
    mask = vcat(mask, ones(1,B))
    return hiddens, edge_heads, edge_labels, corefs, mask
end



## Map contextual representation into specific space (w/ lower dimensionality)
#
function encode(g::DeepBiaffineGraphDecoder, hiddens)
    # hiddens:      (Hy, B, Ty) memory_bank?
    # -> edge_node  (head, modifier): ((B, Ty, edgenode_hidden_size), (B, Ty, edgenode_hidden_size))
    # -> edge_label (head, modifier): ((B, Ty, edgelabel_hidden_size),(B, Ty, edgelabel_hidden_size))

    hiddens = reshape(hiddens, (Hy,:))                    # -> (Hy, B*Ty)
    edge_node_h = elu.(g.edgenode_h_linear(hiddens))      # -> (edgenode_hiddensize, B*Ty)
    edge_node_h = reshape(edge_node_h, (:, B, Ty))        # -> (edgenode_hiddensize, B,Ty)
    edge_node_m = elu.(g.edgenode_m_linear(hiddens))      # -> (edgenode_hiddensize, B*Ty)
    edge_node_m = reshape(edge_node_m, (:, B, Ty))        # -> (edgenode_hiddensize, B,Ty)

    edge_label_h = elu.(g.edgelabel_h_linear(hiddens))    # -> (edgelabel_hiddensize, B*Ty)
    edge_label_h = reshape(edge_label_h, (:, B, Ty))      # -> (edgelabel_hiddensize, B,Ty)
    edge_label_m = elu.(g.edgelabel_m_linear(hiddens))    # -> (edgelabel_hiddensize, B*Ty)
    edge_label_m = reshape(edge_label_m, (:, B, Ty))      # -> (edgelabel_hiddensize, B,Ty)
    
    # TODO: apply 2d dropout to edge_node and edge_label, does it differ if we concat,dropout and chunk them again, or can we apply dropout them seperately?
    # edge_node  = cat(edge_node_h, edge_node_m, dims=3)     # -> (edgenode_hiddensize, B, Ty*2)
    # edge_label = cat(edge_label_h, edge_label_m, dims=3)   # -> (edgelabel_hiddensize, B, Ty*2)
    return (edge_node_h, edge_node_m), (edge_label_h, edge_label_m)
end


function get_edge_node_scores(g, edge_node_h, edge_node_m, mask)
    # edge_node_h: (edgenode_hiddensize, B, Ty)
    # edge_node_m: (edgenode_hiddensize, B, Ty)
    # mask: (Ty, B)
    edge_node_scores = g.biaffine_attention(edge_node_h, edge_node_m, mask_d=mask, mask_e=mask)
    return edge_node_scores
end




