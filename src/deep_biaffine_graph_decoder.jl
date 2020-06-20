include("s2s.jl")
include("linear.jl")
include("biaffine_attention.jl")
include("bilinear.jl")

_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})


struct DeepBiaffineGraphDecoder
    edgenode_h_linear::Linear             # decoderhiddens(Hy, B, Ty) ->  (edgenode_hiddensize,  B, Ty)
    edgenode_m_linear::Linear             # decoderhiddens(Hy, B, Ty) ->  (edgenode_hiddensize,  B, Ty)
    edgelabel_h_linear::Linear            # decoderhiddens(Hy, B, Ty) ->  (edgelabel_hiddensize, B, Ty)
    edgelabel_m_linear::Linear            # decoderhiddens(Hy, B, Ty) ->  (edgelabel_hiddensize, B, Ty)
    biaffine_attention::BiaffineAttention # edgenode_h_linear, edgenode_m_linear, masks -> edgenode_scores
    edgelabel_bilinear::BiLinear          # edgelabel_h_linear, edgelabel_m_linear, masks -> edgelabel_scores
end


# ## Model constructor
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
    biaffine_attention = BiaffineAttention(edgenode_hidden_size, edgenode_hidden_size)
    edgelabel_bilinear = BiLinear(edgelabel_hidden_size, edgelabel_hidden_size, num_labels)
    DeepBiaffineGraphDecoder(edgenode_h_linear, edgenode_m_linear, edgelabel_h_linear, edgelabel_m_linear, biaffine_attention, edgelabel_bilinear)
end

#-
Hy, edgenode_hidden_size, edgelabel_hidden_size, num_labels= 512,256,128,141
g = DeepBiaffineGraphDecoder(Hy, edgenode_hidden_size, edgelabel_hidden_size, num_labels)
@testset "Testing DeepBiaffineGraphDecoder constructor" begin
    @test size(g.edgenode_h_linear.w) == (edgenode_hidden_size, Hy)
    @test size(g.edgelabel_h_linear.w) == (edgelabel_hidden_size, Hy)
    hinput = convert(_atype,randn(Hy, 64, 2))
    hinput = reshape(hinput, (Hy,:)) 
    @test size(g.edgenode_h_linear(hinput),1) == (edgenode_hidden_size)
end

#-remove here
enc_inputs, dec_inputs, generator_inputs, parser_inputs = prepare_batch_input(batch)
encoder_input, encoder_mask = prepare_encode_input(enc_inputs)
mem = encode(m, encoder_input, encoder_mask)
decoder_input = prepare_decode_input(enc_inputs, dec_inputs, parser_inputs)
cell = fill!(similar(mem[1], size(mem[1],1), size(mem[1],3), 2), 0) #TODO: give input 1by1, cell should be (Hy, B, 1)
hiddens, src_attn_vector, srcalignments, tgt_attn_vector, tgtalignments = decode(m, decoder_input, mem, cell)
edge_heads = parser_inputs["edge_heads"]     
edgelabels = parser_inputs["edgelabels"]   
corefs = parser_inputs["corefs"]            
mask = convert(_atype, zeros(1,64)) #TODO: fix here, parser_inputs["mask"]                 
# size.([hiddens, edge_heads, edgelabels, corefs, mask])
# Hy, B, Ty = size(hiddens)


g(hiddens, edge_heads, edgelabels, corefs, mask)

function (g::DeepBiaffineGraphDecoder)(hiddens, edge_heads, edgelabels, corefs, mask)
    # hiddens:      (Hy, B, Ty) memory_bank?
    # edge_heads:   (1(Ty-1?), B)
    # edgelabels:   (1(Ty-1?), B)
    # corefs:       (2(Ty?), B)
    # mask:         (2(Ty?), B)
    input_size = size(hiddens, 1)
    num_nodes  = size(mask, 1)
    hiddens, edge_heads, edgelabels, corefs, mask = add_head_sentinel(hiddens, edge_heads, edgelabels, corefs, mask)
    (edgenode_h, edgenode_m), (edgelabel_h, edgelabel_m) = encode(g, hiddens)
    edgenode_scores = get_edgenode_scores(g, edgenode_h, edgenode_m, mask)


    #=
    TODO:
    edge_node_nll, edgelabel_nll = self.get_loss(
            edgelabel_h, edgelabel_m, edge_node_scores, edge_heads, edgelabels, mask)

    pred_edge_heads, pred_edgelabels = self.decode(
            edgelabel_h, edgelabel_m, edge_node_scores, corefs, mask)
    =#

end

## Add a dummy ROOT at the beginning of each node sequence.
#
function add_head_sentinel(hiddens, edge_heads, edgelabels, corefs, mask)
    # hiddens:      (Hy, B, Ty) memory_bank?
    # edge_heads:   (1(ty-1?), B)
    # edgelabels:  (1(ty-1?), B)
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
    if !isnothing(edgelabels)
        edgelabels = vcat(edgelabels,  zeros(1,B))
    end
    if !isnothing(corefs)
        corefs = vcat(corefs,  zeros(1,B))
    end
    mask = vcat(mask, convert(_atype,ones(1,B)))
    return hiddens, edge_heads, edgelabels, corefs, mask
end



## Map contextual representation into specific space (w/ lower dimensionality)
#
function encode(g::DeepBiaffineGraphDecoder, hiddens)
    # hiddens:      (Hy, B, Ty) memory_bank?
    # -> edgenode   (head, modifier): ((B, Ty, edgenode_hidden_size), (B, Ty, edgenode_hidden_size))
    # -> edgelabel  (head, modifier): ((B, Ty, edgelabel_hidden_size),(B, Ty, edgelabel_hidden_size))

    Hy, B, Ty = size(hiddens)
    hiddens = reshape(hiddens, (Hy,:))                    # -> (Hy, B*Ty)
    edgenode_h = elu.(g.edgenode_h_linear(hiddens))       # -> (edgenode_hiddensize, B*Ty)
    edgenode_h = reshape(edgenode_h, (:, B, Ty))          # -> (edgenode_hiddensize, B,Ty)
    edgenode_m = elu.(g.edgenode_m_linear(hiddens))       # -> (edgenode_hiddensize, B*Ty)
    edgenode_m = reshape(edgenode_m, (:, B, Ty))          # -> (edgenode_hiddensize, B,Ty)

    edgelabel_h = elu.(g.edgelabel_h_linear(hiddens))     # -> (edgelabel_hiddensize, B*Ty)
    edgelabel_h = reshape(edgelabel_h, (:, B, Ty))        # -> (edgelabel_hiddensize, B,Ty)
    edgelabel_m = elu.(g.edgelabel_m_linear(hiddens))     # -> (edgelabel_hiddensize, B*Ty)
    edgelabel_m = reshape(edgelabel_m, (:, B, Ty))        # -> (edgelabel_hiddensize, B,Ty)
    
    # TODO: apply 2d dropout to edgenode and edgelabel, does it differ if we concat,dropout and chunk them again, or can we apply dropout them seperately?
    # edgenode  = cat(edgenode_h, edgenode_m, dims=3)     # -> (edgenode_hiddensize, B, Ty*2)
    # edgelabel = cat(edgelabel_h, edgelabel_m, dims=3)   # -> (edgelabel_hiddensize, B, Ty*2)
    return (edgenode_h, edgenode_m), (edgelabel_h, edgelabel_m)
end


function get_edgenode_scores(g, edgenode_h, edgenode_m, mask)
    # edgenode_h: (edgenode_hiddensize, B, Ty)
    # edgenode_m: (edgenode_hiddensize, B, Ty)
    # mask:       (Ty, B)
    # -> edgenode_scores:(B, Tencoder, Tdecoder, num_labels)
    edgenode_scores = g.biaffine_attention(edgenode_h, edgenode_m, mask_d=mask, mask_e=mask)
    return edgenode_scores
end
