include("linear.jl")
include("biaffine_attention.jl")
include("bilinear.jl")
include("maximum_spanning_tree.jl")
include("metrics.jl")
using Statistics

_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})


struct DeepBiaffineGraphDecoder
    edgenode_h_linear::Linear             # decoderhiddens(Hy, B, Ty) ->  (edgenodehiddensize,  B, Ty)
    edgenode_m_linear::Linear             # decoderhiddens(Hy, B, Ty) ->  (edgenodehiddensize,  B, Ty)
    edgelabel_h_linear::Linear            # decoderhiddens(Hy, B, Ty) ->  (edgelabelhiddensize, B, Ty)
    edgelabel_m_linear::Linear            # decoderhiddens(Hy, B, Ty) ->  (edgelabelhiddensize, B, Ty)
    biaffine_attention::BiaffineAttention # edgenode_h_linear, edgenode_m_linear, masks -> edgenode_scores
    edgelabel_bilinear::BiLinear          # edgelabel_h_linear, edgelabel_m_linear, masks -> edgelabel_scores
    head_sentinel
    metrics::GraphMetrics
end


# ## Model constructor
#
# The `DeepBiaffineGraphDecoder` constructor takes the following arguments:
# * `inputsize`: size of the hidden vectors of the decoder
# * `edgenodehiddensize`: number of edgenodehiddensize;  transform representations into a space for edge node heads and edge node modifiers
# * `edgelabelhiddensize`: number of edgelabelhiddensize; transform representations into a space for edge label heads and edge label modifiers
# * `num_labels`: number of head tags
function DeepBiaffineGraphDecoder(inputsize::Int, edgenodehiddensize::Int, edgelabelhiddensize::Int, num_labels::Int)
    edgenode_h_linear  = Linear(inputsize, edgenodehiddensize)
    edgenode_m_linear  = Linear(inputsize, edgenodehiddensize)
    edgelabel_h_linear = Linear(inputsize, edgelabelhiddensize)
    edgelabel_m_linear = Linear(inputsize, edgelabelhiddensize)
    # TODO: dropout. encode_dropout = torch.nn.Dropout2d(p=dropout)
    biaffine_attention = BiaffineAttention(edgenodehiddensize, edgenodehiddensize)
    edgelabel_bilinear = BiLinear(edgelabelhiddensize, edgelabelhiddensize, num_labels)
    head_sentinel  = param(inputsize,1,1)
    metrics = GraphMetrics(0,0,0,0,0,0,0,0,0)
    DeepBiaffineGraphDecoder(edgenode_h_linear, edgenode_m_linear, edgelabel_h_linear, edgelabel_m_linear, biaffine_attention, edgelabel_bilinear, head_sentinel, metrics)
end


function (g::DeepBiaffineGraphDecoder)(hiddens, parsermask, edgeheads, edgelabels)
    Hy, B, num_nodes = size(hiddens)
    parsermask = _atype(parsermask)
    edgeheads  = _atype(edgeheads)
    edgelabels = _atype(edgelabels)

    ;@size parsermask (B, num_nodes)
    ;@size edgeheads  (B, num_nodes) 
    ;@size edgelabels (B, num_nodes)

    ## Add dummy root at the beginning of sequence
    Ty = num_nodes+1
    dummy = convert(_atype, zeros(Hy,B,1))
    head_sentinel = g.head_sentinel .+ dummy   ;@size head_sentinel (Hy,B,1)
    hiddens = cat(head_sentinel, hiddens, dims=3)                                                ;@size hiddens (Hy,B,Ty)
    if !isnothing(edgeheads); edgeheads = cat(_atype(zeros(B,1)), edgeheads, dims=2); end        ;@size edgeheads  (B,Ty)
    if !isnothing(edgelabels); edgelabels = cat(_atype(zeros(B,1)), edgelabels, dims=2); end     ;@size edgelabels (B,Ty)
    parsermask = cat(_atype(ones(B,1)), parsermask, dims=2)                                      ;@size parsermask (B,Ty)
    hiddens = permutedims(hiddens, [1,3,2]) ;@size hiddens (Hy,Ty,B)
    hidden_rs = reshape(hiddens, :,Ty*B)

    ## Encode nodes
    edgenode_h = elu.(g.edgenode_h_linear(hidden_rs))        ;@size edgenode_h (edgenode_hiddensize, Ty*B)
    edgenode_h = permutedims(reshape(edgenode_h, :,Ty,B), [3,2,1]) # B, T, H
    edgenode_m = elu.(g.edgenode_m_linear(hidden_rs))        ;@size edgenode_m (edgenode_hiddensize, Ty*B)
    edgenode_m = permutedims(reshape(edgenode_m, :,Ty,B), [3,2,1]) # B, T, H

    edgelabel_h = elu.(g.edgelabel_h_linear(hidden_rs))      ;@size edgelabel_h (edgelabel_hiddensize, Ty*B)
    edgelabel_h = permutedims(reshape(edgelabel_h, :,Ty,B), [3,2,1]) # B, T, H
    edgelabel_m = elu.(g.edgelabel_m_linear(hidden_rs))      ;@size edgelabel_m (edgelabel_hiddensize, Ty*B)
    edgelabel_m = permutedims(reshape(edgelabel_m, :,Ty,B), [3,2,1]) # B, T, H
    #TODO: dropout the encoded nodes

    ## Calculate edgeheads scores & loss
    edgenode_scores = g.biaffine_attention(edgenode_h, edgenode_m, parsermask) ;@size edgenode_scores (B,1,Ty,Ty)
    edgenode_scores = reshape(edgenode_scores, B,Ty,Ty) 

    _, max_len, _ = size(edgenode_scores)
    edgeheads_indices = convert(Array{Int32}, edgeheads)
    edgeheadloss = nll3d(permutedims(edgenode_scores, [1,3,2]), edgeheads_indices)

    ## Calculate edgelabels scores & loss
    edgelabel_h = permutedims(edgelabel_h,[2,1,3])

    #Select only the head representations
    _edgelabel_h = reshape(reshape(edgelabel_h,:,edgelabel_hiddensize)[permutedims(edgeheads_indices,[2,1]),:], Ty,B,:)         ;@size _edgelabel_h (Ty,B,edgelabel_hiddensize)
    _edgelabel_h = permutedims(_edgelabel_h, [3,2,1])                                                                           ;@size _edgelabel_h (edgelabel_hiddensize,B,Ty)
    edgelabel_scores = g.edgelabel_bilinear(_edgelabel_h, permutedims(edgelabel_m, [3,1,2]))                                    ;@size edgelabel_scores (B,Ty, num_edgelabels)
    edgelabels_indides = convert(Array{Int32}, edgelabels)
    edgelabelloss = nll3d(edgelabel_scores, edgelabels_indides)

    ## Graphloss calculation
    graphloss = sum(edgeheadloss + edgelabelloss)
    
    ## Greedy decoding for heads and labels
    edgenode_scores= permutedims(edgenode_scores, [3,2,1]) ;@size edgenode_scores (Ty,Ty,B)
    ;@size parsermask (B,Ty); ;@size edgelabel_scores (B,Ty,num_edgelabels)
    diagonal(A::AbstractMatrix, k::Integer=0) = view(A, diagind(A, k)) # Set diagonal elements to -inf to prevent edge returns node itself (no loop)
    a = zeros(Ty,Ty); diagonal(a) .= -Inf
    a = reshape(a, (Ty,Ty,1))
    _edgenode_scores = edgenode_scores .+ convert(_atype,a)             ;@size _edgenode_scores (Ty,Ty,B)                       
    minus_mask = 1 .- parsermask
    settoinf(x) = x==1.0 ? x= -1e8 : x=x                                               
    minus_mask = settoinf.(Array(minus_mask))'                          ;@size minus_mask (Ty,B)
    minus_mask = reshape(minus_mask, (1,Ty,B))                                         
    _edgenode_scores = _edgenode_scores .+ _atype(minus_mask)           ;@size _edgenode_scores (Ty,Ty,B)               
    minus_mask = reshape(minus_mask, (Ty,1,B))                                         
    _edgenode_scores = _edgenode_scores .+ _atype(minus_mask)           ;@size _edgenode_scores (Ty,Ty,B)               
    
    ## Predictions of edge_heads
    ina(x) = return x[2]     # remove cartesian type 
    pred_edgeheads = argmax(value(_edgenode_scores), dims=2)  
    pred_edgeheads = reshape(ina.(pred_edgeheads), (Ty,B))'             ;@size pred_edgeheads (B,Ty)
    

    ## Predictions of edge_labels
    pred_edgeheads_indices  = convert(Array{Int32}, (pred_edgeheads))
    _edgelabel_h = reshape(reshape(edgelabel_h,:,edgelabel_hiddensize)[permutedims(pred_edgeheads_indices,[2,1]),:], Ty,B,:)   ;@size _edgelabel_h (Ty,B,edgelabel_hiddensize)
    _edgelabel_h = permutedims(_edgelabel_h, [3,2,1])                                                    ;@size _edgelabel_h (edgelabel_hiddensize,B,Ty)
    _edgelabel_scores = g.edgelabel_bilinear(_edgelabel_h, permutedims(edgelabel_m, [3,1,2]))            ;@size edgelabel_scores (B,Ty, num_edgelabels)
    inpa(x) = return x[3]     # remove cartesian type 
    pred_edgelabels = argmax(value(_edgelabel_scores), dims=3)               ;@size pred_edgelabels (B,Ty,1)
    pred_edgelabels = reshape(inpa.(pred_edgelabels), (B,Ty))                ;@size pred_edgelabels (B,Ty)


    ## Graph Decoder Metrics
    ;@size edgeheads_indices (B,Ty) ;@size edgelabels_indides (B,Ty) ;@size parsermask (B,Ty) ;@size pred_edgeheads (B,Ty) ;@size pred_edgelabels (B,Ty)
    # Exclude dummy root 
    _edgeheads = edgeheads_indices[:,2:end]               ;@size _edgeheads (B,Ty-1)
    _edgelabels = edgelabels_indides[:,2:end]             ;@size _edgelabels (B,Ty-1)
    _parsermask = parsermask[:,2:end]               ;@size _parsermask (B,Ty-1)
    _pred_edgeheads = pred_edgeheads[:,2:end]       ;@size _pred_edgeheads (B,Ty-1)
    _pred_edgelabels = pred_edgelabels[:,2:end]     ;@size _pred_edgelabels (B,Ty-1)
    g.metrics(_pred_edgeheads,_pred_edgelabels, _edgeheads, _edgelabels, _parsermask, value(graphloss), value(edgeheadloss), value(edgelabelloss))
    
    return graphloss
end


function nll3d(y,a; dims=2, average=true)
    indices = findindices(y, a, dims=dims)
    lp = logp(y,dims=3)[indices] # Changed dims for our case.
    average ? -mean(lp) : -sum(lp)
end


# TODO: test this function
function mst_decode(g, edge_label_h, edge_label_m, edge_node_scores, corefs, mask)
    # Inputs: edge_label_h        -> Hh, B, Ty
    #         edge_label_m        -> Hm, B, Ty
    #         edge_node_scores    -> Ty, Ty, B
    #         corefs              -> B, Ty
    #         mask                -> B, Ty
    # The inputs should ideally be on the cpu, because the operations are too expensive and not efficient on gpu
    # Output: Array of B lists of heads for each sentence in the batch
    #         Array of B lists which correspond to the labels of each edge

    @assert size(edge_label_h)[2:end] == size(edge_label_m)[2:end]
    B, Ty = size(edge_label_h)[2:end]
    Hh = size(edge_label_h, 1)
    Hm = size(edge_label_m, 1)

    # First we need to duplicate edge_label_h Ty times to get H, B, Ty, Ty
    # Along 4th axis
    edge_label_h = [edge_label_h[i, j, k] for i=1:Hh, j=1:B, k=1:Ty, l=1:Ty]   # -> Hh, B, Ty, Ty
    # Along 3rd axis
    edge_label_m = [edge_label_m[i, j, k] for i=1:Hm, j=1:B, l=1:Ty, k=1:Ty]   # -> Hm, B, Ty, Ty

    # Get label scores
    edge_label_scores = g.edgelabel_bilinear(edge_label_h, edge_label_m)       # -> B, Ty, Ty, O
    edge_label_scores = log.(softmax(edge_label_scores, dims=4))               # -> B, Ty, Ty, O

    _etype = typeof(edge_node_scores)
    # Set invalid elements to -inf
    minus_mask = 1 .- mask
    settoinf(x) = x==0.0 ? x= -Inf : x=x                                               
    minus_mask = settoinf.(minus_mask)'                                 ;@size minus_mask (Ty,B)
    minus_mask = reshape(minus_mask, (1,Ty,B))                                         
    edge_node_scores = edge_node_scores .+ _etype(minus_mask)           ;@size edge_node_scores (Ty,Ty,B)
    minus_mask = reshape(minus_mask, (Ty, 1, B))
    edge_node_scores = edge_node_scores .+ _etype(minus_mask)           ;@size edge_node_scores (Ty,Ty,B)
    # TODO: Double check that the softmax is on the 1st dim
    # Original line: https://github.com/sheng-z/stog/blob/f541f004d3c016ae3#5099b708979b1bca15a13bc/stog/modules/decoders/deep_biaffine_graph_decoder.py#L179
    edge_node_scores = log.(softmax(edge_node_scores, dims=1))                 # -> Ty, Ty, B
    edge_node_scores = permutedims(edge_node_scores, [3, 1, 2])                # -> B, Ty, Ty

    batch_energy = exp.(reshape(edge_node_scores, B, Ty, Ty, 1).+edge_label_scores)  # -> B, Ty, Ty, O

    all_heads = []
    all_labels = []

    for idx in 1:B
        energy = batch_energy[idx, :, :, :]
        energy = reshape(Ty, Ty, O)                                            # -> Ty, Ty, O
        # In the original implementation they set the energy of first node with all other nodes to 0
        # so the head does not have more than one child.   

        # Assuming that mask is 1 for pads
        N = Ty-sum(mask[idx, :])+1  # TODO: double check this. +1 because the first node is a pad as well, 
        if corefs != nothing
            heads, labels = decode_mst(energy, N, true)
        else            
            heads, labels = decode_mst_with_corefs(energy, corefs, N, true)
        end

        heads[1] = 0
        labels[1] = 0

        push!(all_heads, heads)
        push!(all_labels, labels)
    end

    all_heads, all_labels
end
