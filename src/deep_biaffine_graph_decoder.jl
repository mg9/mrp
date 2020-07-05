include("linear.jl")
include("biaffine_attention.jl")
include("bilinear.jl")
include("maximum_spanning_tree.jl")

_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})


struct DeepBiaffineGraphDecoder
    edgenode_h_linear::Linear             # decoderhiddens(Hy, B, Ty) ->  (edgenodehiddensize,  B, Ty)
    edgenode_m_linear::Linear             # decoderhiddens(Hy, B, Ty) ->  (edgenodehiddensize,  B, Ty)
    edgelabel_h_linear::Linear            # decoderhiddens(Hy, B, Ty) ->  (edgelabelhiddensize, B, Ty)
    edgelabel_m_linear::Linear            # decoderhiddens(Hy, B, Ty) ->  (edgelabelhiddensize, B, Ty)
    biaffine_attention::BiaffineAttention # edgenode_h_linear, edgenode_m_linear, masks -> edgenode_scores
    edgelabel_bilinear::BiLinear          # edgelabel_h_linear, edgelabel_m_linear, masks -> edgelabel_scores
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
    DeepBiaffineGraphDecoder(edgenode_h_linear, edgenode_m_linear, edgelabel_h_linear, edgelabel_m_linear, biaffine_attention, edgelabel_bilinear)
end

#=
#-
@testset "Testing DeepBiaffineGraphDecoder constructor" begin
    Hy, edgenode_hidden_size, edgelabel_hidden_size, num_labels= 512,256,128,141
    g = DeepBiaffineGraphDecoder(Hy, edgenode_hidden_size, edgelabel_hidden_size, num_labels)
    @test size(g.edgenode_h_linear.w) == (edgenode_hidden_size, Hy)
    @test size(g.edgelabel_h_linear.w) == (edgelabel_hidden_size, Hy)
    hinput = convert(_atype,randn(Hy, 64, 2))
    hinput = reshape(hinput, (Hy,:)) 
    @test size(g.edgenode_h_linear(hinput),1) == (edgenode_hidden_size)
end
=#

function (g::DeepBiaffineGraphDecoder)(hiddens, parsermask, edgeheads, edgelabels)
    Hy,B,Ty = size(hiddens,1), size(hiddens,2), size(hiddens,3)+1

    ;@size parsermask (B, Ty-1); @size edgeheads (B, Ty-1);  @size edgelabels (B, Ty-1)
    head_sentinel = param(Hy, B, 1)  
    hiddens = cat(head_sentinel, hiddens, dims=3)                                           ;@size hiddens (Hy,B,Ty)
    if !isnothing(edgeheads); edgeheads = cat(zeros(B,1), edgeheads, dims=2); end           ;@size edgeheads  (B,Ty)
    if !isnothing(edgelabels); edgelabels = cat(zeros(B,1), edgelabels, dims=2); end        ;@size edgelabels (B,Ty)
    parsermask = cat(ones(B,1), parsermask, dims=2)                                         ;@size parsermask (B,Ty)

    ## Encode nodes
    edgenode_h = elu.(g.edgenode_h_linear(reshape(hiddens, :,B*Ty)))        ;@size edgenode_h (edgenode_hiddensize, B*Ty)
    edgenode_m = elu.(g.edgenode_m_linear(reshape(hiddens, :,B*Ty)))        ;@size edgenode_m (edgenode_hiddensize, B*Ty)
    edgelabel_h = elu.(g.edgelabel_h_linear(reshape(hiddens, :,B*Ty)))      ;@size edgelabel_h (edgelabel_hiddensize, B*Ty)
    edgelabel_m = elu.(g.edgelabel_m_linear(reshape(hiddens, :,B*Ty)))      ;@size edgelabel_m (edgelabel_hiddensize, B*Ty)
    edgelabel_h = reshape(edgelabel_h,(:, B,Ty))                           
    edgelabel_m = reshape(edgelabel_m,(:, B,Ty))                           

    ## Calculate edgeheads scores & loglikelihood
    edgenode_scores = g.biaffine_attention(edgenode_h, edgenode_m, parsermask) ;@size edgenode_scores (Ty,Ty,B)
    tmpsoftmax_mask = (parsermask .== 0)'                       ;@size tmpsoftmax_mask (Ty,B)
    tmpsoftmax_mask = _atype(reshape(tmpsoftmax_mask,1,Ty,B))   ;@size tmpsoftmax_mask (1, Ty,B)
    masked_scores = (edgenode_scores .* tmpsoftmax_mask)        ;@size masked_scores (Ty, Ty,B) 
    edgenode_ll =  -log.(softmax(masked_scores, dims=2))        # TODO: scores for 0-masks is still ~0.11

    ## Calculate edgelabels scores & loglikelihood
    ;@size edgelabel_h (edgelabel_hiddensize, B, Ty) ;@size edgeheads  (B,Ty)
    ;@size edgelabel_m (edgelabel_hiddensize,B,Ty)   ;@size edgelabels (B,Ty)
    
    # returns the node representations with given indices
    function getreps(ind, arr)
        # ind:(T,B), arr: (T,B,H) -> T,B,H
        results =[]
        (t,b,h) = size(arr)
        for i in 1:b
           for j in 1:t
               push!(results, arr[ind[j,i],i,:])
           end
       end
       return reshape(reshape(vcat(results...), h,t*b)', t,b,h) 
    end
    edgeheads_inds  = Integer.(edgeheads)  .+1  #increase for indices 0
    edgelabels_inds = Integer.(edgelabels) .+1  #increase for indices 0
    edgelabel_h = permutedims(edgelabel_h, [3,2,1])                             ;@size edgelabel_h (Ty,B,edgelabel_hiddensize)
    edgelabel_h = getreps(edgeheads_inds', edgelabel_h)                         ;@size edgelabel_h (Ty,B,edgelabel_hiddensize)
    edgelabel_h = permutedims(edgelabel_h, [3,2,1])                             ;@size edgelabel_h (edgelabel_hiddensize,B,Ty)
    edgelabel_scores = g.edgelabel_bilinear(edgelabel_h, edgelabel_m)           ;@size edgelabel_scores (B,Ty, num_edgelabels)
    edgelabel_ll = -log.(softmax(edgelabel_scores, dims=3))
    edgelabel_ll = permutedims(edgelabel_ll, [2,3,1])                           ;@size edgelabel_ll (Ty, num_edgelabels,B)

    ## Calculate edgeheads and edgelabel losses 
    _edgeheads  = reshape(edgeheads_inds',  Ty,B)
    _edgelabels = reshape(edgelabels_inds', Ty,B)
    edgeheadloss = edgelabelloss = 0.0
    for t in 2:Ty # Exclude dummy root
        for b in 1:B
            edgeheadloss +=edgenode_ll[t,_edgeheads[t,b],b]
            edgelabelloss+=edgelabel_ll[t,_edgelabels[t,b],b]
        end
    end
    graphloss = sum(edgeheadloss + edgelabelloss)


    ## Greedy decoding for heads and labels
    ;@size edgenode_scores (Ty,Ty,B) ;@size parsermask (B,Ty); ;@size edgelabel_scores (B,Ty,num_edgelabels)
    diagonal(A::AbstractMatrix, k::Integer=0) = view(A, diagind(A, k)) # Set diagonal elements to -inf to prevent edge returns node itself (no loop)
    a = zeros(Ty,Ty); diagonal(a) .= -Inf
    a = reshape(a, (Ty,Ty,1))
    _edgenode_scores = edgenode_scores .+ convert(_atype,a)             ;@size _edgenode_scores (Ty,Ty,B)                       
    minus_mask = 1 .- parsermask
    settoinf(x) = x==0.0 ? x= -Inf : x=x                                               
    minus_mask = settoinf.(minus_mask)'                                 ;@size minus_mask (Ty,B)
    minus_mask = reshape(minus_mask, (1,Ty,B))                                         
    _edgenode_scores = _edgenode_scores .+ _atype(minus_mask)           ;@size _edgenode_scores (Ty,Ty,B)               
    minus_mask = reshape(minus_mask, (Ty,1,B))                                         
    _edgenode_scores = _edgenode_scores .+ _atype(minus_mask)           ;@size _edgenode_scores (Ty,Ty,B)               
    
    ## Predictions of edge_heads
    ina(x) = return x[2]     # remove cartesian type 
    pred_edgeheads = argmax(value(_edgenode_scores), dims=2)  
    pred_edgeheads = reshape(ina.(pred_edgeheads), (Ty,B))'             ;@size pred_edgeheads (B,Ty)

    ## Predictions of edge_labels
    inpa(x) = return x[3]     # remove cartesian type 
    pred_edgelabels = argmax(value(edgelabel_scores), dims=3)           ;@size pred_edgelabels (B,Ty,1)
    pred_edgelabels = reshape(inpa.(pred_edgelabels), (B,Ty))           ;@size pred_edgelabels (B,Ty)


    ## Graph Decoder Metrics
    #TODO: other metrics s.t. unlabeled exact match, labeled exact match etc., move metrics part into a function
    unlabeled_attachment_score = labeled_attachment_score = edgeloss = edgenode_loss = edgelabel_loss = 0.0
    ;@size _edgeheads (Ty,B) ;@size _edgelabels (Ty,B) ;@size parsermask (B,Ty) ;@size pred_edgeheads (B,Ty) ;@size pred_edgelabels (B,Ty)
    _edgeheads = _edgeheads'[:,2:end]               ;@size _edgeheads (B,Ty-1)
    _edgelabels = _edgelabels'[:,2:end]             ;@size _edgelabels (B,Ty-1)
    _parsermask = parsermask[:,2:end]               ;@size _parsermask (B,Ty-1)
    _pred_edgeheads = pred_edgeheads[:,2:end]       ;@size _pred_edgeheads (B,Ty-1)
    _pred_edgelabels = pred_edgelabels[:,2:end]     ;@size _pred_edgelabels (B,Ty-1)

    #TODO: What about ignore label class mask?
    correct_indices = (_pred_edgeheads .== _edgeheads)  .*  (1 .-_parsermask)   ;@size correct_indices (B,Ty-1)
    correct_labels = (_pred_edgelabels .== _edgelabels) .* (1 .-_parsermask)    ;@size correct_labels (B,Ty-1)
    correct_labels_and_indices = correct_indices .* correct_labels              ;@size correct_labels_and_indices (B,Ty-1)
    unlabeled_correct = sum(correct_indices)               
    labeled_correct = sum(correct_labels_and_indices)
    total_sentences = B
    total_words =  sum(1 .- parsermask)
    total_loss = graphloss
    total_edgenode_loss = edgeheadloss
    total_edgelabel_loss = edgelabelloss

    if total_words > 0
        unlabeled_attachment_score = float(unlabeled_correct) / float(total_words)
        labeled_attachment_score = float(labeled_correct) / float(total_words)
        edgeloss = float(total_loss) / float(total_words)
        edgenode_loss = float(total_edgenode_loss) / float(total_words)
        edgelabel_loss = float(total_edgelabel_loss) / float(total_words)
    end

    UAS = unlabeled_attachment_score
    LAS = labeled_attachment_score
    EL  = edgeloss

    @info "GraphDecoder batch metrics" UAS=UAS LAS=LAS EL=EL total_words=total_words
    return graphloss
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
