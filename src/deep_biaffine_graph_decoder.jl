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
# * `inputsize`: size of the hidden vectors of the decoder
# * `edgenode_hiddensize`: number of edgenode_hiddensize;  transform representations into a space for edge node heads and edge node modifiers
# * `edgelabel_hiddensize`: number of edgelabel_hiddensize; transform representations into a space for edge label heads and edge label modifiers
# * `num_labels`: number of head tags
function DeepBiaffineGraphDecoder(inputsize::Int, edgenode_hiddensize::Int, edgelabel_hiddensize::Int, num_labels::Int)
    edgenode_h_linear  = Linear(inputsize, edgenode_hiddensize)
    edgenode_m_linear  = Linear(inputsize, edgenode_hiddensize)
    edgelabel_h_linear = Linear(inputsize, edgelabel_hiddensize)
    edgelabel_m_linear = Linear(inputsize, edgelabel_hiddensize)
    # TODO: dropout. encode_dropout = torch.nn.Dropout2d(p=dropout)
    biaffine_attention = BiaffineAttention(edgenode_hiddensize, edgenode_hiddensize)
    edgelabel_bilinear = BiLinear(edgelabel_hiddensize, edgelabel_hiddensize, num_labels)
    DeepBiaffineGraphDecoder(edgenode_h_linear, edgenode_m_linear, edgelabel_h_linear, edgelabel_m_linear, biaffine_attention, edgelabel_bilinear)
end

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


function (g::DeepBiaffineGraphDecoder)(hiddens, edge_heads, edgelabels, corefs, parser_mask)
    # hiddens:      (Hy, B, Ty) memory_bank?
    # edge_heads:   (1(Ty-1?), B)
    # edgelabels:   (1(Ty-1?), B)
    # corefs:       (Ty, B)
    # parser_mask:  (Ty, B)
    # -> loss

    ## Part1: Add a dummy ROOT at the beginning of each node sequence.

    Hy, B, Ty = size(hiddens)
    hiddens = hiddens[:,:,2:end]      # Exclude BOS symbol
    corefs = corefs[2:end, :]
    mask = parser_mask[2:end, :]
    head_sentinel = param(Hy, 1, 1)                       # -> (Hy, 1, 1) does this meet randn requirement?
    dummy = convert(_atype, zeros(1, B, 1))
    head_sentinel = head_sentinel .+ dummy                # -> (Hy, B, 1)
    hiddens = cat(head_sentinel, hiddens, dims=3)         # -> (Hy, B, Ty+1)
    if !isnothing(edge_heads); edge_heads = vcat(ones(1,B), edge_heads); end
    if !isnothing(edgelabels); edgelabels = vcat(zeros(1,B), edgelabels); end
    if !isnothing(corefs); corefs = vcat(zeros(1,B), corefs); end  #This may need to change
    mask = vcat(ones(1,B), mask)


    ## Part2: Encode 
    hiddens = reshape(hiddens, (Hy,:))                    # -> (Hy, B*Ty)
    edgenode_h = elu.(g.edgenode_h_linear(hiddens))       # -> (edgenode_hiddensize, B*Ty)
    edgenode_h = reshape(edgenode_h, (:, B, Ty))          # -> (edgenode_hiddensize, B,Ty)
    edgenode_m = elu.(g.edgenode_m_linear(hiddens))       # -> (edgenode_hiddensize, B*Ty)
    edgenode_m = reshape(edgenode_m, (:, B, Ty))          # -> (edgenode_hiddensize, B,Ty)
    edgelabel_h = elu.(g.edgelabel_h_linear(hiddens))     # -> (edgelabel_hiddensize, B*Ty)
    edgelabel_h = reshape(edgelabel_h, (:, B, Ty))        # -> (edgelabel_hiddensize, B,Ty)
    edgelabel_m = elu.(g.edgelabel_m_linear(hiddens))     # -> (edgelabel_hiddensize, B*Ty)
    edgelabel_m = reshape(edgelabel_m, (:, B, Ty))        # -> (edgelabel_hiddensize, B,Ty)
    

    ## Part3: Calculate edge node scores 
    edgenode_scores = g.biaffine_attention(edgenode_h, edgenode_m, mask_d=mask, mask_e=mask)             # -> (B, Tencoder, Tdecoder, num_labels)
    edgenode_scores = reshape(edgenode_scores, (size(edgenode_scores,1), size(edgenode_scores,2), :))    # -> (B, Tencoder, Tdecoder*num_labels=1)


    ## Part4: Get edge heads 
    diagonal(A::AbstractMatrix, k::Integer=0) = view(A, diagind(A, k)) # Set diagonal elements to -inf, to prevent edge between node itself (no loop)
    a = rand(Ty,Ty); diagonal(a) .= -Inf
    a = reshape(a, (1,Ty,Ty))
    edgenode_scores = edgenode_scores .+ convert(_atype,a)                              # -> (B, T, T)
    # TODO: check that part again, did we mask correctly?
    minus_mask = (1 .- mask) * -Inf                                                     # -> (T, B): Set invalid positions to -inf
    minus_mask = reshape(minus_mask, (1,Ty,B))                                           # -> (1, T, B)
    edgenode_scores = permutedims(edgenode_scores, [3,2,1])                             # -> (T, T, B)
    edgenode_scores =edgenode_scores .+convert(_atype,minus_mask)                       # -> (T, T, B) 
    # Compute naive predictions of edge_heads
    in(x) = return x[2]
    a = argmin(value(edgenode_scores), dims=2)
    b = reshape(a, size(a,1),size(a,3)) 
    edge_heads = in.(b)     # -> (T, B): remove cartesian type    


    ## Part5: Calculate edge label scores
    H, B, T = size(edgelabel_h)
    batch_index = reshape(collect(1:B), (B,1)) # Create indexing matrix for batch: [batch, 1]
    # TODO: Is it correct?
    # Select the heads' representations based on the gold/predicted heads.
    edgelabel_h = permutedims(edgelabel_h, [2,3,1])                                     # -> (B, T, edgelabel_hiddensize)
    edgelabel_h = reshape(edgelabel_h, size(edgelabel_h,1)* size(edgelabel_h,2), :)     # -> (B*T, edgelabel_hiddensize)
    edgeheads_rs = reshape(edge_heads, (B*T))                                          # -> (B*T)
    edgelabel_h = edgelabel_h[edgeheads_rs,:]                                           # -> (B*T, edgelabel_hiddensize)
    edgelabel_h = reshape(edgelabel_h, (B,T,H))                                         # -> (B, T, edgelabel_hiddensize)
    edgelabel_h = permutedims(edgelabel_h, [3,1,2])                                     # -> (edgelabel_hiddensize, B, T)
    edgelabel_scores = g.edgelabel_bilinear(edgelabel_h, edgelabel_m)                   # -> (B, T, numlabels(headtags))   


    ## Part6: Get edge labels
    a = argmin(value(edgelabel_scores), dims=3)
    b = reshape(a, size(a,1),size(a,2)*size(a,3)) 
    edgelabels = in.(b)     # -> (T, B): remove cartesian type 


    ## Part7: Calculate loss
    # TODO: Check this masked sotfmax
    mask = reshape(mask, (1,T,B))                                                       # -> (1, T, B)
    #edgenode_scores = permutedims(edgenode_scores, [3,2,1])                            # -> (T, T, B)

    edgenode_scores = edgenode_scores .*convert(_atype,mask)                            # -> (T, T, B) 
    edgenode_log_likelihood = softmax(edgenode_scores,dims=1)                           # -> (T, T, B)
    edgelabel_log_likelihood = log.(softmax(edgelabel_scores, dims=3))                  # -> (B,T,num_labels)
    # TODO: Do this operation better!
    # Index the log likelihood of gold edges.
    edgenode_log_likelihood = permutedims(edgenode_log_likelihood, [3,2,1])           
    gold_edge_node_nll = 0.0
    for b in 1:B
        for i in 2:T       # -> (T-1, B): Exclude the dummy root.
            gold = edge_heads[i,b]      
            if gold==0 continue; end                                  
            gold_edge_node_nll += edgenode_log_likelihood[b,i,gold]
        end
    end
    gold_edge_label_nll = 0.0
    for b in 1:B
        for i in 2:T       # -> (T-1, B): Exclude the dummy root.
            gold = edgelabels'[i,b]
            if gold==0 continue; end                                     
            gold_edge_label_nll += edgelabel_log_likelihood[b,i,gold]
        end
    end
    loss = (-gold_edge_node_nll) + (-gold_edge_label_nll)
    return loss
end