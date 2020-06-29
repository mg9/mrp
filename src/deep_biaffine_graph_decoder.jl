include("linear.jl")
include("biaffine_attention.jl")
include("bilinear.jl")
include("s2s.jl")


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

function (g::DeepBiaffineGraphDecoder)(hiddens, gold_edgeheads, gold_edgelabels, parser_mask, corefs)
    # hiddens:      (Hy, B, Ty) 
    # edge_heads:   (1(Ty-1?), B)
    # edgelabels:   (1(Ty-1?), B)
    # corefs:       (Ty, B)
    # parser_mask:  (Ty, B)
    # -> loss


    ## Part1: Add a dummy ROOT at the beginning of each node sequence.
    Hy, B, Ty = size(hiddens)
    hiddens = hiddens[:,:,2:end]      # Exclude BOS symbol
    mask = parser_mask[2:end, :]
    head_sentinel = param(Hy, 1, 1)                       # -> (Hy, 1, 1) does this meet randn requirement?
    dummy = convert(_atype, zeros(1, B, 1))
    head_sentinel = head_sentinel .+ dummy                # -> (Hy, B, 1)
    hiddens = cat(head_sentinel, hiddens, dims=3)         # -> (Hy, B, Ty+1)
    if !isnothing(gold_edgeheads); gold_edgeheads = vcat(zeros(1,B), gold_edgeheads); end
    if !isnothing(gold_edgelabels); gold_edgelabels = vcat(zeros(1,B), gold_edgelabels); end
    mask = vcat(ones(1,B), mask)

    ## Part2: Encode 
    hiddens   = reshape(hiddens, (Hy,:))                  # -> (Hy, B*Ty)
    edgenode_h = elu.(g.edgenode_h_linear(hiddens))       # -> (edgenodehiddensize, B*Ty)
    edgenode_h = reshape(edgenode_h, (:, B, Ty))          # -> (edgenodehiddensize, B,Ty)
    edgenode_m = elu.(g.edgenode_m_linear(hiddens))       # -> (edgenodehiddensize, B*Ty)
    edgenode_m = reshape(edgenode_m, (:, B, Ty))          # -> (edgenodehiddensize, B,Ty)
    edgelabel_h = elu.(g.edgelabel_h_linear(hiddens))     # -> (edgelabelhiddensize, B*Ty)
    edgelabel_h = reshape(edgelabel_h, (:, B, Ty))        # -> (edgelabelhiddensize, B,Ty)
    edgelabel_m = elu.(g.edgelabel_m_linear(hiddens))     # -> (edgelabelhiddensize, B*Ty)
    edgelabel_m = reshape(edgelabel_m, (:, B, Ty))        # -> (edgelabelhiddensize, B,Ty)
   

    ## Part3: Calculate edge node scores 
    edgenode_scores = g.biaffine_attention(edgenode_h, edgenode_m, mask_d=mask, mask_e=mask)             # -> (B, Tencoder, Tdecoder, num_labels)
    edgenode_scores = reshape(edgenode_scores, (size(edgenode_scores,1), size(edgenode_scores,2), :))    # -> (B, Tencoder, Tdecoder*num_labels=1)
   
    ## Set invalid positions to -inf
    diagonal(A::AbstractMatrix, k::Integer=0) = view(A, diagind(A, k)) # Set diagonal elements to -inf, to prevent edge between node itself (no loop)
    a = zeros(Ty,Ty); diagonal(a) .= -Inf
    a = reshape(a, (1,Ty,Ty))
    edgenode_scores = edgenode_scores .+ convert(_atype,a)                              # -> (B, T, T)
    minus_mask = 1 .- mask
    settoinf(x) = x==1.0 ? x= -Inf : x=x                                                # -> (T, B) 
    minus_mask = settoinf.(minus_mask)
    minus_mask = reshape(minus_mask, (1,Ty,B))                                          # -> (1, T, B)
    edgenode_scores = permutedims(edgenode_scores, [3,2,1])                             # -> (T, T, B)
    edgenode_scores =edgenode_scores .+convert(_atype,minus_mask)                       # -> (T, T, B) 
 

    ## Part4: Pred edge heads 
    ina(x) = return x[2]
    a = argmax(value(edgenode_scores), dims=2)  # Compute naive predictions of edge_heads
    b = reshape(a, size(a,1),size(a,3))         # -> (T,B)
    pred_edgeheads = ina.(b)                    # -> (T,B): remove cartesian type    
    #@info("gold_edgeheads: ", gold_edgeheads) 
    #@info("pred_edgeheads: ", pred_edgeheads)

    ## Part5: Calculate edge label scores
    # TODO: Is it correct?
    # Select the heads' representations based on the gold/predicted heads.
    edgelabel_h = permutedims(edgelabel_h, [2,3,1])                                     # -> (B, Ty, edgelabelhiddensize)
    edgelabel_h = reshape(edgelabel_h, size(edgelabel_h,1)* size(edgelabel_h,2), :)     # -> (B*Ty, edgelabelhiddensize)
    pred_edgeheads = reshape(pred_edgeheads, (B*Ty))                                    # -> (B*Ty)
    
    edgelabel_h = edgelabel_h[pred_edgeheads,:]                                         # -> (B*Ty, edgelabelhiddensize)
    edgelabel_h = reshape(edgelabel_h, (B,Ty,:))                                        # -> (B,Ty, edgelabelhiddensize)
    edgelabel_h = permutedims(edgelabel_h, [3,1,2])                                     # -> (edgelabelhiddensize, B, Ty)
    edgelabel_scores = g.edgelabel_bilinear(edgelabel_h, edgelabel_m)                   # -> (B, Ty, numlabels(headtags))   


    ## Part6: Pred edge labels
    a = argmax(value(edgelabel_scores), dims=3)     # -> (B,Ty,1)
    b = reshape(a, size(a,1),size(a,2)*size(a,3))   # -> (B,Ty)
    pred_edgelabels = ina.(b)                       # -> (B,Ty): remove cartesian type 
    #@info("gold_edgelabels: ", gold_edgelabels) 
    #@info("pred_edgelabels: ", pred_edgelabels')


    ## Part7: Calculate loss
    a = softmax(edgenode_scores, dims=2) .+ 0.00000001  # avoid  -Inf for log
    b = softmax(edgelabel_scores, dims=3).+ 0.00000001  # avoid  -Inf for log

    edgenode_log_likelihood  = log.(a)                  # -> (Ty, Ty, B)
    edgelabel_log_likelihood = log.(b)                  # -> (B,Ty,num_labels)

    # TODO: Do these operations better!
    # Index the log likelihood of gold edges.
    gold_edgenode_nll = 0.0
    for b in 1:B
        for i in 2:Ty                                                                   # -> (T-1, B): Exclude the dummy root.
            gold = gold_edgeheads[i,b] 
            gold_edgenode_nll = edgenode_log_likelihood[i,gold,b]
        end
    end
    gold_edgelabel_nll = 0.0
    for b in 1:B
        for i in 2:Ty                                                                   # -> (T-1, B): Exclude the dummy root.
            gold = gold_edgelabels[i,b]
            if gold ==0 ; continue; end   # Ignore pads
            gold_edgelabel_nll += edgelabel_log_likelihood[b,i,gold]
        end
    end
    loss = (-gold_edgenode_nll) + (-gold_edgelabel_nll)
    return loss
end
