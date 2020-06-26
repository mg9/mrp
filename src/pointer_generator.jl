include("s2s.jl")
include("linear.jl")

_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})


struct PointerGenerator
    projection::Linear          # decoderhiddens(B*Ty, Hy) -> vocabprobs(B,Ty, vocabsize)
    linear_pointer::Linear      # decoderhiddens(B*Ty, Hy) -> pointerprobs(B,Ty, 3): p_copy_source,p_copy_target, p_generate
end


# ## Model constructor
#
# The `PointerGenerator` constructor takes the following arguments:
# * `hidden`: size of the hidden vectors of the decoder
# * `vocab_size`: number of decoder tokens vocabsize
# * `eps`:
# * `force_copy`:
function PointerGenerator(hidden::Int, vocabsize::Int, eps::Float64;  force_copy=true)
    projection = Linear(hidden, vocabsize)
    switch_input_size = hidden
    linear_pointer = Linear(switch_input_size, 3)
    force_copy = true
    eps = 1e-20
    PointerGenerator(projection, linear_pointer) 
end

#-
@testset "Testing PointerGenerator" begin
    H = 512
    pg = PointerGenerator(H, vocabsize, 1e-20)
    @test size(pg.projection.w) == (vocabsize, H)
end


##  Compute a distribution over the target dictionary
#   extended by the dynamic dictionary implied by copying target nodes.
#
#   `hiddens`: decoder outputs
#   `src_attentions`: attention of each source node
#   `src_attentions_maps`: a sparse indicator matrix
#       mapping each source node to its index in the dynamic vocabulary.
#   `tgt_attentions`: attention of each target node
#   `tgt_attention_maps`: a sparse indicator matrix
#        mapping each target node to its index in the dynamic vocabulary.
#   `invalid_indexes`:   indexes which are not considered in prediction.
#   `generate_targets`:  target node index in the vocabulary,
#   `src_copy_targets`:  target node index in the dynamic vocabulary,
#   `tgt_copy_targets`:  target node index in the dynamic vocabulary,
#   `coverage_records`:  Nothing or a tensor recording source-side coverages.
## TODO: Invalid indexes part
function (pg::PointerGenerator)(attn_vector, src_attentions, src_attention_maps, tgt_attentions, 
                                tgt_attention_maps, generate_targets, src_copy_targets, tgt_copy_targets; invalid_indexes=Nothing, vocab_pad_idx = 1)
   
    # attn_vector:    (Hy, B, 1)
    # src_attentions: (Tx,1,B) 
    # src_attentions_maps: (src_dynamic_vocabsize, Tx, B)
    # tgt_attentions: (Ty,1,B) 
    # tgt_attention_maps: (tgt_dynamic_vocabsize, Ty, B)
    # -> probs: (1, vocabsize + src_dynamic_vocabsize + tgt_dynamic_vocabsize, B)
    # -> predictions: (1, 1, B)
    # -> src_dynamic_vocabsize, tgt_dynamic_vocabsize


    ## TODO fix here
    #dummy = convert(_atype, zeros(size(tgt_attention_maps,2),size(tgt_attentions,2),size(tgt_attentions,3)))
    #tgt_attentions = tgt_attentions .+ dummy

    vocab_pad_idx = 1; 
    unknown_idx = 2
    coverage_records=nothing; eps = 1e-20; force_copy=true; vocabsize = 12202; # TODO: take theese dynamically


    Hy, B, Ty = size(attn_vector) 
    attn_vector = reshape(attn_vector, (:,B *Ty)) # -> (H, B*Ty) 

    src_dynamic_vocabsize = size(src_attention_maps, 1)
    tgt_dynamic_vocabsize = size(tgt_attention_maps, 1)


    # Pointer probability.
    p = softmax(pg.linear_pointer(attn_vector), dims=1) # -> (3, B*Ty)
    p_copy_source = reshape(p[1, :], (B*Ty, 1))         # -> (B*Ty,1)
    p_copy_source = reshape(p_copy_source, (Ty,1,B))    # -> (Ty,1,B)

    p_copy_target = reshape(p[2, :], (B*Ty, 1))         # -> (B*Ty,1)
    p_copy_target = reshape(p_copy_target, (Ty,1,B))    # -> (Ty,1,B)

    p_generate = reshape(p[3, :], (B*Ty, 1))            # -> (B*Ty,1)
    p_generate = reshape(p_generate, (Ty,1,B))          # -> (Ty,1,B)


    # Pgen: Probability distribution over the vocabulary.
    scores = pg.projection(attn_vector)                              # -> (vocabsize, B*1)
    scores = permutedims(scores, [2,1])                              # -> (B*Ty, vocabsize)
    #scores[:, vocab_pad_idx] .+= -Inf32                             # TODO: fix here for diff
    vocab_probs = softmax(scores, dims=2)                            # TODO: is it okay?
    vocab_probs = reshape(vocab_probs, (Ty, vocabsize, B))           # -> (1, vocabsize, B)
    dummy = convert(_atype, zeros(1,size(vocab_probs,2),1))
    p_generate = p_generate .+ dummy                                 # -> (1, vocabsize, B)
    scaled_vocab_probs = vocab_probs .* p_generate                   # -> (1, vocabsize, B)

    
    # Probability distribution over the dynamic vocabulary.
    # TODO: make sure for target_node_i, its attention to target_node_j >= target_node_i
    # should be zero.

    # Psrc
    src_attentions = permutedims(src_attentions, [2,1,3])                                           # -> (1,Tx,B)
    dummy = convert(_atype, zeros(1,size(src_attentions,2),1))
    p_copy_source = p_copy_source .+ dummy                                                          # -> (1,Tx,B)
    scaled_src_attentions = src_attentions .* p_copy_source                                         # -> (1,Tx,B)
    src_attention_maps = permutedims(src_attention_maps, [2,1,3])                                   # -> (Tx, srcdynamic_vocabsize, B)
    scaled_copy_source_probs = bmm(scaled_src_attentions, convert(_atype,src_attention_maps))       # -> (1, srcdynamic_vocabsize, B)

    
    # Ptgt
    tgt_attentions = permutedims(tgt_attentions, [2,1,3])                                            # -> (1,Ty,B) - a dummy line
    dummy = convert(_atype, zeros(1,size(tgt_attentions,2),1))
    p_copy_target = p_copy_target .+ dummy                                                           # -> (1,Ty,B)
    scaled_tgt_attentions = tgt_attentions .* p_copy_target                                          # -> (1,Ty,B)
    tgt_attention_maps = permutedims(tgt_attention_maps, [2,1,3])                                    # -> (Ty, tgtdynamic_vocabsize, B)
    scaled_copy_target_probs = bmm(scaled_tgt_attentions, convert(_atype,tgt_attention_maps))        # -> (1, tgtdynamic_vocabsize, B)

    probs = cat(scaled_vocab_probs, scaled_copy_source_probs, scaled_copy_target_probs, dims=2)      # -> (Ty,vocabsize+dynamic_vocabsize,B)
    predictions = argmax(value(probs), dims=2)                                                       # -> (1,1,B)

    # TODO: check this part from original code again
    #   Set the probability of coref NA to 0.
    #  _probs = copy(probs)
    #  _probs[vocab_size + source_dynamic_vocab_size, :, :] = 0
    #_, predictions = _probs.max(2)

    predictions = reshape(predictions, (:, size(predictions,2)*size(predictions,3)))                 # -> (1, B)
    coverage_records = nothing                                                                       # TODO: fix here and use coverage records
    
    loss = calcloss(probs, predictions, src_attentions, generate_targets, src_copy_targets, 
        tgt_copy_targets, src_dynamic_vocabsize, tgt_dynamic_vocabsize, coverage_records)
    return loss
end

    

##  Priority: tgt_copy > src_copy > generate
#
#  `probs`:              probability distribution,
#  `generate_targets`:   target node index in the vocabulary,
#  `src_copy_targets`:   target node index in the dynamic vocabulary,
#  `tgt_copy_targets`:   target node index in the dynamic vocabulary,
#  `coverage_records`:   Nothing or a tensor recording source-side coverages.
#
function calcloss(probs, predictions, copy_attentions, generate_targets, src_copy_targets, tgt_copy_targets, src_dynamic_vocabsize, tgt_dynamic_vocabsize, coverage_records)
    # probs: (1, vocabsize + src_dynamic_vocabsize + tgt_dynamic_vocabsize, B)
    # predictions: (Ty,B)
    # copy_attentions(src_attentions): (Tx,Ty,B) 
    # generate_targets:  (Ty,B)
    # src_copy_targets:  (Ty,B)
    # tgt_copy_targets:  (Ty,B)
    # src_dynamic_vocabsize: 9, remove here
    # tgt_dynamic_vocabsize: 3, remove here

    vocab_pad_idx = 2; unknown_idx = 1; coverage_records=nothing; eps = 1e-20; force_copy=true; vocabsize = 12202; # TODO: take theese dynamically

    non_pad_mask  = (generate_targets .!= vocab_pad_idx)                                                # -> (Ty,B)
    src_copy_mask = (src_copy_targets .!= unknown_idx) .* (src_copy_targets .!= vocab_pad_idx)          # -> (Ty,B): 1 is the index for unknown words 
    non_src_copy_mask = 1 .- src_copy_mask

    tgt_copy_mask = (tgt_copy_targets .!= 1)                                                # -> (Ty,B): 0 (1 for us) is the index for coref NA
    non_tgt_copy_mask = 1 .- tgt_copy_mask                                                 
    offset = vocabsize + src_dynamic_vocabsize
    tgt_copy_targets_with_offset = tgt_copy_targets .+ offset                               # -> (Ty,B)
    tgt_copy_target_probs = probs[tgt_copy_targets_with_offset]                             # -> (Ty,B): TODO: is that true? 
    tgt_copy_target_probs = tgt_copy_target_probs .* convert(_atype,tgt_copy_mask)          # -> (Ty, B)
    

    offset = vocabsize
    src_copy_targets_with_offset = src_copy_targets .+ offset
    src_copy_target_probs = probs[src_copy_targets_with_offset]                             # -> (Ty,B): TODO: is that true? 
    src_copy_target_probs = src_copy_target_probs .* convert(_atype,non_tgt_copy_mask)      # -> (Ty,B)
    src_copy_target_probs = src_copy_target_probs .* convert(_atype,src_copy_mask)          # -> (Ty,B)

    generate_target_probs = probs[generate_targets]                                         # -> (Ty,B): TODO: is that true? 

    # Except copy-oov nodes, all other nodes should be copied.
    likelihood = tgt_copy_target_probs + src_copy_target_probs +                            # -> (Ty,B): 
                (generate_target_probs.* convert(_atype,non_tgt_copy_mask) .* convert(_atype,non_src_copy_mask))

    num_tokens = sum(non_pad_mask)

    if !(force_copy)
        non_generate_oov_mask = (generate_targets .!= 1) 
        additional_generate_mask = (non_tgt_copy_mask .* src_copy_mask .* non_generate_oov_mask)
        likelihood = likelihood + generate_target_probs .* convert(_atype,additional_generate_mask)
        num_tokens += sum(additional_generate_mask)
    end

    likelihood = likelihood .+ eps  # Add eps for numerical stability.
    coverage_loss = 0
    # TODO: implement the part if coverage_records!==nothing 
    loss = -log.(likelihood) .* convert(_atype, non_pad_mask) .+ coverage_loss # Drop pads
    loss = sum(loss) / length(loss)
    return loss
end