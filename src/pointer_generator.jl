include("linear.jl")
include("metrics.jl")

struct PointerGenerator
    projection::Linear          # decoderhiddens(B*Ty, Hy) -> vocabprobs(B,Ty, vocabsize)
    linearpointer::Linear       # decoderhiddens(B*Ty, Hy) -> pointerprobs(B,Ty, 3): p_copy_source,p_copy_target, p_generate
    metrics::PointerGeneratorMetrics
    vocabsize
end

# The `PointerGenerator` constructor takes the following arguments:
# * `hidden`: size of the hidden vectors of the decoder
# * `vocab_size`: number of decoder tokens vocabsize
function PointerGenerator(hidden::Int, vocabsize::Int)
    projection = Linear(hidden, vocabsize)
    linear_pointer = Linear(hidden, 3)
    metrics = PointerGeneratorMetrics(0,0,0,0,0,0,0,0,0)
    PointerGenerator(projection, linear_pointer, metrics, vocabsize) 
end

#=
@testset "Testing PointerGenerator" begin
    H = 512
    pg = PointerGenerator(H, vocabsize, 1e-20)
    @test size(pg.projection.w) == (vocabsize, H)
end
=#


function (pg::PointerGenerator)(attnvector, srcattentions, tgtattentions, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, decodervocab_pad_idx, coverage ; force_copy=true )

    Hy,B = size(attnvector)
    Tx = size(srcattentions,1)
    Ty = size(tgtattentions,1)

    # Pointer probability. (pgen + psrc +ptgt =1)
    p = softmax(pg.linearpointer(attnvector), dims=1)                            ; @size p (3,B);  # switch between pgen-psrc-ptgt
    p_copysrc = p[1:1,:]                                                        ; @size p_copysrc (1,B);
    p_copysrc = reshape(p_copysrc, (1,B,1))                                     ; @size p_copysrc (1,B,1);
    p_copytgt = p[2:2,:]                                                        ; @size p_copytgt (1,B);
    p_copytgt = reshape(p_copytgt, (1,B,1))                                     ; @size p_copytgt (1,B,1);
    p_gen = p[3:3,:]                                                            ; @size p_gen (1,B);
    p_gen = reshape(p_gen, (1,B,1))                                             ; @size p_gen (1,B,1);


    # Pgen: Probability distribution over the vocabulary.
    scores = pg.projection(attnvector)                             ; @size scores (pg.vocabsize,B);
    # Make score of decoder pad -Inf
    infvec = zeros(size(scores,1))
    infvec[decodervocab_pad_idx]=Inf
    scores = scores .- convert(_atype,infvec) 
    vocab_probs = softmax(scores, dims=1)                           ; @size vocab_probs (pg.vocabsize,B)
    vocab_probs = reshape(vocab_probs, (pg.vocabsize, B, 1))           ; @size vocab_probs (pg.vocabsize,B,1);
    scaled_vocab_probs = vocab_probs .* p_gen                       ; @size scaled_vocab_probs (pg.vocabsize,B,1);
    scaled_vocab_probs = permutedims(scaled_vocab_probs, [3,1,2])   ; @size scaled_vocab_probs (1,pg.vocabsize,B);

    # Psrc: 
    srcattentions = reshape(srcattentions, (Tx,B,1))                ; @size srcattentions (Tx,B,1);
    scaled_srcattentions = srcattentions .* p_copysrc               ; @size scaled_srcattentions (Tx,B,1); @size srcattentionmaps (Tx,Tx+2,B); 
    scaled_copysrc_probs = bmm(permutedims(scaled_srcattentions, [3,1,2]), _atype(srcattentionmaps)) ; @size scaled_copysrc_probs (1,Tx+2,B);  # Tx+2 is the dynamic srcvocabsize

    # Ptgt: 
    #TODO: do tgtattentions                                     
    tgtattentions = permutedims(tgtattentions, [1,3,2])         ; @size  tgtattentions (Ty,B,1);
    scaled_tgtattentions = tgtattentions .* p_copytgt           ; @size scaled_tgtattentions (Ty,B,1);  ; @size tgtattentionmaps (Ty,Ty+1,B);
    scaled_copytgt_probs = bmm(permutedims(scaled_tgtattentions, [3,1,2]), _atype(tgtattentionmaps))    ; @size scaled_copytgt_probs (1,Ty+1,B);    # Ty is the dynamic tgtvocabsize

              
    #TODO: Do invalid indexes part.
    probs = cat(scaled_vocab_probs, scaled_copysrc_probs, scaled_copytgt_probs, dims=2)  ; @size probs (1,pg.vocabsize+Tx+2+Ty+1,B)
    # Set probability of coref NA to 0
    NAvec = ones(1, size(probs, 2), 1)
    NAvec[pg.vocabsize+Tx+3] = 0
    # After broadcasting the probability of coref NA is set to 0
    probs = probs .* convert(_atype,NAvec)
    getind(cartindx) = return cartindx[2]                                                            
    predictions = getind.(argmax(softmax(value(probs),dims=2), dims=2))  ;@size predictions (1, 1, B)
    predictions = convert(_atype, reshape(predictions, (1,B)))


    ### PointerGenerator Loss
    ; @size generatetargets (1, B);  ; @size srccopytargets  (1, B);  ; @size tgtcopytargets  (1, B)

    # Masks
    pad_idx = 1 #trn.tgtvocab.token_to_idx["@@PADDING@@"]
    unk_idx = 2 #trn.tgtvocab.token_to_idx["@@UNKNOWN@@"]
    non_pad_mask  = convert(_atype, generatetargets .!= pad_idx)                                        ;@size non_pad_mask (1,B)                                               
    srccopy_mask  = convert(_atype, (srccopytargets .!= unk_idx) .* (srccopytargets .!= pad_idx))      ;@size srccopy_mask (1,B) # 2 is UNK id, 1 is pad id
    non_srccopy_mask = 1 .- srccopy_mask
    tgtcopy_mask = convert(_atype, (tgtcopytargets .!= 0))                                              ;@size tgtcopy_mask (1,B) # 0 is the index for coref NA
    non_tgtcopy_mask = 1 .- tgtcopy_mask
    


    # Calculate the coverage_loss # TODO: Is target mask needed here as well? Not used in original 
    srcattentions = reshape(srcattentions, (Tx, 1, B))
    coverage_loss = sum(min.(srcattentions, coverage) .* reshape(non_pad_mask, (1, 1, B)))
    # Add the current alignments to the coverage vector
    coverage = coverage + srcattentions                                         ; @size coverage (Tx, 1, B)
    # maintain a list of coverages for each timestamp if needed
    # push!(coverage_records, coverage)


    # returns the prob values with given indices
    function getprobs(ind, probarr)
        results = []
        for b in 1:size(probarr,3)
            push!(results,(probarr[:,ind[b],b]))
        end
        return hcat(reshape(results, (1,length(results)))...)
    end

    src_dynamic_vocabsize = Tx+2

    tgtcopytargets_withoffset = tgtcopytargets .+ (pg.vocabsize+src_dynamic_vocabsize+1)                         ;@size tgtcopytargets_withoffset (1,B) ## 
    tgtcopy_targetprobs = getprobs(tgtcopytargets_withoffset, probs)                        ;@size tgtcopy_targetprobs (1,B)
    tgtcopy_targetprobs = tgtcopy_targetprobs .* tgtcopy_mask                               ;@size tgtcopy_targetprobs (1,B)

    srccopytargets_withoffset = srccopytargets .+ (pg.vocabsize +1)                           ;@size srccopytargets_withoffset (1,B) 
    srccopy_targetprobs = getprobs(srccopytargets_withoffset, probs)                        ;@size srccopy_targetprobs (1,B)
    srccopy_targetprobs = srccopy_targetprobs .* srccopy_mask .* non_tgtcopy_mask           ;@size srccopy_targetprobs (1,B)    

    generate_targetprobs = getprobs(generatetargets, probs)                                ;@size generate_targetprobs (1,B)
    generate_targetprobs = generate_targetprobs .* non_srccopy_mask .* non_tgtcopy_mask     ;@size generate_targetprobs (1,B) 

    # Except copy-oov nodes, all other nodes should be copied.
    likelihood = generate_targetprobs + tgtcopy_targetprobs + srccopy_targetprobs           ;@size likelihood (1,B)
    num_tokens = sum(non_pad_mask .== 1)

    if !(force_copy)
        non_generate_oov_mask = (generatetargets .!= unk_idx)  
        additional_generate_mask = non_tgtcopy_mask .* srccopy_mask .* non_generate_oov_mask #?
        likelihood = likelihood + generate_targetprobs .* additional_generate_mask
        num_tokens += sum(additional_generate_mask .== unk_idx)
    end

    # Add eps for numerical stability.
    likelihood = likelihood .+ 1e-20 #eps

    # TODO: Coverage records
    loss = sum(-log.(likelihood) .* non_pad_mask) + coverage_loss     # Drop pads.


    # Mask out copy targets for which copy does not happen.
    targets = _atype(tgtcopytargets_withoffset) .* tgtcopy_mask + 
              _atype(srccopytargets_withoffset) .* srccopy_mask .* non_tgtcopy_mask +
              _atype(generatetargets) .* non_srccopy_mask .* non_tgtcopy_mask                    ;@size targets (1,B)

    pg.metrics(predictions, targets, non_pad_mask, non_tgtcopy_mask, srccopy_mask, tgtcopy_mask, pg.vocabsize, src_dynamic_vocabsize, loss)
    return loss
end