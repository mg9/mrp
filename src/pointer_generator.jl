include("linear.jl")
include("metrics.jl")
include("debug.jl")

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

function (pg::PointerGenerator)(attnvector, srcattentions, tgtattentions, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, decodervocab_pad_idx, coverage=nothing; force_copy=true)

    Hy = size(attnvector,1)
    Tx,Ty,B = size(srcattentions)
    numtarget,Ty,B = size(tgtattentions)

    ; @size attnvector (Hy,Ty*B)
    ; @size srcattentionmaps (Tx,Tx+2,B); @size tgtattentionmaps (numtarget,numtarget+1,B);
    ; @size generatetargets (Ty,B); @size srccopytargets (Ty,B); @size tgtcopytargets (Ty,B)

    # Pointer probability. (pgen + psrc +ptgt =1)
    p = softmax(pg.linearpointer(attnvector), dims=1)                           ; @size p (3,Ty*B);  # switch between pgen-psrc-ptgt
    p_copysrc = p[1:1,:]                                                        ; @size p_copysrc (1,Ty*B);
    p_copysrc = reshape(p_copysrc, (1,Ty,B))                                    ; @size p_copysrc (1,Ty,B);
    p_copytgt = p[2:2,:]                                                        ; @size p_copytgt (1,Ty*B);
    p_copytgt = reshape(p_copytgt, (1,Ty,B))                                    ; @size p_copytgt (1,Ty,B);
    p_gen = p[3:3,:]                                                            ; @size p_gen (1,Ty*B);
    p_gen = reshape(p_gen, (1,Ty, B))                                           ; @size p_gen (1,Ty,B);


    # Pgen: Probability distribution over the vocabulary.
    scores = pg.projection(attnvector)                                  ; @size scores (pg.vocabsize,Ty*B);
    # Make score of decoder pad -Inf
    infvec = zeros(size(scores,1))
    infvec[decodervocab_pad_idx]=Inf
    scores = scores .- convert(_atype,infvec) 

    vocab_probs = softmax(scores, dims=1)                               ; @size vocab_probs (pg.vocabsize,Ty*B)
    vocab_probs = reshape(vocab_probs, (pg.vocabsize, Ty,B))            ; @size vocab_probs (pg.vocabsize,Ty,B);
    scaled_vocab_probs = vocab_probs .* p_gen                           ; @size scaled_vocab_probs (pg.vocabsize,Ty,B);
    scaled_vocab_probs = permutedims(scaled_vocab_probs, [2,1,3])       ; @size scaled_vocab_probs (Ty,pg.vocabsize,B);

    # Psrc: 
    scaled_srcattentions = srcattentions .* p_copysrc                ; @size scaled_srcattentions (Tx,Ty,B); 
    scaled_copysrc_probs = bmm(permutedims(scaled_srcattentions, [2,1,3]), _atype(srcattentionmaps)) ; @size scaled_copysrc_probs (Ty,Tx+2,B);  # Tx+2 is the dynamic srcvocabsize

    # Ptgt: 
    scaled_tgtattentions = tgtattentions .* p_copytgt           ; @size scaled_tgtattentions (numtarget,Ty, B);  
    scaled_copytgt_probs = bmm(permutedims(scaled_tgtattentions, [2,1,3]), _atype(tgtattentionmaps))    ; @size scaled_copytgt_probs (Ty,numtarget+1,B);    # Ty is the dynamic tgtvocabsize


    #TODO: Do invalid indexes part.
    probs = cat(scaled_vocab_probs, scaled_copysrc_probs, scaled_copytgt_probs, dims=2)  ; @size probs (Ty,pg.vocabsize+Tx+2+numtarget+1,B)
    # Set probability of coref NA to 0
    NAvec = ones(1, size(probs, 2), 1)
    NAvec[pg.vocabsize+Tx+3] = 0
    # After broadcasting the probability of coref NA is set to 0
    probs = probs .* convert(_atype,NAvec)


    getind(cartindx) = return cartindx[2]                                                            
    predictions = getind.(argmax(softmax(value(probs),dims=2), dims=2))  ;@size predictions (Ty, 1, B)
    predictions = convert(_atype, reshape(predictions, (Ty,B)))


    ### PointerGenerator Loss
    ; @size generatetargets (Ty, B);  ; @size srccopytargets  (Ty, B);  ; @size tgtcopytargets  (Ty, B)

    # Masks
    pad_idx = 1 #trn.tgtvocab.token_to_idx["@@PADDING@@"]
    unk_idx = 2 #trn.tgtvocab.token_to_idx["@@UNKNOWN@@"]
    non_pad_mask  = convert(_atype, generatetargets .!= pad_idx)                                        ;@size non_pad_mask (Ty,B)                                               
    srccopy_mask  = convert(_atype, (srccopytargets .!= unk_idx) .* (srccopytargets .!= pad_idx))       ;@size srccopy_mask (Ty,B) # 2 is UNK id, 1 is pad id
    non_srccopy_mask = 1 .- srccopy_mask
    tgtcopy_mask = convert(_atype, (tgtcopytargets .!= 0))                                              ;@size tgtcopy_mask (Ty,B) # 0 is the index for coref NA
    non_tgtcopy_mask = 1 .- tgtcopy_mask


    if !isnothing(coverage)
        # Calculate the coverage_loss # TODO: Is target mask needed here as well? Not used in original 
        srcattentions = reshape(srcattentions, (Tx, Ty, B))
        coverage_loss = sum(min.(srcattentions, coverage) .* reshape(non_pad_mask, (Ty, Ty, B)))
        # Add the current alignments to the coverage vector
        coverage = coverage + srcattentions                                         ; @size coverage (Tx, Ty, B)
        # maintain a list of coverages for each timestamp if needed
        # push!(coverage_records, coverage)
    end



    src_dynamic_vocabsize = Tx+2
    tgt_dynamic_vocabsize = numtarget+1
    total_vocabsize = pg.vocabsize + src_dynamic_vocabsize + tgt_dynamic_vocabsize

    probs = permutedims(probs, [2,1,3]) ;@size probs (total_vocabsize, Ty,B)


    tgtcopytargets_withoffset = tgtcopytargets .+ (pg.vocabsize+src_dynamic_vocabsize+1)                ;@size tgtcopytargets_withoffset (Ty,B) 
    tgtcopy_targetprobs = reshape(probs[findindices(probs, tgtcopytargets_withoffset)], Ty,B)           ;@size tgtcopy_targetprobs (Ty,B)
    tgtcopy_targetprobs = tgtcopy_targetprobs .* tgtcopy_mask                                           ;@size tgtcopy_targetprobs (Ty,B)


    srccopytargets_withoffset = srccopytargets .+ (pg.vocabsize +1)                                     ;@size srccopytargets_withoffset (Ty,B) 
    srccopy_targetprobs = reshape(probs[findindices(probs,  srccopytargets_withoffset)], Ty,B)          ;@size srccopy_targetprobs (Ty,B)
    srccopy_targetprobs = srccopy_targetprobs .* srccopy_mask .* non_tgtcopy_mask                       ;@size srccopy_targetprobs (Ty,B)    


    gentargets_withoffset = generatetargets .+ 1                                                        ;@size srccopytargets_withoffset (Ty,B) 
    generate_targetprobs = reshape(probs[findindices(probs, gentargets_withoffset)], Ty,B)              ;@size srccopy_targetprobs (Ty,B)
    generate_targetprobs = generate_targetprobs .* non_srccopy_mask .* non_tgtcopy_mask                 ;@size generate_targetprobs (Ty,B) 


   # Except copy-oov nodes, all other nodes should be copied.
    likelihood = generate_targetprobs + tgtcopy_targetprobs + srccopy_targetprobs                       ;@size likelihood (Ty,B)
    num_tokens = sum(non_pad_mask .== 1)

    if !(force_copy)
        non_generate_oov_mask = (generatetargets .!= unk_idx)  
        additional_generate_mask = non_tgtcopy_mask .* srccopy_mask .* non_generate_oov_mask #?
        likelihood = likelihood + generate_targetprobs .* additional_generate_mask
        num_tokens += sum(additional_generate_mask .== unk_idx)
    end

    # Add eps for numerical stability.
    likelihood = likelihood .+ 1e-20 #eps
    loss = sum(-log.(likelihood) .* non_pad_mask) #+ coverage_loss     # Drop pads.

    # Mask out copy targets for which copy does not happen.
    targets = _atype(tgtcopytargets_withoffset) .* tgtcopy_mask + 
              _atype(srccopytargets_withoffset) .* srccopy_mask .* non_tgtcopy_mask +
              _atype(gentargets_withoffset) .* non_srccopy_mask .* non_tgtcopy_mask                    ;@size targets (Ty,B)

    pg.metrics(predictions, targets , non_pad_mask, non_tgtcopy_mask, srccopy_mask, tgtcopy_mask, pg.vocabsize, src_dynamic_vocabsize, value(loss))
    #println("loss: $loss")
    return loss
end


function findindices(y,a::AbstractArray{<:Integer}; dims=1)
    ninstances = length(a)
    nindices = 0
    indices = Vector{Int}(undef,ninstances)
    if dims == 1                   # instances in first dimension
        y1 = size(y,1)
        y2 = div(length(y),y1)
        if ninstances != y2; throw(DimensionMismatch()); end
        @inbounds for j=1:ninstances
            if a[j] == 0; continue; end
            indices[nindices+=1] = (j-1)*y1 + a[j]
        end
    elseif dims == 2               # instances in last dimension
        y2 = size(y,ndims(y))
        y1 = div(length(y),y2)
        if ninstances != y1; throw(DimensionMismatch()); end
        @inbounds for j=1:ninstances
            if a[j] == 0; continue; end
            indices[nindices+=1] = (a[j]-1)*y1 + j
        end
    else
        error("findindices only supports dims = 1 or 2")
    end
    return (nindices == ninstances ? indices : view(indices,1:nindices))
end
