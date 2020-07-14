include("s2s.jl")
include("pointer_generator.jl")
include("deep_biaffine_graph_decoder.jl")
include("debug.jl")
include("metrics.jl")

struct BaseModel
    s::S2S             
    p::PointerGenerator
    g::DeepBiaffineGraphDecoder
    metrics::AMRMetrics
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
    p = PointerGenerator(H, vocabsize)
    g = DeepBiaffineGraphDecoder(H, edgenode_hidden_size, edgelabel_hidden_size, num_labels)
    metrics = AMRMetrics()

    BaseModel(s, p, g, metrics)
end



function (m::BaseModel)(srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, parsermask, edgeheads, edgelabels,encodervocab_pad_idx, decodervocab_pad_idx ; force_copy=true)

    B, Tx = size(srctokens); Ty = size(tgttokens,2); Hx=Hy = H; Dx =2
    @size srcattentionmaps (Tx,Tx+2,B); @size tgtattentionmaps (Ty,Ty+1,B); 
    mem = encode(m.s, srctokens) ; if mem != (); @size mem[1] (Hy,Tx,B); @size mem[2] (Hx*Dx,Tx,B); end
    #tgt2 = mask!(tgttokens[:,2:end], s.tgteos)    ; @size tgt2 (B,Ty-1)
    tgt2 = tgttokens[:,2:end]  ; @size tgt2 (B,Ty-1)
    sumloss, numwords = 0, 0
    cell = fill!(similar(mem[1], size(mem[1],1), size(mem[1],3), 1), 0)
    hiddens = []
    # Coverage vector for the source vocabulary, accumulated
    coverage = convert(_atype,zeros(Tx,1, B))
    # List of coverages at each timestamp if needed
    # coverage_records = [coverage]

    for t in 1:size(tgt2,2)
        input,output = (@view tgttokens[:,t:t]), (@view tgt2[:,t:t])
        # TODO: Input masks are not considered in the decoding part
        cell, srcattnvector, srcattentions = decode(m.s, input, mem, cell)          ; @size cell (Hy,B,1); @size srcattnvector (Hy,B,1);  @size srcattentions (Tx,1,B);
        push!(hiddens, cell)
        srcattnvector = reshape(srcattnvector, Hy,B)                                ; @size srcattnvector (Hy,B);

        # Pointer probability. (pgen + psrc +ptgt =1)
        p = softmax(m.p.linearpointer(srcattnvector), dims=1)                       ; @size p (3,B);  # switch between pgen-psrc-ptgt
        p_copysrc = p[1:1,:]                                                        ; @size p_copysrc (1,B);
        p_copysrc = reshape(p_copysrc, (1,B,1))                                     ; @size p_copysrc (1,B,1);
        p_copytgt = p[2:2,:]                                                        ; @size p_copytgt (1,B);
        p_copytgt = reshape(p_copytgt, (1,B,1))                                     ; @size p_copytgt (1,B,1);
        p_gen = p[3:3,:]                                                            ; @size p_gen (1,B);
        p_gen = reshape(p_gen, (1,B,1))                                             ; @size p_gen (1,B,1);


        # Pgen: Probability distribution over the vocabulary.
        scores = m.p.projection(srcattnvector)                          ; @size scores (vocabsize,B);
        # Make score of decoder pad -Inf
        infvec = zeros(size(scores,1))
        infvec[decodervocab_pad_idx]=Inf
        scores = scores .- convert(_atype,infvec) 
        vocab_probs = softmax(scores, dims=1)                           ; @size vocab_probs (vocabsize,B)
        vocab_probs = reshape(vocab_probs, (vocabsize, B, 1))           ; @size vocab_probs (vocabsize,B,1);
        scaled_vocab_probs = vocab_probs .* p_gen                       ; @size scaled_vocab_probs (vocabsize,B,1);
        scaled_vocab_probs = permutedims(scaled_vocab_probs, [3,1,2])   ; @size scaled_vocab_probs (1,vocabsize,B);

        # Psrc: 
        srcattentions = reshape(srcattentions, (Tx,B,1))                ; @size srcattentions (Tx,B,1);
        scaled_srcattentions = srcattentions .* p_copysrc               ; @size scaled_srcattentions (Tx,B,1); @size srcattentionmaps (Tx,Tx+2,B); 
        scaled_copysrc_probs = bmm(permutedims(scaled_srcattentions, [3,1,2]), _atype(srcattentionmaps)) ; @size scaled_copysrc_probs (1,Tx+2,B);  # Tx+2 is the dynamic srcvocabsize

        # Ptgt: 
        #TODO: do tgtattentions
        tgtattentions = convert(_atype,(zeros(Ty, B,1)))
        tgtattentions = reshape(tgtattentions, (Ty,B,1))            ; @size tgtattentions (Ty,B,1);
        scaled_tgtattentions = tgtattentions .* p_copytgt           ; @size scaled_tgtattentions (Ty,B,1);  ; @size tgtattentionmaps (Ty,Ty+1,B);
        scaled_copytgt_probs = bmm(permutedims(scaled_tgtattentions, [3,1,2]), _atype(tgtattentionmaps))    ; @size scaled_copytgt_probs (1,Ty+1,B);    # Ty is the dynamic tgtvocabsize

                  
        #TODO: Do invalid indexes part.
        probs = cat(scaled_vocab_probs, scaled_copysrc_probs, scaled_copytgt_probs, dims=2)  ; @size probs (1,vocabsize+Tx+2+Ty+1,B)
        # Set probability of coref NA to 0
        NAvec = ones(1, size(probs, 2), 1)
        NAvec[vocabsize+Tx+3] = 0
        # After broadcasting the probability of coref NA is set to 0
        probs = probs .* convert(_atype,NAvec)
        getind(cartindx) = return cartindx[2]                                                            
        predictions = getind.(argmax(softmax(value(probs),dims=2), dims=2))  ;@size predictions (1, 1, B)
        predictions = convert(_atype, reshape(predictions, (1,B)))


        ### PointerGenerator Loss
        _generatetargets  = generatetargets[t:t,:]                   ; @size _generatetargets (1, B)
        _srccopytargets   = srccopytargets[t:t,:]                    ; @size _srccopytargets  (1, B) 
        _tgtcopytargets   = tgtcopytargets[t:t,:]                    ; @size _tgtcopytargets  (1, B)

        # Masks
        pad_idx = 0#trn.tgtvocab.token_to_idx["@@PADDING@@"]
        unk_idx = 1#trn.tgtvocab.token_to_idx["@@UNKNOWN@@"]
        non_pad_mask  = convert(_atype, _generatetargets .!= pad_idx)                                        ;@size non_pad_mask (1,B)                                               
        srccopy_mask  = convert(_atype, (_srccopytargets .!= unk_idx) .* (_srccopytargets .!= pad_idx))      ;@size srccopy_mask (1,B) # 1 is UNK id, 0 is pad id
        non_srccopy_mask = 1 .- srccopy_mask
        tgtcopy_mask = convert(_atype, (_tgtcopytargets .!= 0))                                              ;@size tgtcopy_mask (1,B) # 0 is the index for coref NA
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

        tgtcopytargets_withoffset = _tgtcopytargets .+ (vocabsize+src_dynamic_vocabsize+1)                         ;@size tgtcopytargets_withoffset (1,B) ## 
        tgtcopy_targetprobs = getprobs(tgtcopytargets_withoffset, probs)                        ;@size tgtcopy_targetprobs (1,B)
        tgtcopy_targetprobs = tgtcopy_targetprobs .* tgtcopy_mask                               ;@size tgtcopy_targetprobs (1,B)

        srccopytargets_withoffset = _srccopytargets .+ (vocabsize +1)                           ;@size srccopytargets_withoffset (1,B) 
        srccopy_targetprobs = getprobs(srccopytargets_withoffset, probs)                        ;@size srccopy_targetprobs (1,B)
        srccopy_targetprobs = srccopy_targetprobs .* srccopy_mask .* non_tgtcopy_mask           ;@size srccopy_targetprobs (1,B)    

        generate_targetprobs = getprobs(_generatetargets, probs)                                ;@size generate_targetprobs (1,B)
        generate_targetprobs = generate_targetprobs .* non_srccopy_mask .* non_tgtcopy_mask     ;@size generate_targetprobs (1,B) 

        # Except copy-oov nodes, all other nodes should be copied.
        likelihood = generate_targetprobs + tgtcopy_targetprobs + srccopy_targetprobs           ;@size likelihood (1,B)
        num_tokens = sum(non_pad_mask .== 1)

        if !(force_copy)
            non_generate_oov_mask = (_generatetargets .!= unk_idx)  
            additional_generate_mask = non_tgtcopy_mask .* srccopy_mask .* non_generate_oov_mask #?
            likelihood = likelihood + generate_targetprobs .* additional_generate_mask
            num_tokens += sum(additional_generate_mask .== unk_idx)
        end

        # Add eps for numerical stability.
        likelihood = likelihood .+ 1e-20 #eps

        loss = sum(-log.(likelihood) .* non_pad_mask) + coverage_loss     # Drop pads.
        sumloss += loss

        # Mask out copy targets for which copy does not happen.
        targets = _atype(tgtcopytargets_withoffset) .* tgtcopy_mask + 
                  _atype(srccopytargets_withoffset) .* srccopy_mask .* non_tgtcopy_mask +
                  _atype(_generatetargets) .* non_srccopy_mask .* non_tgtcopy_mask                    ;@size targets (1,B)


        targets = targets .* non_pad_mask                       ;@size targets (1,B)
        pred_eq = (predictions .== targets) .* non_pad_mask     ;@size targets (1,B)
        num_non_pad = sum(non_pad_mask .== 1) #num_tokens
        num_correct_pred = sum(pred_eq .== 1)
        num_target_copy =  sum((tgtcopy_mask .*   non_pad_mask) .==1)
        num_correct_target_copy =  sum((pred_eq .* tgtcopy_mask) .==1)
        num_correct_target_point = sum( ((predictions .>= vocabsize+src_dynamic_vocabsize+1) .* tgtcopy_mask .* non_pad_mask) .==1)
        num_source_copy =  sum((srccopy_mask .* non_tgtcopy_mask .*  non_pad_mask) .==1)
        num_correct_source_copy =  sum((pred_eq .* non_tgtcopy_mask .* srccopy_mask) .==1)
        num_correct_source_point = sum( ((predictions .>= vocabsize+1) .* (predictions .< vocabsize+src_dynamic_vocabsize+1)  .*non_tgtcopy_mask .*srccopy_mask .* non_pad_mask) .== 1)

        ### PointerGenerator Metrics and Statistics
        m.metrics.n_words += num_non_pad
        m.metrics.n_correct += num_correct_pred
        m.metrics.n_source_copies += num_source_copy
        m.metrics.n_correct_source_copies += num_correct_source_copy
        m.metrics.n_correct_source_points += num_correct_source_point
        m.metrics.n_target_copies += num_target_copy
        m.metrics.n_correct_target_copies += num_correct_target_copy
        m.metrics.n_correct_target_points += num_correct_target_point
    end
  
    hiddens = cat(hiddens...,dims=3) ;@size hiddens (Hy,B,Ty-1)
    graphloss = m.g(hiddens, parsermask, edgeheads, edgelabels)
    sumloss += graphloss
    return sumloss
end


function resetmetrics(m::BaseModel)
    m.metrics.n_words = 0
    m.metrics.n_correct = 0
    m.metrics.n_source_copies = 0
    m.metrics.n_correct_source_copies = 0
    m.metrics.n_correct_source_points = 0
    m.metrics.n_target_copies = 0
    m.metrics.n_correct_target_copies = 0
    m.metrics.n_correct_target_points = 0

    m.g.metrics.unlabeled_correct = 0        
    m.g.metrics.labeled_correct = 0
    m.g.metrics.total_sentences = 0
    m.g.metrics.total_words =  0
    m.g.metrics.total_loss = 0
    m.g.metrics.total_edgenode_loss = 0
    m.g.metrics.total_edgelabel_loss = 0
end