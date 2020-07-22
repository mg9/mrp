# Script for comparison with the original pytorch model.

include("embed.jl")
include("pointer_generator.jl")
include("metrics.jl")
include("deep_biaffine_graph_decoder.jl")
include("bilinear.jl")

using HDF5

_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})


path = "../data/model_statedict_epoch_4.h5"
function rnn_cell(ws...; bidirectional=false, perm=[1,2,3,4])
    # RNN.w is organized in memory as follows:
    # w1xi, w1xf, w1xc, w1xo, w1hi, w1hf, w1hc, w1ho, w2xi, ..., wLho
    # b1xi, b1xf, b1xc, b1xo, b1hi, b1hf, b1hc, b1ho, b2xi, ..., bLho
    # for bidirectional fw and bw layers alternate

    # ws contains w1,b1,w2,b2...,wL,bL
    # wi=(2H,4H), bi=(4H,) for each layer except the first
    # w1=(X+H,4H), b1=(4H,)

    # [i,c,f,o] -> [i,f,c,o] (tf->cudnn)
    # [1,3,2,4] best permutation, the following also confirms the i,c,f,o order:
    # https://stats.stackexchange.com/questions/280995/accessing-lstm-weights-tensors-in-tensorflow
    println("ws:", size.(ws))
    XH,H4 = size(ws[1])
    H = H4 ÷ 4; X = XH - H
    L = length(ws) ÷ 2
    L2 = bidirectional ? L ÷ 2 : L
    r = RNN(X, H; numLayers=L2, bidirectional=bidirectional)
    rw = r.w.value; i2 = 0
    for m in 1:2, l in 1:L, i in 1:8
        println("m: $m, l: $l, i:$i")
        # Determine target location
        inputlayer = (l==1 || (bidirectional && l==2))
        len = (m==2 ? H : inputlayer && i<=4 ? X*H : H*H)
        i1,i2 = i2+1,i2+len
        # Determine source array/location
        w = ws[2l+m-2]
        j = i > 4 ? i-4 : i
        j = perm[j]        # source order: i,f,c,o => i,c,f,o
        if m==2 && i<=4 # no need to use both biases
            j1,j2 = ((j-1)*H+1),(j*H)
            # rw[i1:i2] = w[j1:j2]
            # This hack is for compatibility with forget_bias=1.0 in nmt
            rw[i1:i2] = (i == 2 ? (1 .+ w[j1:j2]) : w[j1:j2])
        elseif m==2 # i>4
            rw[i1:i2] .= 0
        elseif i<=4 # m==1 input matrices
            # w has dims (X+H,4H), already transposed
            # hopefully X comes first and columns follow the i,c,f,o order
            rows = (inputlayer ? (1:X) : (1:H))
            cols = ((j-1)*H+1):(j*H)
            rw[i1:i2] .= vec(w[rows,cols])
        else                    # hidden matrices
            rows = (inputlayer ? (X+1:X+H) : (H+1:2H))
            cols = ((j-1)*H+1):(j*H)
            rw[i1:i2] .= vec(w[rows,cols])
        end
    end
    println("i2: $i2:", length(rw))
    @assert i2 == length(rw)
    return r
end


function S2S(srcvocab,tgtvocab; perm=[1,3,2,4])
    srctokenembed    = Embed(mparams["encoder_token_embedding.weight"])         # 300x18002
    srcposembed      = Embed(mparams["encoder_pos_embedding.weight"])           # 100x52
    srcmustcopyembed = Embed(mparams["encoder_must_copy_embedding.weight"])     # 50x3

    tgttokenembed = Embed(mparams["decoder_token_embedding.weight"])    # 300x18002
    tgtposembed   = Embed(mparams["decoder_pos_embedding.weight"])      # 100x52
    tgtcorefembed = Embed(mparams["decoder_coref_embedding.weight"])    # 50x500

    w1 = vcat(mparams["encoder._module.forward_layer_0.input_linearity.weight"], mparams["encoder._module.forward_layer_0.state_linearity.weight"])
    b1 = mparams["encoder._module.forward_layer_0.state_linearity.bias"]
    w2 = vcat(mparams["encoder._module.backward_layer_0.input_linearity.weight"], mparams["encoder._module.backward_layer_0.state_linearity.weight"])
    b2 = mparams["encoder._module.backward_layer_0.state_linearity.bias"]

    w3 = vcat(mparams["encoder._module.forward_layer_1.input_linearity.weight"], mparams["encoder._module.forward_layer_1.state_linearity.weight"])
    b3 = mparams["encoder._module.forward_layer_1.state_linearity.bias"]
    w4 = vcat(mparams["encoder._module.backward_layer_1.input_linearity.weight"], mparams["encoder._module.backward_layer_1.state_linearity.weight"])
    b4 = mparams["encoder._module.backward_layer_1.state_linearity.bias"]
    encoder = rnn_cell(w1,b1, bidirectional=false, perm=perm)


    #Our decoder part does not match with original one.
    #=
    memory = Memory(kget(w, "dynamic_seq2seq/decoder/memory_layer/kernel:0"))
    tgtembed = Embed(kget(w, "embeddings/decoder/embedding_decoder:0"))
    @assert size(tgtembed.w,2) == length(tgtvocab)
    decoder = rnn_cell(get(w,"dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0"),
                       get(w,"dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0"),
                       get(w,"dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0"),
                       get(w,"dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0"),
                       perm=perm)
    wattn = kget(w, "dynamic_seq2seq/decoder/attention/attention_layer/kernel:0")
    attng = kget1(w, "dynamic_seq2seq/decoder/attention/luong_attention/attention_g:0")
    attention = Attention(1, wattn, attng)
    projection = Dense(kget(w, "dynamic_seq2seq/decoder/output_projection/kernel:0"))
    @assert size(projection.w,1) == length(tgtvocab)
    inputfeeding = true
    dropout = 0.2
    S2S(srcembed,encoder,memory,tgtembed,decoder,attention,projection,inputfeeding,dropout,srcvocab,tgtvocab)
    =#
end


weightspath = "/kuacc/users/mugekural/workfolder/stog/model.h5"
weights = h5open(weightspath, "r") do file
    read(file)
end

pointerresultspath = "/kuacc/users/mugekural/workfolder/stog/pointergeneretor_results.h5"
pointerdata = h5open(pointerresultspath, "r") do file
    read(file)
end

batchpath = "/kuacc/users/mugekural/workfolder/stog/preparebatch.h5"
data = h5open(batchpath, "r") do file
    read(file)
end

pglossthingspath = "/kuacc/users/mugekural/workfolder/stog/pointergeneretor_loss.h5"
pglossthingsdata = h5open(pglossthingspath, "r") do file
    read(file)
end

seq2seqmet = "/kuacc/users/mugekural/workfolder/stog/seq2seqmetrics.h5"
seq2seqmetrics = h5open(seq2seqmet, "r") do file
    read(file)
end


# Mock targets
generatetargets = pglossthingsdata["generate_targets"]    # Ty,B   25,3
srccopytargets = pglossthingsdata["source_copy_targets"]  # Ty,B   25,3
tgtcopytargets = pglossthingsdata["target_copy_targets"]  # Ty,B   25,3


# Mock Pg
pg = PointerGenerator(1024, 12202)
pg.projection.w[:] = _atype(weights["generator.linear.weight"]')
pg.projection.b[:] = _atype(weights["generator.linear.bias"]')
pg.linearpointer.w[:] = _atype(weights["generator.linear_pointer.weight"]')
pg.linearpointer.b[:] = _atype(weights["generator.linear_pointer.bias"])


# Mock inputs
hiddens = _atype(pointerdata["hiddens"]) #1024,75
srcattentions = _atype(pointerdata["source_attentions"]) # 39×25×3
tgtattentions = _atype(pointerdata["target_attentions"]) # 25×25×3
srcattentionmaps = _atype(pointerdata["source_attention_maps"]) # 41×39×3
tgtattentionmaps = _atype(pointerdata["target_attention_maps"]) # 26×25×3

attnvector = hiddens
#hiddens = reshape(hiddens, :,1,3) 
#srcattentions = permutedims(srcattentions, [1,3,2]) 
#tgtattentions = permutedims(tgtattentions, [1,3,2]) 
srcattentionmaps = permutedims(srcattentionmaps, [2,1,3]) # 39,41,3
tgtattentionmaps = permutedims(tgtattentionmaps, [2,1,3]) # 25,26,3
decodervocab_pad_idx=1
reset(pg.metrics)
#pg(hiddens, srcattentions, tgtattentions, srcattentionmaps, tgtattentionmaps,generatetargets, srccopytargets, tgtcopytargets, 1)


function (pg::PointerGenerator)(attnvector, srcattentions, tgtattentions, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, decodervocab_pad_idx; coverage=nothing, force_copy=true)

    Hy = size(attnvector,1)
    Tx,Ty,B = size(srcattentions)
    numtarget,Ty,B = size(tgtattentions)


    ; @size attnvector (Hy,Ty*B)
    ; @size srcattentionmaps (Tx,Tx+2,B); @size tgtattentionmaps (numtarget,numtarget+1,B);
    ; @size generatetargets (Ty,B); @size srccopytargets (Ty,B); @size tgtcopytargets (Ty,B)


    # Pointer probability. (pgen + psrc +ptgt =1)
    p = softmax(pg.linearpointer(attnvector), dims=1)                            ; @size p (3,Ty*B);  # switch between pgen-psrc-ptgt
    p_copysrc = p[1:1,:]                                                        ; @size p_copysrc (1,Ty*B);
    p_copysrc = reshape(p_copysrc, (1,Ty,B))                                     ; @size p_copysrc (1,Ty,B);
    p_copytgt = p[2:2,:]                                                        ; @size p_copytgt (1,Ty*B);
    p_copytgt = reshape(p_copytgt, (1,Ty,B))                                     ; @size p_copytgt (1,Ty, B);
    p_gen = p[3:3,:]                                                            ; @size p_gen (1,Ty*B);
    p_gen = reshape(p_gen, (1,Ty, B))                                             ; @size p_gen (1,Ty,B);


    # Pgen: Probability distribution over the vocabulary.
    scores = pg.projection(attnvector)                             ; @size scores (pg.vocabsize,Ty*B);
    # Make score of decoder pad -Inf
    infvec = zeros(size(scores,1))
    infvec[decodervocab_pad_idx]=Inf
    scores = scores .- convert(_atype,infvec) 

    scores = reshape(scores, :,Ty,B)
    #scores .- _atype(pointerdata["scores"])
    @assert isapprox(scores[2:end,:,:], _atype(pointerdata["scores"][2:end,:,:]))
    scores = reshape(scores, :,Ty*B)


    vocab_probs = softmax(scores, dims=1)                           ; @size vocab_probs (pg.vocabsize,Ty*B)
    vocab_probs = reshape(vocab_probs, (pg.vocabsize, Ty,B))           ; @size vocab_probs (pg.vocabsize,Ty,B);
    scaled_vocab_probs = vocab_probs .* p_gen                       ; @size scaled_vocab_probs (pg.vocabsize,Ty,B);
    scaled_vocab_probs = permutedims(scaled_vocab_probs, [2,1,3])   ; @size scaled_vocab_probs (Ty,pg.vocabsize,B);
    @assert isapprox(permutedims(scaled_vocab_probs, [2,1,3]), _atype(pointerdata["scaled_vocab_probs"]))

    # Psrc: 
    #srcattentions = reshape(srcattentions, (Tx,B,Ty))                ; @size srcattentions (Tx,B,Ty);
    scaled_srcattentions = srcattentions .* p_copysrc                ; @size scaled_srcattentions (Tx,Ty,B); 
    scaled_copysrc_probs = bmm(permutedims(scaled_srcattentions, [2,1,3]), _atype(srcattentionmaps)) ; @size scaled_copysrc_probs (Ty,Tx+2,B);  # Tx+2 is the dynamic srcvocabsize
    #checked
    @assert isapprox(scaled_srcattentions, pointerdata["scaled_source_attentions"])
    @assert isapprox(scaled_copysrc_probs, permutedims(pointerdata["scaled_copy_source_probs"], [2,1,3]))


    # Ptgt: 
    scaled_tgtattentions = tgtattentions .* p_copytgt           ; @size scaled_tgtattentions (numtarget,Ty, B);  
    scaled_copytgt_probs = bmm(permutedims(scaled_tgtattentions, [2,1,3]), _atype(tgtattentionmaps))    ; @size scaled_copytgt_probs (Ty,numtarget+1,B);    # Ty is the dynamic tgtvocabsize
    #checked
    @assert isapprox(scaled_tgtattentions, pointerdata["scaled_target_attentions"])
    @assert isapprox(scaled_copytgt_probs, permutedims(pointerdata["scaled_copy_target_probs"], [2,1,3]))



    #TODO: Do invalid indexes part.
    probs = cat(scaled_vocab_probs, scaled_copysrc_probs, scaled_copytgt_probs, dims=2)  ; @size probs (Ty,pg.vocabsize+Tx+2+numtarget+1,B)
    # Set probability of coref NA to 0
    NAvec = ones(1, size(probs, 2), 1)
    NAvec[pg.vocabsize+Tx+3] = 0
    # After broadcasting the probability of coref NA is set to 0
    probs = probs .* convert(_atype,NAvec)
    #checked is approx
    @assert isapprox(probs, permutedims(pointerdata["probs"], [2,1,3]))


    getind(cartindx) = return cartindx[2]                                                            
    predictions = getind.(argmax(softmax(value(probs),dims=2), dims=2))  ;@size predictions (Ty, 1, B)
    predictions = convert(_atype, reshape(predictions, (Ty,B)))


    ### LOSS 

    ### PointerGenerator Loss
    ; @size generatetargets (Ty, B);  ; @size srccopytargets  (Ty, B);  ; @size tgtcopytargets  (Ty, B)

    # Masks
    pad_idx = 0 #trn.tgtvocab.token_to_idx["@@PADDING@@"]
    unk_idx = 1 #trn.tgtvocab.token_to_idx["@@UNKNOWN@@"]
    non_pad_mask  = convert(_atype, generatetargets .!= pad_idx)                                        ;@size non_pad_mask (Ty,B)                                               
    srccopy_mask  = convert(_atype, (srccopytargets .!= unk_idx) .* (srccopytargets .!= pad_idx))      ;@size srccopy_mask (Ty,B) # 2 is UNK id, 1 is pad id
    non_srccopy_mask = 1 .- srccopy_mask
    tgtcopy_mask = convert(_atype, (tgtcopytargets .!= pad_idx))                                              ;@size tgtcopy_mask (Ty,B) # 0 is the index for coref NA
    non_tgtcopy_mask = 1 .- tgtcopy_mask
    #masks checked


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

    ## TMP
    generatetargets = pglossthingsdata["generate_targets"]  # Ty,B   25,3
    srccopytargets = pglossthingsdata["source_copy_targets"]   # Ty,B   25,3
    tgtcopytargets = pglossthingsdata["target_copy_targets"]  # Ty,B   25,3


    tgtcopytargets_withoffset = tgtcopytargets .+ (pg.vocabsize+src_dynamic_vocabsize+1)        ;@size tgtcopytargets_withoffset (Ty,B) ## 
    tgtcopy_targetprobs = reshape(probs[findindices(probs,convert(Array{Integer,2}, tgtcopytargets_withoffset))], Ty,B)                   ;@size tgtcopy_targetprobs (Ty,B)
    tgtcopy_targetprobs = tgtcopy_targetprobs .* tgtcopy_mask                               ;@size tgtcopy_targetprobs (Ty,B)
    @assert isapprox(tgtcopy_targetprobs, pglossthingsdata["target_copy_target_probs"])


    srccopytargets_withoffset = srccopytargets .+ (pg.vocabsize +1)                           ;@size srccopytargets_withoffset (Ty,B) 
    srccopy_targetprobs = reshape(probs[findindices(probs, convert(Array{Integer,2}, srccopytargets_withoffset))], Ty,B)               ;@size srccopy_targetprobs (Ty,B)
    srccopy_targetprobs = srccopy_targetprobs .* srccopy_mask .* non_tgtcopy_mask           ;@size srccopy_targetprobs (Ty,B)    
    @assert isapprox(srccopy_targetprobs, pglossthingsdata["source_copy_target_probs"])


    gentargets_withoffset = generatetargets .+ 1                                            ;@size srccopytargets_withoffset (Ty,B) 
    generate_targetprobs = reshape(probs[findindices(probs,convert(Array{Integer,2}, gentargets_withoffset))], Ty,B)               ;@size srccopy_targetprobs (Ty,B)
    @assert isapprox(pglossthingsdata["generate_target_probs"],generate_targetprobs)
    generate_targetprobs = generate_targetprobs .* non_srccopy_mask .* non_tgtcopy_mask     ;@size generate_targetprobs (Ty,B) 

   ## TMP
    generatetargets = pglossthingsdata["generate_targets"]  # Ty,B   25,3
    srccopytargets = pglossthingsdata["source_copy_targets"]   # Ty,B   25,3
    tgtcopytargets = pglossthingsdata["target_copy_targets"]  # Ty,B   25,3


   # Except copy-oov nodes, all other nodes should be copied.
    likelihood = generate_targetprobs + tgtcopy_targetprobs + srccopy_targetprobs           ;@size likelihood (Ty,B)
    isapprox( pglossthingsdata["likelihood"], likelihood)
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

    pg.metrics(predictions.-1, targets .-1, non_pad_mask, non_tgtcopy_mask, srccopy_mask, tgtcopy_mask, pg.vocabsize, src_dynamic_vocabsize, loss)
    
    accuracy, xent, ppl, srccopy_accuracy, tgtcopy_accuracy, n_words = calculate_pointergenerator_metrics(pg.metrics)
    println("PointerGeneratormetrics, all_acc=$accuracy src_acc=$srccopy_accuracy tgt_acc=$tgtcopy_accuracy, ppl=$ppl, n_words: ", n_words)
    return  probs, tgtcopy_targetprobs, srccopy_targetprobs, generate_targetprobs, likelihood, loss
end

pg.metrics.n_correct  = seq2seqmetrics["n_correct"]
pg.metrics.n_source_copies  = seq2seqmetrics["n_source_copies"]
pg.metrics.n_target_copies  = seq2seqmetrics["n_target_copies"]
pg.metrics.n_correct_source_points  = seq2seqmetrics["n_correct_source_points"]
pg.metrics.n_correct_target_copies  = seq2seqmetrics["n_correct_target_copies"]
pg.metrics.generatorloss  = seq2seqmetrics["loss"]
pg.metrics.n_words  = seq2seqmetrics["n_words"]
pg.metrics.n_correct_source_copies = seq2seqmetrics["n_correct_source_copies"]



## Graph decoder part tests


weightspath = "/kuacc/users/mugekural/workfolder/stog/model.h5"
weights = h5open(weightspath, "r") do file
    read(file)
end

graphdecoderpath = "/kuacc/users/mugekural/workfolder/stog/graphdecoder.h5"
graphdecoderdata = h5open(graphdecoderpath, "r") do file
    read(file)
end

attachmentscores = "/kuacc/users/mugekural/workfolder/stog/attachmentscores.h5"
attachmentscoresdata = h5open(attachmentscores, "r") do file
    read(file)
end

biaffinepath = "/kuacc/users/mugekural/workfolder/stog/biaffine.h5"
biaffinedata = h5open(biaffinepath, "r") do file
    read(file)
end


nnpath = "/kuacc/users/mugekural/workfolder/stog/nn.h5"
nn = h5open(nnpath, "r") do file
    read(file)
end


mmul(w,x) = (w == 1 ? x : w == 0 ? 0 : reshape(w * reshape(x,size(x,1),:), (:, size(x)[2:end]...)))


# Mock Pg
num_edgelabels = 141
edgenode_hiddensize= 256
edgelabel_hiddensize = 128
g = DeepBiaffineGraphDecoder(1024, edgenode_hiddensize, edgelabel_hiddensize, num_edgelabels)

g.biaffine_attention.u[:] = _atype(permutedims(weights["graph_decoder.biaffine_attention.U"], [3,2,1]))
g.biaffine_attention.wd[:] = _atype(weights["graph_decoder.biaffine_attention.W_d"]')
g.biaffine_attention.we[:] = _atype(weights["graph_decoder.biaffine_attention.W_e"]')
g.biaffine_attention.b[:] = _atype(weights["graph_decoder.biaffine_attention.b"])

g.edgelabel_bilinear.u[:] = _atype(permutedims(weights["graph_decoder.edge_label_bilinear.U"], [2,3,1]))
g.edgelabel_bilinear.wl[:] = _atype(weights["graph_decoder.edge_label_bilinear.W_l"]')
g.edgelabel_bilinear.wr[:] = _atype(weights["graph_decoder.edge_label_bilinear.W_r"]')
g.edgelabel_bilinear.b[:] = _atype(weights["graph_decoder.edge_label_bilinear.bias"])
g.edgenode_h_linear.w[:] = _atype(weights["graph_decoder.edge_node_h_linear.weight"]')
g.edgenode_h_linear.b[:] = _atype(weights["graph_decoder.edge_node_h_linear.bias"])
g.edgenode_m_linear.w[:] = _atype(weights["graph_decoder.edge_node_m_linear.weight"]')
g.edgenode_m_linear.b[:] = _atype(weights["graph_decoder.edge_node_m_linear.bias"])
g.edgelabel_h_linear.w[:] = _atype(weights["graph_decoder.edge_label_h_linear.weight"]')
g.edgelabel_h_linear.b[:] = _atype(weights["graph_decoder.edge_label_h_linear.bias"])
g.edgelabel_m_linear.w[:] = _atype(weights["graph_decoder.edge_label_m_linear.weight"]')
g.edgelabel_m_linear.b[:] = _atype(weights["graph_decoder.edge_label_m_linear.bias"])

g.head_sentinel[:] = _atype(weights["graph_decoder.head_sentinel"])


hiddens = _atype(graphdecoderdata["memory_bank_first"]) #1024×24×3
parsermask = _atype(graphdecoderdata["mask_first"]') # 3x24
edgeheads = _atype(graphdecoderdata["edge_heads_first"]')  # 3x24
edgelabels = _atype(graphdecoderdata["edge_labels_first"]') # 3x24

function (g::DeepBiaffineGraphDecoder)(hiddens, parsermask, edgeheads, edgelabels)
        
    Hy, num_nodes, B = size(hiddens)
    ;@size parsermask (B, num_nodes); @size edgeheads (B, num_nodes);  @size edgelabels (B, num_nodes)

    Ty = num_nodes+1
    dummy = convert(_atype, zeros(Hy,1,B))
    head_sentinel = g.head_sentinel .+ dummy   ;@size head_sentinel (Hy,1,B)
    hiddens = cat(head_sentinel, hiddens, dims=2)                                           ;@size hiddens (Hy,Ty,B)
    @assert isapprox(hiddens, _atype(graphdecoderdata["memory_bank"]))

    if !isnothing(edgeheads); edgeheads = cat(_atype(zeros(B,1)), edgeheads, dims=2); end           ;@size edgeheads  (B,Ty)
    if !isnothing(edgelabels); edgelabels = cat(_atype(zeros(B,1)), edgelabels, dims=2); end        ;@size edgelabels (B,Ty)
    parsermask = cat(_atype(ones(B,1)), parsermask, dims=2)                                         ;@size parsermask (B,Ty)


    hidden_rs = reshape(hiddens, :,Ty*B)


    ## Encode nodes
    edgenode_h = elu.(g.edgenode_h_linear(hidden_rs))        ;@size edgenode_h (edgenode_hiddensize, Ty*B)

    py_edgenodeh =permutedims(graphdecoderdata["edge_node_h"], [3,2,1]) # B,T,H
    edgenode_h = permutedims(reshape(edgenode_h, :,Ty,B), [3,2,1]) # B, T, H
    @assert isapprox(edgenode_h, py_edgenodeh)


    edgenode_m = elu.(g.edgenode_m_linear(hidden_rs))        ;@size edgenode_m (edgenode_hiddensize, Ty*B)
    py_edgenodem =permutedims(graphdecoderdata["edge_node_m"], [3,2,1]) # B,T,H
    edgenode_m = permutedims(reshape(edgenode_m, :,Ty,B), [3,2,1]) # B, T, H
    @assert isapprox(edgenode_m, py_edgenodem)


    edgelabel_h = elu.(g.edgelabel_h_linear(hidden_rs))      ;@size edgelabel_h (edgelabel_hiddensize, Ty*B)
    py_edgelabelh =permutedims(graphdecoderdata["edge_label_h"], [3,2,1]) # B,T,H
    edgelabel_h = permutedims(reshape(edgelabel_h, :,Ty,B), [3,2,1]) # B, T, H
    @assert isapprox(edgelabel_h, py_edgelabelh)


    edgelabel_m = elu.(g.edgelabel_m_linear(hidden_rs))      ;@size edgelabel_m (edgelabel_hiddensize, Ty*B)
    py_edgelabelm =permutedims(graphdecoderdata["edge_label_m"], [3,2,1]) # B,T,H
    edgelabel_m = permutedims(reshape(edgelabel_m, :,Ty,B), [3,2,1]) # B, T, H
    @assert isapprox(edgelabel_m, py_edgelabelm)


    ## Calculate edgeheads scores & loglikelihood
   
    input_d = permutedims(_atype(biaffinedata["input_d"]), [3,2,1])  ;@size input_d (B,Ty,edgenode_hiddensize)
    input_e = permutedims(_atype(biaffinedata["input_e"]), [3,2,1])  ;@size input_e (B,Ty, edgenode_hiddensize)
    mask = parsermask
    edgenode_scores = g.biaffine_attention(input_d, input_e, parsermask) ;@size edgenode_scores (B,1,Ty,Ty)
    edgenode_scores = reshape(edgenode_scores, B,Ty,Ty) 
    @assert isapprox(edgenode_scores, permutedims(_atype(graphdecoderdata["edge_node_scores"]),[3,2,1]))



    B, max_len, _ = size(edgenode_scores)
    tmpsoftmax_mask = (parsermask .== 1)                      ;@size tmpsoftmax_mask (B,Ty)
    tmpsoftmax_mask = tmpsoftmax_mask .+ 1e-45
    mask_1= reshape(tmpsoftmax_mask, B,1,Ty)
    mask_2= reshape(tmpsoftmax_mask, B,Ty,1)
    full_mask = mask_1 .+ mask_2
    py_mask = permutedims(_atype(graphdecoderdata["wtmask"]), [3,2,1]) # B,Ty,Ty
    @assert isapprox(full_mask, py_mask)
    
    edgenode_ll = log.(softmax(edgenode_scores + log.(full_mask .+ 1e-45), dims=2))
    py_edgenodell = permutedims(_atype(graphdecoderdata["edge_node_log_likelihood"]), [3,2,1]) # B,Ty,Ty
    @assert isapprox(edgenode_ll, py_edgenodell)


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


    #Mock dropout
    #Selection(getreps) works accurately.
    _k = permutedims(graphdecoderdata["edge_label_h_after_dp"], [3,2,1])
    edgeheads_inds  = convert(Array{Int32}, (edgeheads))  .+1  #increase for indices 0
    edgelabels_inds =convert(Array{Int32}, (edgelabels))  .+1  #increase for indices 0
    _k = permutedims(_k, [2,1,3])                               ;@size _k (Ty,B,edgelabel_hiddensize)
    _edgelabel_h = getreps(edgeheads_inds', _k)                         ;@size _edgelabel_h (Ty,B,edgelabel_hiddensize)
    _edgelabel_h = permutedims(_edgelabel_h, [3,2,1])                             ;@size _edgelabel_h (edgelabel_hiddensize,B,Ty)
    selected_labels= permutedims(graphdecoderdata["edge_label_h_getlabelscoress"], [1,3,2]) #128,3,25
    _edgelabel_h .==selected_labels
    _lh= _atype(graphdecoderdata["edge_label_h_getlabelscoress"])
    _lh =permutedims(_lh, [1,3,2])   #128,3,25  
    @assert isapprox(_lh, _edgelabel_h)


    # Bilinear works accurately
    _lm =_atype(graphdecoderdata["edge_label_m_getlabelscoress"])
    _lm =permutedims(_lm, [1,3,2])   #128,3,25
    edgelabel_scores = g.edgelabel_bilinear(_lh, _lm)                          ;@size edgelabel_scores (B,Ty, num_edgelabels)
    pyedgelabel_scores =_atype(graphdecoderdata["edge_label_scores"])
    pyedgelabel_scores = permutedims(pyedgelabel_scores, [3,2,1]) #3,25,141
    @assert isapprox(edgelabel_scores, pyedgelabel_scores) 


    # Loss calculation works accurately
    edgelabel_ll = log.(softmax(edgelabel_scores, dims=3))  # B,Ty,num_tags
    py_edgelabel_ll =  permutedims(_atype(graphdecoderdata["edge_label_log_likelihood"]), [3,2,1])
    @assert isapprox(edgelabel_ll, py_edgelabel_ll)


    ## Calculate edgeheads and edgelabel losses 
    edgeheadloss = edgelabelloss = 0.0

    for b in 1:B
        for t in 2:Ty # Exclude dummy root
            xh = edgenode_ll[b,edgeheads_inds[b,t],t]
            if isnan(xh) 
                println("xh NAN at $t $b ", edgeheads_inds[b,t]) 
            end
            println("edgeheadloss $edgeheadloss ")

            edgeheadloss += -xh

            xl = edgelabel_ll[b,t,edgelabels_inds[b,t]]
            if isnan(xl) println("xh: $xl") end
            edgelabelloss+= -xl
        end
    end
    ## Loss Calculation
    graphloss = sum(edgeheadloss + edgelabelloss)

    
    # Greedy decoding works accurately
    ## Greedy decoding for heads and labels
    edgenode_scores= permutedims(edgenode_scores, [3,2,1])

    ;@size edgenode_scores (Ty,Ty,B) ;@size parsermask (B,Ty); ;@size edgelabel_scores (B,Ty,num_edgelabels)
    diagonal(A::AbstractMatrix, k::Integer=0) = view(A, diagind(A, k)) # Set diagonal elements to -inf to prevent edge returns node itself (no loop)
    a = zeros(Ty,Ty); diagonal(a) .= -Inf
    a = reshape(a, (Ty,Ty,1))
    _edgenode_scores = edgenode_scores .+ convert(_atype,a)             ;@size _edgenode_scores (Ty,Ty,B)                       
    minus_mask = 1 .- parsermask
    settoinf(x) = x==1.0 ? x= -1e8 : x=x                                               
    minus_mask = settoinf.(Array(minus_mask))'                                 ;@size minus_mask (Ty,B)
    minus_mask = reshape(minus_mask, (1,Ty,B))                                         
    _edgenode_scores = _edgenode_scores .+ _atype(minus_mask)           ;@size _edgenode_scores (Ty,Ty,B)               
    minus_mask = reshape(minus_mask, (Ty,1,B))                                         
    _edgenode_scores = _edgenode_scores .+ _atype(minus_mask)           ;@size _edgenode_scores (Ty,Ty,B)               
    
    # Works accurately
    ## Predictions of edge_heads
    ina(x) = return x[2]     # remove cartesian type 
    pred_edgeheads = argmax(value(_edgenode_scores), dims=2)  
    pred_edgeheads = reshape(ina.(pred_edgeheads), (Ty,B))'             ;@size pred_edgeheads (B,Ty)
    graphdecoderdata["edge_heads_greedy"]' .+1 .== pred_edgeheads



    _k2 = permutedims(graphdecoderdata["edge_label_h_before"], [2,3,1]) #T,B,H
    _k3 = permutedims(graphdecoderdata["edge_label_m_before"], [2,3,1]) #T,B,H

    pred_edgeheads_inds  = convert(Array{Int32}, (pred_edgeheads))    #B,Ty
    _edgelabel_h = getreps(pred_edgeheads_inds', _k2)                                                    ;@size _edgelabel_h (Ty,B,edgelabel_hiddensize)
    _edgelabel_h = permutedims(_edgelabel_h, [3,2,1])                                                    ;@size _edgelabel_h (edgelabel_hiddensize,B,Ty)
    permutedims(_edgelabel_h,[2,3,1]) .== permutedims(graphdecoderdata["edge_label_h_getlabelscoress"], [3,2,1])

  

    _edgelabel_scores = g.edgelabel_bilinear(_atype(_edgelabel_h), _atype(permutedims(_k3, [3,2,1])))                ;@size edgelabel_scores (B,Ty, num_edgelabels)
    py_edgelabel_scores = permutedims(graphdecoderdata["edge_label_scores_greedy"], [3,2,1])
    @assert isapprox(_edgelabel_scores, py_edgelabel_scores)



    ## Predictions of edge_labels
    inpa(x) = return x[3]     # remove cartesian type 
    pred_edgelabels = argmax(value(_edgelabel_scores), dims=3)           ;@size pred_edgelabels (B,Ty,1)
    pred_edgelabels = reshape(inpa.(pred_edgelabels), (B,Ty))           ;@size pred_edgelabels (B,Ty)
    graphdecoderdata["edge_labels_greedy"]' .+1 .== pred_edgelabels


    ## Graph Decoder Metrics
    ;@size edgeheads_inds (B,Ty) ;@size edgelabels_inds (B,Ty) ;@size parsermask (B,Ty) ;@size pred_edgeheads (B,Ty) ;@size pred_edgelabels (B,Ty)
    # Exclude dummy root 
    _edgeheads = edgeheads_inds[:,2:end]               ;@size _edgeheads (B,Ty-1)
    _edgelabels = edgelabels_inds[:,2:end]             ;@size _edgelabels (B,Ty-1)
    _parsermask = parsermask[:,2:end]               ;@size _parsermask (B,Ty-1)
    _pred_edgeheads = pred_edgeheads[:,2:end]       ;@size _pred_edgeheads (B,Ty-1)
    _pred_edgelabels = pred_edgelabels[:,2:end]     ;@size _pred_edgelabels (B,Ty-1)
    g.metrics(_pred_edgeheads,_pred_edgelabels, _edgeheads, _edgelabels, _parsermask, graphloss, edgeheadloss, edgelabelloss)



   
    correct_indices = _atype((pred_edgeheads .== gold_edgeheads))  .*  parsermask 
    correct_labels = _atype((pred_edgelabels .== gold_edgelabels)) .* parsermask  
    correct_labels_and_indices = correct_indices .* correct_labels  

    @assert correct_indices == _atype(attachmentscoresdata["correct_indices"])'
    @assert correct_labels == _atype(attachmentscoresdata["correct_labels"])'
    @assert correct_labels_and_indices == _atype(attachmentscoresdata["correct_labels_and_indices"])'



    gm.unlabeled_correct += sum(correct_indices)    
    gm.labeled_correct += sum(correct_labels_and_indices)
    gm.total_sentences += size(pred_edgeheads,1)

    gm.total_words +=  sum(parsermask)
    gm.total_loss += graphloss
    gm.total_edgenode_loss += edgeheadloss
    gm.total_edgelabel_loss += edgelabelloss


    gm.unlabeled_correct = attachmentscoresdata["_unlabeled_correct"]
    gm.labeled_correct = attachmentscoresdata["_labeled_correct"]
    gm.total_sentences = attachmentscoresdata["_total_sentences"]
    gm.total_words = attachmentscoresdata["_total_words"]
    gm.total_loss = attachmentscoresdata["_total_loss"]

    gm.total_edgenode_loss = attachmentscoresdata["_total_edge_node_loss"]
    gm.total_edgelabel_loss = attachmentscoresdata["_total_edge_label_loss"]


    #Works accurately
    calculate_graphdecoder_metrics(gm)
    #(0.1446700507614213, 0.07106598984771574, 8.337492524064738)

    return graphloss
end


