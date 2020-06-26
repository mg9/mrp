    include("data.jl")
    include("basemodel.jl")
    import Base: length, iterate
    using YAML, Knet, IterTools, Random
    using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random, CuArrays, IterTools

    # Model configs
    BERT_EMB_DIM = 768
    TOKEN_EMB_DIM = 300 #glove
    POS_EMB_DIM = 100
    CNN_EMB_DIM = 100
    MUSTCOPY_EMB_DIM = 50
    COREF_EMB_DIM = 50
    POSTAG_VOCAB_SIZE = 51
    ENCODER_TOKEN_VOCAB_SIZE = 18002
    ENCODER_CHAR_VOCAB_SIZE = 125
    MUSTCOPY_VOCAB_SIZE = 3
    DECODER_TOKEN_VOCAB_SIZE = 12202
    DECODER_CHAR_VOCAB_SIZE = 87
    DECODER_COREF_VOCAB_SIZE = 500
    Ex = BERT_EMB_DIM + TOKEN_EMB_DIM + POS_EMB_DIM + MUSTCOPY_EMB_DIM + CNN_EMB_DIM
    Ey = TOKEN_EMB_DIM + POS_EMB_DIM + COREF_EMB_DIM + CNN_EMB_DIM
    H, L, Pdrop = 512, 2, 0.33
    vocabsize = 12202
    edgenode_hiddensize = 256
    edgelabel_hiddensize = 128
    num_labels = 141
    epochs = 10


    function iterate(d::Dataset, state=collect(1:d.totalbatch))
        new_state = copy(state)
        new_state_len = length(new_state) 
        if new_state_len == 0 ; return nothing; end
        batch = d.batches[new_state[1]]
        enc_inputs, dec_inputs, generator_inputs, parser_inputs = prepare_batch_input(batch)
        src, src_mask = prepare_encode_input(enc_inputs)                        # -> (Ex, B, Tx), (B, Tx)
        tgt = prepare_decode_input(enc_inputs, dec_inputs, parser_inputs)       # -> (Ey, B, Ty)
        generate_targets = generator_inputs["vocab_targets"]                    # -> (Ty, B)
        src_copy_targets = generator_inputs["copy_targets"]                     # -> (Ty, B)
        tgt_copy_targets = generator_inputs["coref_targets"]                    # -> (Ty, B)
        src_attention_maps = generator_inputs["copy_attention_maps"]            # -> (src_dynamic_vocabsize, Tx, B)
        tgt_attention_maps = generator_inputs["coref_attention_maps"]           # -> (tgt_dynamic_vocabsize, Ty, B)
        deleteat!(new_state, 1)
        return  ((src, src_mask, tgt, src_attention_maps, tgt_attention_maps, generate_targets, src_copy_targets, tgt_copy_targets) , new_state)
    end
    function length(d::Dataset); return d.totalbatch; end


    function trainmodel(epochs)
        batches = read_batches_from_h5("../data/data-100batches.h5")  # TODO: prepare batches dynamically, for now load them
        trn = Dataset(batches)
        trnbatches = collect(trn)
        model = BaseModel(H,Ex,Ey,L,vocabsize, edgenode_hiddensize, edgelabel_hiddensize, num_labels; bidirectional=true, dropout=Pdrop)
        epoch = adam(model, (( src, src_mask, tgt, src_attention_maps, tgt_attention_maps, generate_targets, src_copy_targets, tgt_copy_targets) for (src, src_mask, tgt, src_attention_maps, tgt_attention_maps, generate_targets, src_copy_targets, tgt_copy_targets) in trnbatches))
        ctrn = collect(trn)
        traindata = (collect(flatten(shuffle!(ctrn) for i in 1:epochs)))
        progress!(adam(model,traindata), seconds=10) do y
            println("hello seker")
        end
    end

    trainmodel(epochs)