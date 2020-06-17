using HDF5

mutable struct batchdata
    encoder_tokens
    encoder_characters
    src_token_ids
    src_token_subword_index
    src_must_copy_tags
    decoder_tokens
    decoder_characters
    src_pos_tags
    tgt_pos_tags 
    tgt_copy_indices 
    tgt_copy_mask
    tgt_copy_map
    src_copy_indices
    src_copy_map
    head_tags
    head_indices 
end

function read_batches_from_h5(file)
    ## For now data has 3 batches
    data = h5open(file, "r") do file
        read(file)
    end
    batches = Dict() 
    num_batches = Integer(length(data) / 16)
    for i in 1:num_batches
        batches[i] = Dict()
    end
    for (k,v) in data
        for i in 1:num_batches
            if occursin(string("batch_",i,"_"), k)
                field = replace(k, string("batch_",i,"_") => "")
                batches[i][field]= v
            end
        end
    end
    return batches
end



batches = read_batches_from_h5("../data/data.h5")
batch = batches[1]


function prepare_batch_input(batch)
    encoder_inputs = Dict(); decoder_inputs = Dict(); generator_inputs = Dict(); parser_inputs = Dict()
    
    ## Encoder inputs
    bert_token_inputs = batch["src_token_ids"]
    encoder_token_subword_index = batch["src_token_subword_index"]
    encoder_token_inputs = batch["encoder_tokens"]
    encoder_pos_tags = batch["src_pos_tags"]
    encoder_must_copy_tags = batch["src_must_copy_tags"]
    encoder_char_inputs = batch["encoder_characters"]
    encoder_mask = zeros(size(batch["encoder_tokens"])) ## TODO change here, not correct!!!

    encoder_inputs["bert_token"] = bert_token_inputs
    encoder_inputs["token_subword_index"] = encoder_token_subword_index
    encoder_inputs["token"] = encoder_token_inputs
    encoder_inputs["pos_tag"] = encoder_pos_tags
    encoder_inputs["must_copy_tag"] = encoder_must_copy_tags
    encoder_inputs["char"] = encoder_char_inputs
    encoder_inputs["mask"] = encoder_mask

    ## Decoder inputs
    decoder_token_inputs = batch["decoder_tokens"][1:end-1,:]
    decoder_pos_tags = batch["tgt_pos_tags"][1:end-1, : ]
    decoder_char_inputs = batch["decoder_characters"][:,1:end-1,:]
    decoder_coref_inputs = zeros(size(batch["tgt_copy_indices"][1:end-1,:])) ## TODO change here, not correct!!!

    decoder_inputs["token"] = decoder_token_inputs
    decoder_inputs["pos_tag"] = decoder_pos_tags
    decoder_inputs["char"] = decoder_char_inputs
    decoder_inputs["coref"] = decoder_coref_inputs


    ## Generator inputs, TODO check here again!!!
    # [batch, num_tokens]
    vocab_targets = batch["decoder_tokens"][2:end,:]
    # [batch, num_tokens]
    coref_targets = batch["tgt_copy_indices"][2:end,:]
    # [batch, num_tokens, num_tokens + coref_na]
    coref_attention_maps = batch["tgt_copy_map"][:,2:end,:]  # exclude BOS
    # [batch, num_tgt_tokens, num_src_tokens + unk]
    copy_targets = batch["src_copy_indices"][2:end,:]
    # [batch, num_src_tokens + unk, src_dynamic_vocab_size]
    # Exclude the last pad.
    copy_attention_maps = batch["src_copy_map"][:,2:end-1,:]

    generator_inputs["vocab_targets"] = vocab_targets
    generator_inputs["coref_targets"] = coref_targets
    generator_inputs["coref_attention_maps"] = coref_attention_maps
    generator_inputs["copy_targets"] = copy_targets
    generator_inputs["copy_attention_maps"] = copy_attention_maps


    ## Parser inputs
    # Remove the last two pads so that they have the same size of other inputs?
    edge_heads =  batch["head_indices"][1:end-2,:]
    edge_labels = batch["head_tags"][1:end-2,:]
    parser_mask = zeros(size(decoder_token_inputs)) ## TODO change here, not correct!!!

    parser_inputs["edge_heads"] = edge_heads
    parser_inputs["edge_labels"] = edge_labels
    parser_inputs["corefs"] = decoder_coref_inputs
    parser_inputs["mask"] = parser_mask
    return encoder_inputs, decoder_inputs, generator_inputs, parser_inputs
end

