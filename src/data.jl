using HDF5

START_SYMBOL = "@start@"
END_SYMBOL = "@end@"
#self.token_to_idx = {self.pad_token : 0, self.unk_token : 1} +=1 change them for 1-indexed


mutable struct Dataset
    batches::Dict
    totalbatch
end

function Dataset(batches)
    Dataset(batches, length(batches))
end


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
    ## For now data has only 5 batches
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


function prepare_batch_input(batch)
    encoder_inputs = Dict(); decoder_inputs = Dict(); generator_inputs = Dict(); parser_inputs = Dict()
    #batch["head_indices"] .+= 1 # head_indices are 0-indexed but julia 1-indexed so we increase them.
    #batch["head_tags"] .+= 1 # head_indices are 0-indexed but julia 1-indexed so we increase them.

    batch["encoder_tokens"] .+= 1
    batch["encoder_characters"] .+= 1

    batch["src_token_ids"] .+= 1
    batch["src_copy_indices"] .+= 1
    batch["src_copy_map"] .+= 1
    
    batch["decoder_tokens"] .+= 1
    batch["decoder_characters"] .+= 1
    batch["tgt_copy_indices"] .+= 1


    ## Encoder inputs
    bert_token_inputs = batch["src_token_ids"]
    encoder_token_subword_index = batch["src_token_subword_index"]
    encoder_token_inputs = batch["encoder_tokens"]
    encoder_pos_tags = batch["src_pos_tags"]
    encoder_must_copy_tags = batch["src_must_copy_tags"]
    encoder_char_inputs = batch["encoder_characters"]
    mask(x) = if x != 0; x=1; else; x=0; end
    encoder_mask = mask.(batch["encoder_tokens"]) 

    encoder_inputs["bert_token"] = bert_token_inputs
    encoder_inputs["token_subword_index"] = encoder_token_subword_index
    encoder_inputs["token"] = encoder_token_inputs
    encoder_inputs["pos_tag"] = encoder_pos_tags
    encoder_inputs["must_copy_tag"] = encoder_must_copy_tags            # TODO: How they define must copy tags?
    encoder_inputs["char"] = encoder_char_inputs
    encoder_inputs["mask"] = encoder_mask

    ## Decoder inputs
    decoder_token_inputs = batch["decoder_tokens"][1:end-1,:]           # -> (num_tokens,B) exclude EOS?
    decoder_pos_tags = batch["tgt_pos_tags"][1:end-1, : ]               # -> (num_tokens,B) exclude EOS?
    decoder_char_inputs = batch["decoder_characters"][:,1:end-1,:]      # -> (num_chars, num_tokens,B) exclude EOS?
    raw_coref_inputs = batch["tgt_copy_indices"][1:end-1,:]             # -> (num_tokens,B) exclude EOS?, TODO: check these copy indices
    coref_happen_mask = (raw_coref_inputs .!= 1)
    T, B = size(raw_coref_inputs)
    arange= reshape(repeat(collect(1:T),B), (T,B))
    decoder_coref_inputs =  arange
    b =decoder_coref_inputs.*coref_happen_mask
    decoder_coref_inputs = decoder_coref_inputs .- b
    decoder_coref_inputs += raw_coref_inputs                            # -> (num_tokens,B) exclude EOS?, TODO: is this line necessary?   

    decoder_inputs["token"] = decoder_token_inputs
    decoder_inputs["pos_tag"] = decoder_pos_tags
    decoder_inputs["char"] = decoder_char_inputs
    decoder_inputs["coref"] = decoder_coref_inputs


    ## Generator inputs, TODO: check here again!
    vocab_targets = batch["decoder_tokens"][2:end,:]                    # -> (num_tokens,B) exclude BOS
    coref_targets = batch["tgt_copy_indices"][2:end,:]                  # -> (num_tokens,B) exclude BOS
    #TODO: coref_attention_maps = batch["tgt_copy_map"][2:end,:,:]             # -> (num_tokens,num_tokens+coref_na, B) exclude BOS
    coref_attention_maps = batch["tgt_copy_map"][:,2:end,:]             # -> (num_tokens+coref_na,num_tokens,B) exclude BOS


    copy_targets = batch["src_copy_indices"][2:end,:]                   # -> (num_tokens,B) exclude BOS
    a = permutedims(batch["src_copy_map"] , [2,1,3])                    # -> (num_src_tokens +unk )  
    copy_attention_maps = a[:,2:end-1,:]                                # -> (rc_dynamic_vocab_size, num_src_tokens + unk , B) exclude BOS, the last pad
    
    #TODO:
    #_, src_dynamic_vocabsize, B= size(copy_attention_maps)
    #copy_attention_maps = hcat(1, zeros(src_dynamic_vocabsize,B),copy_attention_maps) # -> (num_src_tokens + unk, src_dynamic_vocab_size,  B) exclude the last pad
    #copy_attention_maps = permutedims(copy_attention_maps, [2,1,3])

    generator_inputs["vocab_targets"] = vocab_targets
    generator_inputs["coref_targets"] = coref_targets
    generator_inputs["coref_attention_maps"] = coref_attention_maps
    generator_inputs["copy_targets"] = copy_targets                     
    generator_inputs["copy_attention_maps"] = copy_attention_maps


    ## Parser inputs
                                                                        # Original comment: Remove the last two pads so that they have the same size of other inputs?
    edge_heads =  batch["head_indices"][1:end-2,:]                      # TODO: only one node, one head?
    edge_labels = batch["head_tags"][1:end-2,:]
    parser_token_inputs = copy(decoder_token_inputs) 
    id_END_SYMBOL = 3                                                   # TODO: take this dynamically
    parser_mask = (parser_token_inputs .== id_END_SYMBOL)               # -> (num_tokens,B) exclude EOS? 

    parser_inputs["edge_heads"] = edge_heads
    parser_inputs["edge_labels"] = edge_labels
    parser_inputs["corefs"] = decoder_coref_inputs
    parser_inputs["mask"] = parser_mask
    return encoder_inputs, decoder_inputs, generator_inputs, parser_inputs
end

