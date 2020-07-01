using HDF5

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