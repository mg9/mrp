_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})


struct BiaffineAttention; we; wd; b; u; end

# ## Model constructor
#
# The `BiaffineAttention` constructor takes the following arguments:
# * `encoder_inputsize`: the dimension of the encoder input
# * `decoder_inputsize`: the dimension of the decoder input
# * `num_labels`: the number of labels of the crf layer
function BiaffineAttention(encoder_inputsize::Int, decoder_inputsize::Int; num_labels=1)
    we = param(num_labels, encoder_inputsize)
    wd = param(num_labels, decoder_inputsize)
    b  = param(num_labels,1,1)
    u  = param(decoder_inputsize, encoder_inputsize, num_labels)
    BiaffineAttention(we, wd, b, u)
end



# * `input_e`: the encoder input, edgenode_h
# * `input_d`: the decoder input, edgenode_m
# * `mask_e`: the encoder mask
# * `mask_d`: the decoder mask
#
function (ba::BiaffineAttention)(input_e, input_d; mask_e=nothing, mask_d=nothing)
    # input_e: (encoder_inputsize, B, Tencoder)
    # input_d: (decoder_inputsize, B, Tdecoder)
    # mask_e: (Ty, B)
    # mask_d: (Ty, B)
    # -> output: (B, Tencoder, Tdecoder, num_labels)

    _, B, Tencoder = size(input_e)
    _, B, Tdecoder = size(input_d)

    decoder_inputsize, encoder_inputsize, num_labels = size(ba.u)

    out_e = mmul(ba.we, input_e)                                                # -> (num_labels, B, Tencoder)
    out_d = mmul(ba.wd, input_d)                                                # -> (num_labels, B, Tdecoder)


    input_d = permutedims(input_d, [2,3,1])                                     # -> (B,Tdecoder, decoder_inputsize)
    input_d = reshape(input_d, (size(input_d,1) * size(input_d,2),:))           # -> (B*Tdecoder, decoder_inputsize)

    a = reshape(ba.u, size(ba.u,1),size(ba.u,2) * size(ba.u,3))                 # -> (decoder_inputsize, decoder_inputsize*num_labels)
    output = input_d * a                                                        # -> (B*Tdecoder, decoder_inputsize*num_labels)
    output = reshape(output, (B, Tdecoder, decoder_inputsize, num_labels))      # -> (B,Tdecoder, decoder_inputsize,num_labels)

    output = permutedims(output, [2,4,3,1])                                     # -> (Tdecoder,num_labels, decoder_inputsize, B)
    output = reshape(output, (size(output,1) * 
                size(output,2), size(output,3), size(output,4)))                # -> (Tdecoder*num_labels, decoder_inputsize, B)
    input_e = permutedims(input_e, [1,3,2])                                     # -> (decoder_inputsize, Tencoder, B)
    output = bmm(output,input_e)                                                # -> (Tdecoder*num_labels, Tencoder, B)
    output = reshape(output, (B, Tencoder, Tdecoder, num_labels))               # -> (B, Tencoder, Tdecoder, num_labels)
    out_e = permutedims(out_e, [2,3,1])                                         # -> (B, Tencoder, num_labels)
    out_e = reshape(out_e, (size(out_e,1), size(out_e,2), size(out_e,3), 1))    # -> (B, Tencoder, num_labels, 1)

    output = output .+out_e                                                     # -> (B, Tencoder, Tdecoder, num_labels)
    output = permutedims(output, [1,3,2,4])                                     # -> (B, Tdecoder, Tencoder, num_labels)
    
    out_d = permutedims(out_d, [2,3,1])                                         # -> (B, Tdecoder, num_labels)
    out_d = reshape(out_d, (size(out_d,1), size(out_d,2), size(out_d,3), 1))    # -> (B, Tdecoder, num_labels,1)

    output = output .+ out_d                                                    # -> (B, Tencoder, Tdecoder, num_labels)
    output = output .+ ba.b

    if mask_d === nothing && mask_e === nothing
        mask_d = permutedims(mask_d, [2,1])                                     # -> (B, Tdecoder)
        output = output .* mask_d                                               # -> (B, Tencoder, Tdecoder, num_labels)
        mask_e = permutedims(mask_e, [2,1])                                     # -> (B, Tencoder)
        output = output .* mask_e                                               # -> (B, Tencoder, Tdecoder, num_labels)
    end
    return output
end