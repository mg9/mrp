_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})


struct BiaffineAttention; we; wd; b; u; end

# ## Part 1. Model constructor
#
# The `BiaffineAttention` constructor takes the following arguments:
# * `encoder_inputsize`: the dimension of the encoder input
# * `decoder_inputsize`: the dimension of the decoder input
# * `num_labels`: the number of labels of the crf layer
function BiaffineAttention(encoder_inputsize::Int, decoder_inputsize::Int; num_labels=1)
    we = param(num_labels, encoder_inputsize)
    wd = param(num_labels, decoder_inputsize)
    b  = param(num_labels,1,1)
    u  = param(decoder_inputsize, encoder_inputsize, edge_node_m)
    BiaffineAttention(we, wd, b, u)
end


ba = BiaffineAttention(256, 256)

    
   

# * `e_input`: the encoder input, edge_node_h
# * `d_input`: the decoder input, edge_node_m
# * `mask_e`: the encoder mask
# * `mask_d`: the decoder mask
#
function (ba::BiaffineAttention)(e_input, d_input; e_mask=nothing, d_mask=nothing)
    # e_input: (edgenode_hiddensize, B, Tencoder)
    # d_input: (edgenode_hiddensize, B, Tdecoder)
    # e_mask: (Ty, B)
    # d_mask: (Ty, B)


    _, B, Tencoder = e_input
    _, B, Tdecoder = d_input

    # TODO: Check these operations !!
    out_e = mmul(ba.we, e_input) # -> (num_labels, B, Tencoder)
    out_d = mmul(ba.wd, d_input) # -> (num_labels, B, Tdecoder)


    d_input = reshape(d_input, (size(d_input,1),1,size(d_input,2),:))  # -> (edgenode_hiddensize, 1, B, Tdecoder)

        # output shape [batch, num_label, length_decoder, length_encoder]
        if self.biaffine:
            # compute bi-affine part
            # [batch, 1, length_decoder, input_size_decoder] * [num_labels, input_size_decoder, input_size_encoder]
            # output shape [batch, num_label, length_decoder, input_size_encoder]
            output = torch.matmul(input_d.unsqueeze(1), self.U)
            # [batch, num_label, length_decoder, input_size_encoder] * [batch, 1, input_size_encoder, length_encoder]
            # output shape [batch, num_label, length_decoder, length_encoder]
            output = torch.matmul(output, input_e.unsqueeze(1).transpose(2, 3))

            output = output + out_d + out_e + self.b
        else:
            output = out_d + out_d + self.b

        if mask_d is not None and mask_e is not None:
            output = output * mask_d.unsqueeze(1).unsqueeze(3) * mask_e.unsqueeze(1).unsqueeze(2)

        return output


end