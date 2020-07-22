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
    u  = param(encoder_inputsize, decoder_inputsize, num_labels)
    BiaffineAttention(we, wd, b, u)
end


# * `input_d`: the encoder input, edgenode_h
# * `input_e`: the decoder input, edgenode_m
# * `mask`: the encoder, decoder masks
#
function (ba::BiaffineAttention)(input_d, input_e, mask)
    # (input_d * U * input_e) + (Wd * input_d + We * input_e + b)

    B,Ty,H  = size(input_d)
    num_labels = size(ba.u,3)

    ;@size ba.u (H, H, 1) 
    ;@size input_d (B, Ty, H) 
    ;@size input_e (B, Ty, H) 
    ;@size mask (B,Ty)
    ;@size ba.wd (num_labels,H)

    # Wd * input_d
    input_d = reshape(input_d, B*Ty,:)
    out_d = input_d * ba.wd'
    out_d = reshape(out_d, B,Ty,:) #B,Ty,num_labels
    out_d = reshape(permutedims(out_d, [1,3,2]), B,num_labels,Ty,1) 

    # Wd * input_e
    input_e = reshape(input_e, B*Ty,:) #B*Ty,H
    out_e = input_e * ba.we'
    out_e = reshape(out_e, B,Ty,:) #B,Ty,num_labels
    out_e = reshape(permutedims(out_e, [1,3,2]), B,num_labels,1,Ty) 

    # input_d x U
    output_1 = input_d * reshape(ba.u, H,H)  # B*T, H*num_labels
    output_1 = reshape(output_1, B,Ty,H,num_labels)
    output_1 = permutedims(output_1, [1,4,2,3])     ;@size output_1 (B,num_labels,Ty,H)
    input_e = reshape(input_e, B,Ty,H)
    input_e = reshape(input_e, B,1,Ty,H)
    input_e = permutedims(input_e, [1,2,4,3])       ;@size input_e (B,1,H,Ty) 
    _output_1 = permutedims(output_1, [4,3,2,1])    ;@size _output_1 (H,Ty,num_labels,B)
    _input_e = permutedims(input_e, [4,3,2,1])      ;@size _input_e (Ty,H,1,B)

    # U x input_e
    output_2 = bmm(_input_e, _output_1 )            ;@size output_2 (Ty,Ty,num_labels,B)
    output_2= permutedims(output_2, [4,3,2,1])      ;@size output_2 (B,num_labels, Ty,Ty)
    output = output_2 .+ out_d .+ out_e .+ ba.b     ;@size output (B,num_labels, Ty,Ty)

    # Apply mask if any
    if mask !== nothing
        mask_1= reshape(mask, B,1,Ty,1)
        mask_2= reshape(mask, B,1,1,Ty)
        output  = output .* mask_1 .* mask_2
    end
    return output
end
