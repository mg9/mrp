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
# * `mask`: the encoder, decoder masks
#
function (ba::BiaffineAttention)(input_e, input_d, mask)
    edgenode_hiddensize= size(input_e,1); B,Ty = size(mask) 
    ;@size ba.u (edgenode_hiddensize, edgenode_hiddensize,1) ;@size input_d (edgenode_hiddensize,B*Ty) ;@size mask (B,Ty)
    out_e = mmul(ba.we, reshape(input_e, (:, B,Ty)))                ;@size out_e (1,B,Ty) # 1 is the number of crf layer
    out_d = mmul(ba.wd, reshape(input_d, (:, B,Ty)))                ;@size out_d (1,B,Ty) # 1 is the number of crf layer
    output = mmul(input_e', ba.u)                                   ;@size output (B*Ty, edgenode_hiddensize, 1) 
    output = reshape(output, (B,Ty, edgenode_hiddensize, 1))            
    output = permutedims(output, [2,3,1,4])                         ;@size output (Ty,edgenode_hiddensize,B, 1) 
    input_d = reshape(input_d, (:, B, Ty,1))                        ;@size input_d (edgenode_hiddensize,B,Ty, 1)
    input_d = permutedims(input_d, [1,3,2,4])                       ;@size input_d (edgenode_hiddensize,Ty,B, 1) 
    output = bmm(output, input_d)                                   ;@size output  (Ty,Ty,B,1) # 1 is the number of crf layer  
    out_d = reshape(permutedims(out_d, [3,2,1]), 1,Ty,B,1)          ;@size out_d   (1,Ty,B,1)
    out_e = reshape(permutedims(out_e, [3,2,1]), 1,Ty,B,1)          ;@size out_e   (1,Ty,B,1)
    output = output .+ out_d .+ out_e  .+  ba.b                     ;@size output  (Ty,Ty,B,1)
    output = reshape(output, Ty,Ty,B)                               ;@size output  (Ty,Ty,B)
    if mask !== nothing
       rowmask = _atype(reshape(mask', (1,Ty,B)))
       output  = output .* rowmask
       colmask = _atype(reshape(mask', (Ty,1,B)))
       output  = output .* colmask
    end
    return output
end