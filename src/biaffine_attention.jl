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
    
    B,Ty,H  = size(input_d)
    num_labels = size(ba.u,3)

    ;@size ba.u (H, H, 1) 
    ;@size input_d (B, Ty, H) 
    ;@size input_e (B, Ty, H) 
    ;@size mask (B,Ty)
    
    ;@size ba.wd (num_labels,H)

    

    input_d = reshape(input_d, B*Ty,:)
    out_d = input_d * ba.wd'
    out_d = reshape(out_d, B,Ty,:) #B,Ty,num_labels
    out_d = reshape(permutedims(out_d, [1,3,2]), B,num_labels,Ty,1) #B,num_labels,Ty,1

    #_pyoutd = permutedims(biaffinedata["out_d"], [4,3,2,1])
    #@assert isapprox(_pyoutd, out_d)


    input_e = reshape(input_e, B*Ty,:) #B*Ty,H
    out_e = input_e * ba.we'
    out_e = reshape(out_e, B,Ty,:) #B,Ty,num_labels
    out_e = reshape(permutedims(out_e, [1,3,2]), B,num_labels,1,Ty) #B,num_labels,1,Ty

    #_pyoute = permutedims(biaffinedata["out_e"], [4,3,2,1])
    #@assert isapprox(_pyoute, out_e) 


    output_1 = input_d * reshape(ba.u, H,H)  # B*T, H*num_labels
    output_1 = reshape(output_1, B,Ty,H,num_labels)
    output_1 = permutedims(output_1, [1,4,2,3])   ;@size output_1 (B,num_labels,Ty,H)
    #@assert isapprox(output_1,permutedims(biaffinedata["output_1"], [4,3,2,1]))

    input_e = reshape(input_e, B,Ty,H)
    input_e = reshape(input_e, B,1,Ty,H)
    input_e = permutedims(input_e, [1,2,4,3])  ; @size input_e (B,1,H,Ty) 


    _output_1 = permutedims(output_1, [4,3,2,1]) ;@size _output_1 (H,Ty,num_labels,B)
    _input_e = permutedims(input_e, [4,3,2,1]) ;@size _input_e (Ty,H,1,B)


    output_2 = bmm(_input_e, _output_1 )  ;@size output_2 (Ty,Ty,num_labels,B)
    output_2= permutedims(output_2, [4,3,2,1]) ;@size output_2 (B,num_labels, Ty,Ty)
   # @assert isapprox(output_2,permutedims(biaffinedata["output_2"], [4,3,2,1]))


    output = output_2 .+ out_d .+ out_e .+ ba.b  ;@size output (B,num_labels, Ty,Ty)
    #@assert isapprox(output,permutedims(biaffinedata["output_3"], [4,3,2,1]))



    if mask !== nothing
        mask_1= reshape(mask, B,1,Ty,1)
        mask_2= reshape(mask, B,1,1,Ty)
        output  = output .* mask_1 .* mask_2
        #@assert isapprox(output,permutedims(biaffinedata["masked_output_4"], [4,3,2,1]))
    end
    return output
end


#=
function (ba::BiaffineAttention)(input_d, input_e, mask)
   
    B,Ty,H  = size(input_d)
    num_labels = size(ba.u,3)

    ;@size ba.u (H, H, 1) 
    ;@size input_d (B, Ty, H) 
    ;@size input_e (B, Ty, H) 
    ;@size mask (B,Ty)
    
    ;@size ba.wd (H,num_labels)


    input_d = reshape(input_d, B*Ty,:)
    out_d = input_d * ba.wd
    out_d = reshape(out_d, B,Ty,:) #B,Ty,num_labels
    out_d = reshape(permutedims(out_d, [1,3,2]), B,num_labels,Ty,1) #B,num_labels,Ty,1

    _pyoutd = permutedims(biaffinedata["out_d"], [4,3,2,1])
    @assert isapprox(_pyoutd, out_d)


    input_e = reshape(input_e, B*Ty,:) #B*Ty,H
    out_e = input_e * ba.we
    out_e = reshape(out_e, B,Ty,:) #B,Ty,num_labels
    out_e = reshape(permutedims(out_e, [1,3,2]), B,num_labels,1,Ty) #B,num_labels,1,Ty

    _pyoute = permutedims(biaffinedata["out_e"], [4,3,2,1])
    @assert isapprox(_pyoute, out_e) 

    input_left= input_d'    # H,B*Ty
    input_right = input_e'


    input_left = reshape(input_left, (H, :))                                   # -> HL, S
    # Calculate x'U, or the left half of x'Uy
    # inputs: input_left -> HL, (BxT) -> Transpose    -> (BxT), HL
    #         bl.u       -> HL, O, HR -> will reshape -> HL, (OxHR)     
    left = input_left' * reshape(ba.u, (H, :))                                 # -> S, (OxHR)
    left = reshape(left, (:, num_labels, H))                                          
    left = reshape(left, (B,Ty, num_labels, H))                                       # B,T,O,H
    @assert isapprox(left,midmid)
    input_right = permutedims(input_right, [1,3,2]) # H,T,B
    left = permutedims(left, [2,4,3,1])           #T, H, O,B                                 
    input_right = reshape(input_right, HR,Ty,1,B) # H, T, 1, B
    # Calculate x'U times y
    out = bmm(left, input_right)     
    out = permutedims(out, [4,3,2,1])      # B,O,T ,T                                        
    @assert isapprox(out, output_middle)            #T,T,O,B 

end
=#
