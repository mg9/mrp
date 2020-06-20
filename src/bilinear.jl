_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})


struct BiLinear; u; wl; wr; b; end

# ## Model constructor
#
# Applies a bilinear transformation to the incoming data: y = (x_1 A x_2) + b 
# The `BiLinear` constructor takes the following arguments:
# * `left_features`: size of left input
# * `right_features`: size of right input
# * `out_features`:  size of output
function BiLinear(left_features::Int, right_features::Int, out_features::Int)
    u   = param(out_features, left_features, right_features)
    wl  = param(out_features, left_features)
    wr  = param(out_features, right_features) #or left_features?
    b   = param(out_features)
    BiLinear(u, wl, wr, b)
end


bl = BiLinear(128, 128, 141)


function (bl::BiLinear)(input_left, input_right)
    # input_left:  (edgelabel_hiddensize, B, Tencoder):  edgelabel_h
    # input_right: (edgelabel_hiddensize, B, Tencoder):  edgelabel_m
    # -> output: (B, Tencoder, out_features)
   
    left_size  = size(input_left)
    right_size = size(input_right)
    B = left_size[2]  # batch = int(np.prod(left_size[:-1]))

    input_left = reshape(input_left, size(input_left,1), size(input_left,2)* size(input_left,3))       # -> (left_features, B*Tencoder)
    input_left = permutedims(input_left, [2,1])                                                        # -> (B*Tencoder, left_features)
    input_right = reshape(input_right, size(input_right,1), size(input_right,2)* size(input_right,3))  # -> (right_features, B*Tencoder)
    input_right = permutedims(input_right, [2,1])                                                      # -> (B*Tencoder, right_features)


    a = permutedims(bl.u, [2,1,3])                                                                     # -> (left_features, out_features, right_features)
    b = mmul(input_left, a)                                                                            # -> (B*Tencoder, out_features, right_features)                                                   
    c = permutedims(b, [3,2,1])                                                                        # -> (right_features, out_features, B*Tencoder)  
    d = mmul(input_right, c)                                                                           # -> (B*Tencoder, out_features, B*Tencoder)  

    # TODO: this part needs to be done
    # output [batch, out_features]
    output = F.bilinear(input_left, input_right, self.U, self.bias)
    output = output + F.linear(input_left, self.W_l, None) + F.linear(input_right, self.W_r, None)
    # convert back to [batch1, batch2, ..., out_features]
    return output.view(left_size[:-1] + (self.out_features, ))
end
