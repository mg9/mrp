_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})


struct BiLinear; u; wl; wr; b; end

# ## Model constructor
#
# Applies a bilinear transformation to the incoming data: y = (x_1 A x_2) + b 
# The `BiLinear` constructor takes the following arguments:
# * `left_features`: size of left input
# * `right_features`: size of right input
# * `out_features`:  size of output (num of headtags)
function BiLinear(left_features::Int, right_features::Int, out_features::Int)
    u   = param(left_features, out_features, right_features)
    wl  = param(out_features, left_features)
    wr  = param(out_features, right_features) #or left_features?
    b   = param(out_features)
    BiLinear(u, wl, wr, b)
end


function (bl::BiLinear)(input_left, input_right)
    # input_left:  (edgelabel_hiddensize, B, T):  edgelabel_h
    # input_right: (edgelabel_hiddensize, B, T):  edgelabel_m
    # -> output: (B, T, out_features)

    left_size  = size(input_left)
    right_size = size(input_right)
    out_size   = size(bl.u,2)
    B = left_size[2]  # batch = int(np.prod(left_size[:-1]))

    input_left = reshape(input_left, size(input_left,1), size(input_left,2)* size(input_left,3))       # -> (left_features, B*T)
    input_left = input_left'                                                                           # -> (B*T, left_features)
    out = mmul(input_left, bl.u)                                                                       # -> (B*T, out_features, right_features)  

    input_right = reshape(input_right, size(input_right,1), size(input_right,3)*size(input_right,2))   # -> (right_features, B*T)
    input_right = input_right'                                                                         # -> (B*T, right_features)
    out = permutedims(out, [3,2,1])                                                                    # -> (right_features, out_features, B*T)  

    out = mmul(input_right, out)                                                                       # -> (B*T, out_features, B*T)  
    left_sum = sum(out,dims=1)                                                                         # -> (1, out_features,B*T)
    out = permutedims(out, [3,2,1])
    right_sum = sum(out,dims=1)                                                                        # -> (1, out_features,B*T)
    total = left_sum + right_sum
    total = reshape(total, (size(total,2),size(total,3)))                                              # -> (out_features,B*T)

    input_left = input_left'                                            # -> (left_features, B*T)
    input_right = input_right'                                          # -> (right_features,B*T)
    output = total+(bl.wl * input_left) + (bl.wr * input_right) .+bl.b  # -> (out_features, B*T)
    output = reshape(output, (B,:,out_size))                            # -> (B, T, out_features)
    return output
end
