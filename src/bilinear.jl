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
    # input_left: (HL, B...): edgelabel_h (HL: edgelabel_hiddensize)
    # input_right: (HR, B...): edgelabel_m (HR: edgelabel_hiddensize) 
    # -> Output: (B..., O) (O: out_features)
    # HL and HR should be the same for our usage, but this function generalizes to situations with
    # different sizes
    
    HL = size(input_left, 1)
    HR = size(input_right, 1)
    O = size(bl.u,2)
    @assert size(input_right)[2:end]==size(input_left)[2:end]
    @assert HL==size(bl.u,1)
    @assert HR==size(bl.u,3)
    B = size(input_right)[2:end]
    # S = Π(B...)
    
    input_left = reshape(input_left, (HL, :))                                   # -> HL, S
    
    # Calculate x'U, or the left half of x'Uy
    # inputs: input_left -> HL, (BxT) -> Transpose    -> (BxT), HL
    #         bl.u       -> HL, O, HR -> will reshape -> HL, (OxHR)     
    left = input_left' * reshape(bl.u, (HL, :))                                 # -> S, (OxHR)
    left = reshape(left, (:, O, HR))                                            # -> S, O, HR
    left = permutedims(left, [2, 3, 1])                                         # -> O, HR, S
    
    input_right = reshape(input_right, (HR, 1, :))                              # -> HR, 1, S
    # Calculate x'U times y
    out = bmm(left, input_right)                                                # -> O, 1, S
    
    out = reshape(out, (O, :))                                                  # -> O, S
    # add the bias                                                             
    out = out .+ bl.b                                                           # -> O, S
    
    # reshape for the next operation
    input_left = reshape(input_left, (HL, :))
    input_right = reshape(input_right, (HR, :))
    
    linear_out = bl.wl*input_left + bl.wr*input_right                           # -> O, S
    linear_out = reshape(linear_out, (O, :))
    
    # This is used in the paper (Extra addition to regular bilinear)
    out = out .+ linear_out                                                     # -> O, S
              
    out = permutedims(out, [2, 1])                                              # -> S, O
    
    # Double check the shape we want. For now, it will be B, T, O
    out = reshape(out, B..., O)                                                 # -> B..., O
end

function slowBilinear(bl::BiLinear, input_left, input_right)
    # input_left: (HL, B, T): edgelabel_h (HL: edgelabel_hiddensize)
    # input_right: (HR, B, T): edgelabel_m (HR: edgelabel_hiddensize) 
    # -> Output: (B, T, O) (O: out_features)
    # This is purely to test that the vectorized bilinear implementation works as intended
    # Do not use this anywhere other than for testing
    HL, B, T = size(input_left)
    HR = size(input_right,1)
    O = size(bl.u,2)
    
    out = zeros(B, T, O)
    for i in 1:B
        for j in 1:T
            for k in 1:O
                inpleft = input_left[:, i, j]      # -> HL
                inpleft = reshape(inpleft, (1, :)) # -> 1, HL
                inpright = input_right[:, i, j]    # -> HR
                u = bl.u[:, k, :]                  # -> HL x HR
                b = bl.b[k]                        # -> 1
                # calc x'Uy + b
                out[i, j, k] += (inpleft*u*inpright)[1]+b
                # reshape back to HL
                inpleft = reshape(inpleft, (:))
                # add wl*inpleft and wr*inpright
                wl = bl.wl[k, :]                   # -> HL
                wr = bl.wr[k, :]                   # -> HR
                out[i, j, k] += wl'*inpleft + wr'*inpright
            end
        end
    end
    out
end


@testset "Testing Bilinear Layer" begin
    HL, HR, O, B, T = 6, 6, 10, 26, 15
    bl = BiLinear(randn(HL, O, HR), randn(O, HL), randn(O, HR), randn(O))
    inp1 = randn(HL, B, T)
    inp2 = randn(HR, B, T)
    
    @test bl(inp1, inp2) ≈ slowBilinear(bl, inp1, inp2)
    
    # Test using regular constructor (Using GPU if available)
    inp1 = _atype(inp1); inp2 = _atype(inp2)
    bl = BiLinear(HL, HR, O)
    @test bl(inp1, inp2) ≈ slowBilinear(bl, inp1, inp2) 
end