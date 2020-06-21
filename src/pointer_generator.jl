include("s2s.jl")
include("linear.jl")

_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})


struct PointerGenerator
    projection::Linear          # decoderhiddens(B*Ty, Hy) -> vocabprobs(B,Ty, vocabsize)
    linear_pointer::Linear      # decoderhiddens(B*Ty, Hy) -> pointerprobs(B,Ty, 3): p_copy_source,p_copy_target, p_generate
end


# ## Model constructor
#
# The `PointerGenerator` constructor takes the following arguments:
# * `hidden`: size of the hidden vectors of the decoder
# * `vocab_size`: number of decoder tokens vocabsize
# * `eps`:
# * `force_copy`:
function PointerGenerator(hidden::Int, vocabsize::Int, eps::Float64;  force_copy=true)
    projection = Linear(hidden, vocabsize)
    switch_input_size = hidden
    linear_pointer = Linear(switch_input_size, 3)
    force_copy = true
    eps = 1e-20
    PointerGenerator(projection, linear_pointer) 
end

#-
@testset "Testing PointerGenerator" begin
    batches = read_batches_from_h5("../data/data.h5")
    tbatch = batches[1]
    vocab_pad_idx = 1 # 0th index
    vocabsize = 12202
    enc_inputs, dec_inputs, generator_inputs , parser_inputs = prepare_batch_input(tbatch)
    encoder_input, encoder_mask = prepare_encode_input(enc_inputs)
    key, val = encode(m, encoder_input, encoder_mask)
    decoder_input = prepare_decode_input(enc_inputs, dec_inputs, parser_inputs)
    cell = randn!(similar(key, size(key,1), size(key,3), 2))
    hiddens, _, srcalignments, _, tgtalignments = decode(m, decoder_input, (key,val), cell)
    Hy = size(hiddens,1)
    pg = PointerGenerator(Hy, vocabsize, 1e-20)
    @test size(pg.projection.w) == (vocabsize, Hy)
end

## remove here
#=
vocab_pad_idx = 1 # 0th index
vocabsize = 12202
enc_inputs, dec_inputs, generator_inputs , parser_inputs = prepare_batch_input(batch)
encoder_input, encoder_mask = prepare_encode_input(enc_inputs)
key, val = encode(m, encoder_input, encoder_mask)
decoder_input = prepare_decode_input(enc_inputs, dec_inputs, parser_inputs)
cell = randn!(similar(key, size(key,1), size(key,3), 2))
hiddens, _, srcalignments, _, tgtalignments = decode(m, decoder_input, (key,val), cell)
Hy = size(hiddens,1)
pg = PointerGenerator(Hy, vocabsize, 1e-20)
source_attentions = srcalignments
source_attention_maps = generator_inputs["copy_attention_maps"]
target_attentions = tgtalignments
target_attention_maps = generator_inputs["coref_attention_maps"]
=#


##  Compute a distribution over the target dictionary
#   extended by the dynamic dictionary implied by copying target nodes.
#
#   `hiddens`: decoder outputs
#   `source_attentions`: attention of each source node
#   `source_attention_maps`: a sparse indicator matrix
#       mapping each source node to its index in the dynamic vocabulary.
#   `target_attentions`: attention of each target node
#   `target_attention_maps`: a sparse indicator matrix
#        mapping each target node to its index in the dynamic vocabulary.
#   `invalid_indexes`: indexes which are not considered in prediction.
#

## TODO: Invalid indexes part
function (pg::PointerGenerator)(hiddens, source_attentions, source_attention_maps, target_attentions, target_attention_maps; invalid_indexes=Nothing,  vocab_pad_idx = 1)
    # hiddens: (Hy, B, Ty)
    # source_attentions: (Tx,Ty,B) 
    # source_attention_maps: (src_dynamic_vocabsize, Tx, B)
    # target_attentions: (Ty,Ty,B) 
    # target_attention_maps: (tgt_dynamic_vocabsize, Ty, B)
    # -> probs: (Ty, vocabsize + src_dynamic_vocabsize + tgt_dynamic_vocabsize, B)
    # -> predictions: (Ty, 1, B)
    # -> src_dynamic_vocabsize, tgt_dynamic_vocabsize

    Hy, B, Ty = size(hiddens) 
    source_dynamic_vocab_size = size(source_attention_maps, 1)
    target_dynamic_vocab_size = size(target_attention_maps, 1)
    hiddens = reshape(hiddens, (:,B *Ty)) # -> (H, B*Ty) 


    # Pointer probability.
    p = softmax(pg.linear_pointer(hiddens), dims=1)  # -> (3, B*Ty)
    p_copy_source = reshape(p[1, :], (B*Ty, 1))      # -> (B*Ty,1)
    p_copy_source = reshape(p_copy_source, (Ty,1,B)) # -> (Ty,1,B)

    p_copy_target = reshape(p[2, :], (B*Ty, 1))      # -> (B*Ty,1)
    p_copy_target = reshape(p_copy_target, (Ty,1,B)) # -> (Ty,1,B)

    p_generate = reshape(p[3, :], (B*Ty, 1))         # -> (B*Ty,1)
    p_generate = reshape(p_generate, (Ty,1,B))       # -> (Ty,1,B)



    # Pgen: Probability distribution over the vocabulary.
    scores = pg.projection(hiddens)                                  # -> (vocabsize, B*Ty)
    scores = permutedims(scores, [2,1])                              # -> (B*Ty, vocabsize)
    scores[:, vocab_pad_idx] = -Inf32
    vocab_probs = softmax(scores, dims=2)                            # todo is it okay?
    vocab_probs = reshape(vocab_probs, (Ty, vocabsize, B))           # -> (Ty, vocabsize, B)
    dummy = convert(_atype, zeros(1,size(vocab_probs,2),1))
    p_generate = p_generate .+ dummy                                 # -> (Ty, vocabsize, B)
    scaled_vocab_probs = vocab_probs .* p_generate                   # -> (Ty, vocabsize, B)

    
    # Probability distribution over the dynamic vocabulary.
    # TODO: make sure for target_node_i, its attention to target_node_j >= target_node_i
    # should be zero.

    # Psrc
    source_attentions = permutedims(source_attentions, [2,1,3])                                      # -> (Ty,Tx,B)
    dummy = convert(_atype, zeros(1,size(source_attentions,2),1))
    p_copy_source = p_copy_source .+ dummy                                                           # -> (Ty,Tx,B)
    scaled_source_attentions = source_attentions .* p_copy_source                                    # -> (Ty,Tx,B)
    source_attention_maps = permutedims(source_attention_maps, [2,1,3])                              # -> (Tx, srcdynamic_vocabsize, B)
    scaled_copy_source_probs = bmm(scaled_source_attentions, convert(_atype,source_attention_maps))  # -> (Ty, srcdynamic_vocabsize, B)

    
    # Ptgt
    target_attentions = permutedims(target_attentions, [2,1,3])                                      # -> (Ty,Ty,B) - a dummy line
    dummy = convert(_atype, zeros(1,size(target_attentions,2),1))
    p_copy_target = p_copy_target .+ dummy                                                           # -> (Ty,Ty,B)
    scaled_target_attentions = target_attentions .* p_copy_target                                    # -> (Ty,Ty,B)
    target_attention_maps = permutedims(target_attention_maps, [2,1,3])                              # -> (Ty, tgtdynamic_vocabsize, B)
    scaled_copy_target_probs = bmm(scaled_target_attentions, convert(_atype,target_attention_maps))  # -> (Ty, tgtdynamic_vocabsize, B)



    probs = cat(scaled_vocab_probs, scaled_copy_source_probs, scaled_copy_target_probs, dims=2)     # -> (Ty,vocabsize+dynamic_vocabsize,B)
    predictions = argmax(probs, dims=2) # -> (Ty,1,B)

    # TODO: check this part from original code again
    #   Set the probability of coref NA to 0.
    #  _probs = copy(probs)
    #  _probs[vocab_size + source_dynamic_vocab_size, :, :] = 0
    #_, predictions = _probs.max(2)

    # probs, predictions, source_dynamic_vocab_size, target_dynamic_vocab_size

    #loss = loss(probs, predictions)
    loss = 0.0
    return loss
end



function loss(probs, predictions, generate_targets,
                     source_copy_targets, source_dynamic_vocab_size,
                     target_copy_targets, target_dynamic_vocab_size,
                     coverage_records, copy_attentions)
    return 0
    ## TODO
    #=
        """
        Priority: target_copy > source_copy > generate

        :param probs: probability distribution,
            [batch_size, num_target_nodes, vocab_size + dynamic_vocab_size]
        :param predictions: [batch_size, num_target_nodes]
        :param generate_targets: target node index in the vocabulary,
            [batch_size, num_target_nodes]
        :param source_copy_targets:  target node index in the dynamic vocabulary,
            [batch_size, num_target_nodes]
        :param source_dynamic_vocab_size: int
        :param target_copy_targets:  target node index in the dynamic vocabulary,
            [batch_size, num_target_nodes]
        :param target_dynamic_vocab_size: int
        :param coverage_records: None or a tensor recording source-side coverages.
            [batch_size, num_target_nodes, num_source_nodes]
        :param copy_attentions: [batch_size, num_target_nodes, num_source_nodes]
        """

        non_pad_mask = generate_targets.ne(self.vocab_pad_idx)

        source_copy_mask = source_copy_targets.ne(1) & source_copy_targets.ne(0)  # 1 is the index for unknown words
        non_source_copy_mask = 1 - source_copy_mask

        target_copy_mask = target_copy_targets.ne(0)  # 0 is the index for coref NA
        non_target_copy_mask = 1 - target_copy_mask

        # [batch_size, num_target_nodes, 1]
        target_copy_targets_with_offset = target_copy_targets.unsqueeze(2) + self.vocab_size + source_dynamic_vocab_size
        # [batch_size, num_target_nodes]
        target_copy_target_probs = probs.gather(dim=2, index=target_copy_targets_with_offset).squeeze(2)
        target_copy_target_probs = target_copy_target_probs.mul(target_copy_mask.float())

        # [batch_size, num_target_nodes, 1]
        source_copy_targets_with_offset = source_copy_targets.unsqueeze(2) + self.vocab_size
        # [batch_size, num_target_nodes]
        source_copy_target_probs = probs.gather(dim=2, index=source_copy_targets_with_offset).squeeze(2)
        source_copy_target_probs = source_copy_target_probs.mul(non_target_copy_mask.float()).mul(source_copy_mask.float())

        # [batch_size, num_target_nodes]
        generate_target_probs = probs.gather(dim=2, index=generate_targets.unsqueeze(2)).squeeze(2)

        # Except copy-oov nodes, all other nodes should be copied.
        likelihood = target_copy_target_probs + source_copy_target_probs + \
                     generate_target_probs.mul(non_target_copy_mask.float()).mul(non_source_copy_mask.float())
        num_tokens = non_pad_mask.sum().item()

        if not self.force_copy:
            non_generate_oov_mask = generate_targets.ne(1)
            additional_generate_mask = (non_target_copy_mask & source_copy_mask & non_generate_oov_mask)
            likelihood = likelihood + generate_target_probs.mul(additional_generate_mask.float())
            num_tokens += additional_generate_mask.sum().item()

        # Add eps for numerical stability.
        likelihood = likelihood + self.eps

        coverage_loss = 0
        if coverage_records is not None:
            coverage_loss = torch.sum(
                torch.min(coverage_records, copy_attentions), 2).mul(non_pad_mask.float())

        # Drop pads.
        loss = -likelihood.log().mul(non_pad_mask.float()) + coverage_loss

        # Mask out copy targets for which copy does not happen.
        targets = target_copy_targets_with_offset.squeeze(2) * target_copy_mask.long() + \
                  source_copy_targets_with_offset.squeeze(2) * non_target_copy_mask.long() * source_copy_mask.long() + \
                  generate_targets * non_target_copy_mask.long() * non_source_copy_mask.long()
        targets = targets * non_pad_mask.long()

        pred_eq = predictions.eq(targets).mul(non_pad_mask)

        num_non_pad = non_pad_mask.sum().item()
        num_correct_pred = pred_eq.sum().item()

        num_target_copy = target_copy_mask.mul(non_pad_mask).sum().item()
        num_correct_target_copy = pred_eq.mul(target_copy_mask).sum().item()
        num_correct_target_point = predictions.ge(self.vocab_size + source_dynamic_vocab_size).\
            mul(target_copy_mask).mul(non_pad_mask).sum().item()

        num_source_copy = source_copy_mask.mul(non_target_copy_mask).mul(non_pad_mask).sum().item()
        num_correct_source_copy = pred_eq.mul(non_target_copy_mask).mul(source_copy_mask).sum().item()
        num_correct_source_point = predictions.ge(self.vocab_size).mul(predictions.lt(self.vocab_size + source_dynamic_vocab_size)).\
            mul(non_target_copy_mask).mul(source_copy_mask).mul(non_pad_mask).sum().item()

        self.metrics(loss.sum().item(), num_non_pad, num_correct_pred,
                     num_source_copy, num_correct_source_copy, num_correct_source_point,
                     num_target_copy, num_correct_target_copy, num_correct_target_point
                     )

        return dict(
            loss=loss.sum().div(float(num_tokens)),
            total_loss=loss.sum(),
            num_tokens=torch.tensor([float(num_tokens)]).type_as(loss),
            predictions=predictions
        )
    =#
end




