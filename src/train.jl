
function batchloss(batch)
    encoder_inputs, decoder_inputs, generator_inputs, parser_inputs = prepare_batch_input(batch)
    encoder_inputs = prepare_encode_input(
        encoder_inputs["bert_token"],
        encoder_inputs["token_subword_index"],
        encoder_inputs["token"],
        encoder_inputs["pos_tag"],
        encoder_inputs["must_copy_tag"],
        encoder_inputs["char"],
        encoder_inputs["mask"]
    )
    ## TODO....
    decode_for_training()
    generator()
    generator_compute_loss()
    graph_decode()
end
