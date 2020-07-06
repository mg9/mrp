DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@" #Unk token

mutable struct Vocab
    srctokens
    padtoken
    unktoken
    token_to_idx
    idx_to_token
    vocabsize
end


function Vocab(srctokens)
    token_to_idx = Dict()
    token_to_idx[DEFAULT_PADDING_TOKEN] = 0
    token_to_idx[DEFAULT_OOV_TOKEN] = 1
    idx_to_token = Dict()
    idx_to_token[0] = DEFAULT_PADDING_TOKEN
    idx_to_token[1] = DEFAULT_OOV_TOKEN
    vocabsize = 2
    if isa(srctokens, Array)
        for token in srctokens
            if !haskey(token_to_idx, token)
                token_to_idx[token] = vocabsize
                idx_to_token[vocabsize] = token
                vocabsize += 1
            end
        end
    end
    return Vocab(srctokens, DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN, token_to_idx, idx_to_token, vocabsize)
end


function vocab_addtokens(vocab::Vocab, srctokens)
    for token in srctokens
        if !haskey(vocab.token_to_idx, token)
            vocab.token_to_idx[token] = vocab.vocabsize
            vocab.idx_to_token[vocab.vocabsize] = token
            vocab.vocabsize += 1
        end
    end
end


function vocab_indexsequence(vocab::Vocab, list_tokens)
    indexseq = []
    for token in list_tokens
        if haskey(vocab.token_to_idx, token)
            push!(indexseq, vocab.token_to_idx[token])
        else
            push!(indexseq, vocab.token_to_idx[DEFAULT_OOV_TOKEN])
        end
    end
    return indexseq
end


function vocab_getcopymap(vocab::Vocab, src_tokens)
    copymap = []
    srcindices = pushfirst!(vocab_indexsequence(vocab, src_tokens), vocab.token_to_idx[DEFAULT_OOV_TOKEN])

    for (src_idx, src_token_idx) in enumerate(srcindices)
        push!(copymap, (src_idx, src_token_idx))
    end
    return copymap
end



