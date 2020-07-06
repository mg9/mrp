include("amr.jl")
include("vocab.jl")
import Base: length, iterate
using Random


START_SYMBOL = bos = "@start@"
END_SYMBOL = eos =  "@end@"

ENCODER_TOKEN_VOCAB_SIZE = 3406 #18002
DECODER_TOKEN_VOCAB_SIZE = 2908 #12202
encodervocab_pad_idx =ENCODER_TOKEN_VOCAB_SIZE          ## DO not confuse about vocab_pad_idx, only srctokens and tgttokens pads have been changed for embedding layers
decodervocab_pad_idx =DECODER_TOKEN_VOCAB_SIZE

mutable struct AMRDataset
    amrinstances
    ninstances
    batchsize
    shuffled
    word_splitter
    number_bert_ids
    number_bert_oov_ids
    number_non_oov_pos_tags
    number_pos_tags
    srcvocab
    srccharactervocab
    srcpostagvocab
    srcmustcopytagsvocab
    srccopyindicesvocab
    tgtvocab
    tgtcharactervocab
    tgtpostagvocab
    tgtcopymaskvocab
    tgtcopyindicesvocab
    headtagsvocab
    headindicesvocab
end


function AMRDataset(amrinstances, batchsize; shuffled=true, word_splitter=nothing)
    word_splitter= nothing #TODO: nothing until BERT time 
    _number_bert_ids = 0
    _number_bert_oov_ids = 0
    _number_non_oov_pos_tags = 0
    _number_pos_tags = 0
    srcvocab = Vocab([])
    srccharactervocab = Vocab([])
    srcpostagvocab = Vocab([])
    srcmustcopytagsvocab = Vocab([])
    srccopyindicesvocab = Vocab([])
    tgtvocab = Vocab([])
    tgtcharactervocab = Vocab([])
    tgtpostagvocab = Vocab([])
    tgtcopymaskvocab = Vocab([])
    tgtcopyindicesvocab = Vocab([])
    headtagsvocab = Vocab([])
    headindicesvocab = Vocab([])
    AMRDataset(amrinstances, length(amrinstances), batchsize,shuffled, word_splitter,  _number_bert_ids, _number_bert_oov_ids, _number_non_oov_pos_tags,_number_pos_tags,
    srcvocab, srccharactervocab, srcpostagvocab, srcmustcopytagsvocab, srccopyindicesvocab, tgtvocab, tgtcharactervocab, tgtpostagvocab, tgtcopymaskvocab, tgtcopyindicesvocab,
    headtagsvocab, headindicesvocab)
end





function make_instance(d::AMRDataset, amr, evaluation=false)
    amrgraph_buildextras(amr.graph)
    max_tgt_length = Any
    if evaluation; max_tgt_length = nothing
    else; max_tgt_length = 60; end
    list_data = amrgraph_getlistdata(amr, bos=START_SYMBOL, eos=END_SYMBOL, bert_tokenizer=d.word_splitter, max_tgt_length=max_tgt_length)

    #TODO: BERT thing for src_token_ids & src_token_subword_index

    srccharacters= []
    for token in list_data["src_tokens"]
        for c in token; push!(srccharacters,c); end
    end

    tgtcharacters= []
    for token in list_data["tgt_tokens"]
       for c in token; push!(tgtcharacters,c); end
    end

    vocab_addtokens(d.srcvocab, list_data["src_tokens"])
    vocab_addtokens(d.tgtvocab, list_data["tgt_tokens"])
    vocab_addtokens(d.srccharactervocab, srccharacters)
    vocab_addtokens(d.tgtcharactervocab, tgtcharacters)
    vocab_addtokens(d.srcpostagvocab, list_data["src_pos_tags"])
    vocab_addtokens(d.tgtpostagvocab, list_data["tgt_pos_tags"])
    vocab_addtokens(d.headtagsvocab, list_data["head_tags"]) 

    d.number_pos_tags += length(list_data["tgt_pos_tags"])
    for tag in list_data["tgt_pos_tags"]
        if tag != DEFAULT_OOV_TOKEN; d.number_non_oov_pos_tags +=1; end
    end


    function findsrccopymax()
        maxval = 0
         for (i,t) in list_data["src_copy_map"]
           if t>maxval
               maxval=t
           end
        end
        return maxval
    end
    maxval= findsrccopymax()


    srccopymap = zeros(length(list_data["src_copy_map"]), length(list_data["src_copy_map"]))
    for (k,v) in list_data["src_copy_map"]; srccopymap[k,v] = 1; end
    srccopymap = hcat(zeros(size(srccopymap,1),1), srccopymap)
    srccopymap = vcat(zeros(1,size(srccopymap,2)), srccopymap)
    tgtcopymap = zeros(length(list_data["tgt_copy_map"]), length(list_data["tgt_copy_map"]))
    for (k,v) in list_data["tgt_copy_map"]; tgtcopymap[k,v] = 1; end
   
    # TODO: Look again sequenceLabelField in original code
    vocab_addtokens(d.srcmustcopytagsvocab, list_data["src_must_copy_tags"]) 
    vocab_addtokens(d.srccopyindicesvocab, list_data["src_copy_indices"]) 
    vocab_addtokens(d.tgtcopyindicesvocab, list_data["tgt_copy_indices"]) ## coref_tags
    vocab_addtokens(d.tgtcopymaskvocab, list_data["tgt_copy_mask"]) ## coref_mask_tags
    vocab_addtokens(d.headindicesvocab, list_data["head_indices"]) 

    instance = Dict()
    instance["head_tags"] = vocab_indexsequence(d.headtagsvocab, list_data["head_tags"])
    instance["encoder_tokens"] = vocab_indexsequence(d.srcvocab, list_data["src_tokens"])
    instance["decoder_characters"] = vocab_indexsequence(d.tgtcharactervocab, tgtcharacters)
    instance["tgt_pos_tags"] = vocab_indexsequence(d.tgtpostagvocab, list_data["tgt_pos_tags"])
    instance["encoder_characters"] = vocab_indexsequence(d.srccharactervocab, srccharacters)
    instance["decoder_tokens"] = vocab_indexsequence(d.tgtvocab, list_data["tgt_tokens"])
    instance["src_pos_tags"] = vocab_indexsequence(d.srcpostagvocab, list_data["src_pos_tags"])
    instance["tgt_copy_mask"] = list_data["tgt_copy_mask"] ## TODO: is that correct??
    instance["tgt_copy_indices"] = list_data["tgt_copy_indices"] ## TODO: is that correct??
    instance["src_must_copy_tags"] = list_data["src_must_copy_tags"] ## TODO: is that correct??
    instance["src_copy_indices"] = list_data["src_copy_indices"] ## TODO: is that correct??
    instance["head_indices"] = list_data["head_indices"] ## TODO: is that correct??
    instance["src_token_ids"] = nothing # BERT
    instance["src_token_subword_index"] = nothing #BERT
    instance["tgt_copy_map"]  = tgtcopymap
    instance["src_copy_map"]  = srccopymap

    return instance
end




function iterate(d::AMRDataset, state=ifelse(d.shuffled, randperm(d.ninstances), collect(1:d.ninstances)))
    new_state = state
    new_state_len = length(new_state) 
    max_ind = 0
    if new_state_len == 0 
        return nothing 
    end
    max_ind = min(new_state_len, d.batchsize)    
    amrs_raw = d.amrinstances[new_state[1:max_ind]]

    head_tags= []
    tgt_copy_mask = []
    encoder_tokens=[]
    tgt_copy_map = []
    src_token_ids = []
    src_token_subword_index = []
    decoder_characters = []
    tgt_pos_tags = []
    encoder_characters = []
    tgt_copy_indices = []
    src_must_copy_tags = []
    decoder_tokens = []
    head_indices = []
    src_pos_tags = []
    src_copy_indices = []
    src_copy_map = []

    for (i,amr) in enumerate(amrs_raw)
        println("i: $i")
        instance = make_instance(d, amr)
        push!(head_tags, instance["head_tags"])
        push!(tgt_copy_mask, instance["tgt_copy_mask"])
        push!(encoder_tokens, instance["encoder_tokens"])
        push!(tgt_copy_map, instance["tgt_copy_map"])
        push!(src_token_ids, instance["src_token_ids"])
        push!(src_token_subword_index, instance["src_token_subword_index"])
        push!(decoder_characters, instance["decoder_characters"])
        push!(tgt_pos_tags, instance["tgt_pos_tags"])
        push!(encoder_characters, instance["encoder_characters"])
        push!(tgt_copy_indices, instance["tgt_copy_indices"])
        push!(src_must_copy_tags, instance["src_must_copy_tags"])
        push!(decoder_tokens, instance["decoder_tokens"])
        push!(head_indices, instance["head_indices"])
        push!(src_pos_tags, instance["src_pos_tags"])
        push!(src_copy_indices, instance["src_copy_indices"])
        push!(src_copy_map, instance["src_copy_map"])
    end

    function pad(x, maxlength)
        append!(x, zeros(maxlength-length(x)))
     end

    function pad2d(x, maxlength)
        x = hcat(x, zeros(size(x,1),maxlength-size(x,1)))
        x = vcat(x, zeros(maxlength-size(x,1),size(x,2)))
     end

    maxheadtags= maximum(length.(head_tags))
    pad.(head_tags,maxheadtags)
    head_tags = hcat(head_tags...)

    maxencodertokens= maximum(length.(encoder_tokens))
    pad.(encoder_tokens,maxencodertokens)
    encoder_tokens = hcat(encoder_tokens...)

    maxtgtcopymap= Integer(sqrt(maximum(length.(tgt_copy_map))))
    tgt_copy_map = pad2d.(tgt_copy_map, maxtgtcopymap)
    tgt_copy_map = cat(tgt_copy_map..., dims=3)


    maxdecodercharacters= maximum(length.(decoder_characters))
    pad.(decoder_characters,maxdecodercharacters)
    decoder_characters = hcat(decoder_characters...)

    maxtgtpostags = maximum(length.(tgt_pos_tags))
    pad.(tgt_pos_tags, maxtgtpostags)
    tgt_pos_tags = hcat(tgt_pos_tags...)

    maxencodercharacters= maximum(length.(encoder_characters))
    pad.(encoder_characters,maxencodercharacters)
    encoder_characters = hcat(encoder_characters...)

    maxtgtcopyindices= maximum(length.(tgt_copy_indices))
    pad.(tgt_copy_indices,maxtgtcopyindices)
    tgt_copy_indices = hcat(tgt_copy_indices...)

    maxdecodertokens= maximum(length.(decoder_tokens))
    pad.(decoder_tokens,maxdecodertokens)
    decoder_tokens = hcat(decoder_tokens...)

    
    maxheadindices = maximum(length.(head_indices))
    pad.(head_indices, maxheadindices)
    head_indices = hcat(head_indices...)


    maxsrcpostags = maximum(length.(src_pos_tags))
    pad.(src_pos_tags, maxsrcpostags)
    src_pos_tags = hcat(src_pos_tags...)


    maxsrccopyindices = maximum(length.(src_copy_indices))
    pad.(src_copy_indices, maxsrccopyindices)
    src_copy_indices = hcat(src_copy_indices...)

    maxsrccopymap= Integer(sqrt(maximum(length.(src_copy_map))))
    src_copy_map = pad2d.(src_copy_map, maxsrccopymap)
    src_copy_map = cat(src_copy_map..., dims=3)

    #tgt_copy_mask = vcat(tgt_copy_mask...)
    #src_token_ids = vcat(src_token_ids...)
    #src_token_subword_index = vcat(src_token_subword_index...)
    #encoder_characters = vcat(encoder_characters...)
    #src_must_copy_tags = vcat(src_must_copy_tags...)

   
    encoder_tokens = Integer.(encoder_tokens)
    decoder_tokens = Integer.(decoder_tokens)
    src_copy_map = Integer.(src_copy_map)
    tgt_copy_map = Integer.(tgt_copy_map)
    src_copy_indices = Integer.(src_copy_indices)
    tgt_copy_indices = Integer.(tgt_copy_indices)
    head_indices = Integer.(head_indices)
    head_tags = Integer.(head_tags)


    encoder_tokens[encoder_tokens.==0] .= encodervocab_pad_idx                           # replace pad idx 0 with the latest index of array
    decoder_tokens[decoder_tokens.==0] .= decodervocab_pad_idx                           # replace pad idx 0 with the latest index of array


    srctokens = encoder_tokens'                                                # -> (B,Tx)
    tgttokens = decoder_tokens[1:end-1,:]'                                     # -> (B,Ty)
    srcattentionmaps = permutedims(src_copy_map, [2,1,3])[2:end-1,:,:]         # -> (Tx,Tx+2, B)
    tgtattentionmaps = permutedims(tgt_copy_map, [2,1,3])[2:end,:,:]           # -> (Ty,Ty+1, B)
    generatetargets = decoder_tokens[2:end,:]       # -> (Ty,B) ## DO not confuse vocab_pad_idx thing, for embeddings src and tgttokens pads have been changed
    srccopytargets = src_copy_indices[2:end,:]      # -> (Ty,B)
    tgtcopytargets = tgt_copy_indices[2:end,:]      # -> (Ty,B)
    parsermask = copy(tgttokens[:,2:end])
    parsermask[parsermask.==decodervocab_pad_idx] .= 1
    parsermask[parsermask.!=decodervocab_pad_idx] .= 0
    edgeheads = head_indices[1:end,:]'
    edgelabels = head_tags[1:end, :]'


    deleteat!(new_state, 1:d.batchsize)
    return  ((srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, parsermask, edgeheads, edgelabels) , new_state)
end


function length(d::AMRDataset)
    d, r = divrem(d.ninstances, d.batchsize)
    return r == 0 ? d : d+1
end






function read_preprocessed_files(file_path)
    amrs = []
    open(file_path) do file
        id =sentence = tokens = lemmas = pos_tags = ner_tags = abstract_map =Any
        graph_lines = []
        misc_lines = []
        for line in eachline(file)
            line = rstrip(line)
            if line == ""
                if length(graph_lines) != 0
                    amr = AMR(id, sentence, Any, tokens, lemmas, pos_tags, ner_tags, abstract_map, misc_lines)
                    amrgraph = amrgraph_decode(join(graph_lines," "))
                    amrgraph.srctokens = amrgraph_setsrctokens(amrgraph, amr_getsrctokens(amr))
                    amr.graph = amrgraph
                    push!(amrs, amr)
                end
                graph_lines = []
                misc_lines = []
            elseif startswith(line, "# ::")
                if startswith(line, "# ::id ")
                    id = line[length("# ::id "):end] #check here
                elseif startswith(line, "# ::snt ")
                    sentence = line[length("# ::snt "):end]
                elseif startswith(line, "# ::tokens ")
                    tokens  = JSON.parse(line[length("# ::tokens "):end])
                elseif startswith(line, "# ::lemmas ")
                    lemmas = JSON.parse(line[length("# ::lemmas "):end])
                elseif startswith(line, "# ::pos_tags ")
                    pos_tags = JSON.parse(line[length("# ::pos_tags "):end])
                elseif startswith(line, "# ::ner_tags ")
                    ner_tags = JSON.parse(line[length("# ::ner_tags "):end])
                elseif startswith(line, "# ::abstract_map ")
                    abstract_map = JSON.parse(line[length("# ::abstract_map "):end])
                else
                    push!(misc_lines, line)
                end
            else
                push!(graph_lines, line)
            end
        end
        if length(graph_lines) !=0
            amr = AMR(id, sentence, Any, tokens, lemmas, pos_tags, ner_tags, abstract_map, misc_lines)
            amrgraph = amrgraph_decode(join(graph_lines," "))
            amrgraph.srctokens = amrgraph_setsrctokens(amrgraph, amr_getsrctokens(amr))
            amr.graph = amrgraph
            push!(amrs, amr)
        end
        return amrs
    end
end 

