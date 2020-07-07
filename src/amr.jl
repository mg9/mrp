include("vocab.jl")
using PyCall, JSON
import StatsBase: countmap

penman = pyimport("penman")
nx = pyimport("networkx")
re = pyimport("re")

# Disable inverting ':mod' relation.
penman.AMRCodec._inversions = delete!(penman.AMRCodec._inversions, "domain")
penman.AMRCodec._deinversions = delete!(penman.AMRCodec._deinversions, "mod")
amr_codec = penman.AMRCodec(indent=6)


mutable struct AMR
    id
    sentence
    graph
    tokens
    lemmas
    pos_tags
    ner_tags
    abstract_map
    misc
end


mutable struct AMRGraph
    triples
    top
    srctokens
    G
    variable_to_node
end


mutable struct AMRNode
    identifier
    attributes
    num_copies
    copy_of
end

function AMRNode(identifier; attributes=nothing, copy_of=nothing)
    if attributes==nothing
        attributes = []
    end
    _num_copies = 0
    AMRNode(identifier, attributes, _num_copies, copy_of)
end


function amrgraph_attributes(amrgraph; source=nothing, relation=nothing, target=nothing)
    # Refine attributes because there's a bug in penman.attributes()
    # See https://github.com/goodmami/penman/issues/29
    variables = amrgraph.variables()
    triples =  pycall(amrgraph.triples, Tuple)
    attrs = []
    for t in triples
        if !(t.target in variables) || t.relation =="instance"
            push!(attrs, t)
        end
    end
    filtered_attrs = []
    for a in attrs
        if (source==nothing || source == a.source) && (relation==nothing || relation==a.relation) && (target==nothing || target==a.target)
            push!(filtered_attrs,a)
        end
    end
    return filtered_attrs
end


function amrgraph_buildextras(amrgraph)
    G = nx.DiGraph()
    variable_to_node = Dict() 
    for v in amrgraph.variables()
        if !(v isa String)
            continue;
        end
        attributes = [(t[2], t[3]) for t in amrgraph_attributes(amrgraph, source=v)]
        node = AMRNode(v, attributes=attributes)
        G.add_node(node)
        variable_to_node[v] = node
    end
    amrgraph.variable_to_node = variable_to_node
    edge_set = []
    edges = pycall(amrgraph.edges, Tuple)
    for edge in edges
        if !(edge.source isa String) || !(edge.target isa String) ## TODO: why didn't the original code need target control?
            continue;
        end
        source = variable_to_node[edge.source]
        target = variable_to_node[edge.target]
        relation = edge.relation
        if relation == "instance"
            continue;
        end
        if source == target
            continue;
        end
        if edge.inverted
            source, target, relation = target, source, amr_codec.invert_relation(edge.relation)
        end
        if (source, target) in edge_set
            target = amrnode_copy(target)
        end
        push!(edge_set, (source, target))
        G.add_edge(source, target, label=relation)
    end
    amrgraph.G = G
end


function amrnode_copy(amrnode)
    attributes = nothing
    if amrnode.attributes != nothing
        attributes = amrnode.attributes[:]
    end
    amrnode.num_copies += 1
    identifier = string(amrnode.identifier,"_copy_",amrnode.num_copies)
    copy = AMRNode(identifier, attributes=attributes, copy_of=amrnode)
    return copy
end

function amrgraph_decode(raw_graph_string)
    _graph = amr_codec.decode(raw_graph_string)
 end


function amrgraph_setsrctokens(AMRGraph, sentence)
    if !isa(sentence, Array)
        sentence = split(sentence, " ")
    end
    return sentence
end


function amr_getsrctokens(amr)
    if amr.lemmas != nothing
        return amr.lemmas
    else
       return split(amr.sentence)
    end
end


function amrgraph_getlistnode(amrgraph, replace_copy=true)
    node_list = []
    visited = Dict()
    function dfs(node, relation, parent, visited)
        node_to_append = Any
        parent_to_append = Any
        if node.copy_of == nothing || !(replace_copy)
            node_to_append = node
        else
            node_to_append = node.copy_of
        end
        if parent.copy_of == nothing || !(replace_copy)
             parent_to_append = parent
        else
            parent_to_append = parent.copy_of
        end
       
        push!(node_list, (node_to_append,relation, parent_to_append))
       
        edges = pycall(amrgraph.G.edges, Dict{Tuple{AMRNode,AMRNode},Any})
        if node in amrgraph.G.nodes  && !haskey(visited, node) # TODO check here again!
            visited[node] = 1
            for ((s,t),r) in edges
                if s == node
                    dfs(t, r["label"], s, visited)
                end
            end
        end
    end
    dfs(amrgraph.variable_to_node[amrgraph.top], "root", amrgraph.variable_to_node[amrgraph.top], visited)
    return node_list
end



function amrgraph_getlistdata(amr; bos=nothing, eos=nothing, bert_tokenizer=nothing, max_tgt_length=nothing)
        node_list = amrgraph_getlistnode(amr.graph)
        tgt_tokens = []
        head_tags = []
        head_indices = []

        node_to_idx = Dict()
        visited = Dict()

        function update_info(node, relation, parent_node, token)
            push!(head_indices,  node_to_idx[parent_node][length(node_to_idx[parent_node])])  ## 1 +  node_to_idx[parent_node][length(node_to_idx[parent_node])]
            push!(head_tags, relation)
            push!(tgt_tokens, string(token))
        end

        for (node, relation, parent_node) in node_list
            if !haskey(node_to_idx, node)
                node_to_idx[node] = []
            end
            push!(node_to_idx[node], length(tgt_tokens)+1)
            
            instance = [attr[2] for attr in node.attributes if attr[1] == "instance"]
            @assert length(instance) == 1
            instance = instance[1]
            update_info(node, relation, parent_node, instance)
            if length(node.attributes) > 1 && !haskey(visited, node)
                for attr in node.attributes
                    if attr[1] != "instance"
                        update_info(node, attr[1], node, attr[2])
                    end
                end
            end
            visited[node] = 1
        end


        function trim_very_long_tgt_tokens(tgt_tokens, head_tags, head_indices, node_to_idx)
            tgt_tokens = tgt_tokens[1:min(length(tgt_tokens),max_tgt_length)]
            head_tags = head_tags[1:min(length(head_tags),max_tgt_length)]
            head_indices = head_indices[1:min(length(head_indices),max_tgt_length)]

            invalid_indices = []
            for (node, indices) in node_to_idx
                for index in indices
                    if index >= max_tgt_length
                        push!(invalid_indices, index)
                    end
                end
                for index in invalid_indices
                    deleteat!(indices, findall(x->x==index,indices))
                end
            end
            return tgt_tokens, head_tags, head_indices, node_to_idx
        end


        if max_tgt_length != nothing
            tgt_tokens, head_tags, head_indices, node_to_idx = trim_very_long_tgt_tokens(
                tgt_tokens, head_tags, head_indices, node_to_idx)
        end

        copy_offset = 0
        if bos !=nothing
            pushfirst!(tgt_tokens, bos)
            copy_offset += 1
        end
        if eos !=nothing
            push!(tgt_tokens,eos)
        end

        head_indices[node_to_idx[amr.graph.variable_to_node[amr.graph.top]][1]] = 0 

        # Target side Coreference(Copy)
        tgt_copy_indices = []
        for i in collect(1:length(tgt_tokens))
            push!(tgt_copy_indices, i)
        end

        for (node, indices) in node_to_idx
            if length(indices) > 1
                copy_idx = indices[1] + copy_offset
                for token_idx in indices[2:end]
                    tgt_copy_indices[token_idx + copy_offset] = copy_idx
                end
            end
        end

        tgt_copy_map=[]
        for (token_idx, copy_idx) in enumerate(tgt_copy_indices)
            push!(tgt_copy_map, (token_idx, copy_idx))
        end


        for (i, copy_index) in enumerate(tgt_copy_indices)
            # Set the coreferred target to 0 if no coref is available.
            if i == copy_index
                tgt_copy_indices[i] = 0
            end
        end


        tgt_token_counter = countmap(tgt_tokens)
        tgt_copy_mask = repeat([0],length(tgt_tokens))

        for (i, token) in enumerate(tgt_tokens)
            if tgt_token_counter[token] > 1
                tgt_copy_mask[i] = 1
            end
        end


        function add_source_side_tags_to_target_side(_src_tokens, _src_tags)
            @assert length(_src_tags) == length(_src_tokens)
            tag_counter = Dict()
            for (src_token, src_tag) in zip(_src_tokens, _src_tags)
                if !haskey(tag_counter, src_token)
                    tag_counter[src_token] = Dict()
                end
                if !haskey(tag_counter[src_token], src_tag)
                    tag_counter[src_token][src_tag] = 0
                end
                tag_counter[src_token][src_tag] += 1
            end

            tag_lut = Dict(DEFAULT_OOV_TOKEN => DEFAULT_OOV_TOKEN,
                       DEFAULT_PADDING_TOKEN => DEFAULT_OOV_TOKEN)
            for src_token in _src_tokens
                maxval = 0
                for (k,v) in tag_counter[src_token]
                    if v>maxval
                        maxval=v
                    end
                end
                for (tag,v) in tag_counter[src_token]
                    if v==maxval
                        tag_lut[src_token] = tag
                    end
                end
            end
            tgt_tags = []
            for tgt_token in tgt_tokens
                sim_token = find_similar_token(tgt_token, _src_tokens)
                if sim_token != nothing
                    index = findfirst(x->x==sim_token, _src_tokens)
                    tag = _src_tags[index]
                else
                    tag = DEFAULT_OOV_TOKEN
                end
                push!(tgt_tags, tag)
            end
            return tgt_tags, tag_lut
        end

        # Source Copy
        src_tokens = amr.graph.srctokens
        src_token_ids = nothing
        src_token_subword_index = nothing
        src_pos_tags = amr.pos_tags
        srccopyvocab = Vocab(src_tokens)
        src_copy_indices = vocab_indexsequence(srccopyvocab, tgt_tokens)
        src_copy_map = vocab_getcopymap(srccopyvocab, src_tokens)
        tgt_pos_tags, pos_tag_lut = add_source_side_tags_to_target_side(src_tokens, src_pos_tags)
        

        #if bert_tokenizer != nothing
        #    src_token_ids, src_token_subword_index = bert_tokenizer.tokenize(src_tokens, True)
        #end


        src_must_copy_tags = []
        for t in src_tokens
            if is_abstract_token(t)
                push!(src_must_copy_tags,1)
            else
                push!(src_must_copy_tags,0)
            end
        end
        
        src_copy_invalid_ids = []
        tmptokens =[]
        for t in src_tokens
            if is_english_punct(t)
                push!(tmptokens,t)
            end
        end
        src_copy_invalid_ids = vocab_indexsequence(srccopyvocab, tmptokens)

        return Dict(
            "tgt_tokens" => tgt_tokens,
            "tgt_pos_tags" => tgt_pos_tags,
            "tgt_copy_indices" => tgt_copy_indices,
            "tgt_copy_map" => tgt_copy_map,
            "tgt_copy_mask" => tgt_copy_mask,
            "src_tokens" => src_tokens,
            "src_token_ids" => src_token_ids,
            "src_token_subword_index" => src_token_subword_index,
            "src_must_copy_tags" => src_must_copy_tags,
            "src_pos_tags" => src_pos_tags,
            "src_copy_vocab"  => srccopyvocab,
            "src_copy_indices" => src_copy_indices,
            "src_copy_map" => src_copy_map,
            "pos_tag_lut" => pos_tag_lut,
            "head_tags" => head_tags,
            "head_indices" => head_indices,
            "src_copy_invalid_ids" => src_copy_invalid_ids
        )
end


function is_abstract_token(token)
    return occursin(r"^([A-Z]+_)+\d+$", token) || occursin(r"^\d0*$", token) 
end

function is_english_punct(c)
    return occursin(r"^[,.?!:;\"\"-(){}\[\]]$", c)
end

function find_similar_token(token, tokens)
    token = replace(token, r"-\d\d$" => "")
    for (i, t) in enumerate(tokens)
        if token == t
            return tokens[i]
        end
    end
    return nothing
end




