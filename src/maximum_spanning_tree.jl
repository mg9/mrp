import Base: length, iterate
using YAML, Knet, IterTools, Random
using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random,  IterTools

# Maximum Spanning Tree extraction using Chu Liu Edmond's Algorithm.
# Adapted from: https://github.com/sheng-z/stog/blob/master/stog/algorithms/maximum_spanning_tree.py

function decode_mst(energy, N, has_labels=true)
    # energy: a tensor with shape (Ty, Ty, num_labels) if has_labels is true
    # (Ty, Ty) otherwise. Contains the score of each edge
    # Since the graph is directed: energy[i, j, k] should denote the score for the edge from node i to node j, when the edge is of type k
    # N: The length of this sequence. This is useful for when the input is from a batch operation and has been padded
    # has_labels: default value is true, whether 
    # Returns:
    #         The maximum scoring arborescence in the graph
    #         - Heads: An array of size length, where the element at index i is the parent of the ith node
    #         - Head_labels: An array of size length, where the label of the ith node is at index i.
    #                        If no labels, an array of 1s
    @assert (!has_labels && length(size(energy))==2) || (has_labels && length(size(energy))==3) 
    @assert size(energy)[1]==size(energy)[2]
    @assert size(energy, 1)<=N
    
    max_length = size(energy, 2)
    
    # Clip the energy matrix to size NxN
    # Extract the label of each edge
    energy = energy[1:N, 1:N, :]
    label_matrix = (x->x[3]).(argmax(energy, dims=3))
    # Convert to two dimensional matrix, only take the highest possible score for each edge among
    # the different scores for different labels
    energy = energy[argmax(energy, dims=3)]
    energy = reshape(energy, N, N)
    
    
    # get original score matrix
    original_score_matrix = energy
    # initialize a copy of the score matrix
    score_matrix = copy(original_score_matrix)
    
    old_input = zeros(Int, N, N)
    old_output = zeros(Int, N, N)
    current_nodes = [true for i in 1:N]
    representatives = []
    
    for node1 in 1:N
        original_score_matrix[node1, node1] = 0
        score_matrix[node1, node1] = 0
        push!(representatives, Set(node1))
        
        for node2 in node1+1:N
            old_input[node1, node2] = node1
            old_output[node1, node2] = node2
            
            old_input[node2, node1] = node2
            old_output[node2, node1] = node1
        end
    end
    
    final_edges = Dict{Int, Int}()
    
    # Algorithm operates in place
    chu_liu_edmonds!(N, score_matrix, current_nodes, final_edges, old_input, old_output, representatives)
    
    heads = zeros(Int, N)
    head_type = ones(Int, N)
    
    for (child,parent) in final_edges
        heads[child]=parent
        if parent!=-1
            head_type[child]=label_matrix[parent, child]
        end
    end
    
    heads, head_type
end

function chu_liu_edmonds!(N, score_matrix, current_nodes, final_edges, old_input, old_output, representatives)
    # Apply the chu-liu-edmonds algorithm recursively to a graph with edge scores
    # according to score_matrix. Inplace function, modifies input variables.
    # TODO: There is an iterative approach, but for now this should suffice
    # Node '1' is always the root node
    parents = [-1]
    for node1 in 2:N
        # Init the parent of each node to be the root
        push!(parents, 1)
        if current_nodes[node1]
        # If the node is a representative, find the max outgoing edge to other
        # non-root representative, and update its parent
            max_score = score_matrix[1, node1]
            for node2 in 2:N
                if node1 == node2 || !current_nodes[node2]
                    continue
                end
                
                new_score = score_matrix[node2, node1]
                if new_score > max_score
                    max_score = new_score
                    parents[node1] = node2
                end
            end
        end
    end
    # Check if there is a cycle in the current solution
    has_cycle, cycle = find_cycle(parents, N, current_nodes)

    # if no cycles, find all edges and return
    if !has_cycle
        final_edges[1] = -1
        for node in 2:N
            if !current_nodes[node]
                continue
            end

            # Translate the indices back to the original indices
            parent = old_input[parents[node], node]
            child = old_output[parents[node], node]
            final_edges[child] = parent
        end
        return
    end

    # Otherwise, remove an edge to get rid of the cycle
    # Contraction stage
    cycle_score = 0
    # Find the score of the cycle
    for node in cycle
        cycle_score += score_matrix[parents[node], node]
    end

    # For each node in the graph, find the maximum incoming weight
    # and outgoing edge into the cycle
    cycle_rep = cycle[1]
    for node in 1:N
        # We want to look at nodes not in the cycle
        if !current_nodes[node] || node in cycle
            continue
        end

        in_edge_score = -Inf
        in_edge = -1
        out_edge_score = -Inf
        out_edge = -1

        for node_in_cycle in cycle
            if score_matrix[node_in_cycle, node] > in_edge_score
                in_edge_score = score_matrix[node_in_cycle, node]
                in_edge = node_in_cycle
            end

            # Add the new edge score to the cycle score
            # Subtract the edge we're considering removing

            score = (cycle_score + 
                     score_matrix[node, node_in_cycle] -
                     score_matrix[parents[node_in_cycle], node_in_cycle])

            if score > out_edge_score
                out_edge_score = score
                out_edge = node_in_cycle    
            end
        end

        score_matrix[cycle_rep, node] = in_edge_score
        old_input[cycle_rep, node] = old_input[in_edge, node]
        old_output[cycle_rep, node] = old_output[in_edge, node]

        score_matrix[node, cycle_rep] = out_edge_score
        old_input[node, cycle_rep] = old_input[node, out_edge]
        old_output[node, cycle_rep] = old_output[node, out_edge]
    end

    # For the next recursive iteration, we want to collapse this cycle
    # and treat is as a single node. This first node is arbitrarily chosen.
    # All other nodes will not be considered in the next iteration. We
    # also need to keep track of which representatives we are considering
    # this iteration because we need them below to check if we're done
    considered_reps = []
    for (i, node_in_cycle) in enumerate(cycle)
        push!(considered_reps, Set())
        # Choose the first as the representative, disable the rest
        if i>1
            current_nodes[node_in_cycle]=false
        end

        for node in representatives[node_in_cycle]
            push!(considered_reps[i], node)
            if i>1
                push!(representatives[cycle_rep], node)
            end
        end
    end

    chu_liu_edmonds!(N, score_matrix, current_nodes, final_edges, old_input, old_output, representatives)

    # Expansion stage
    # Check each node in cycle, if one of its reps is a key
    # in the final edges, it is the one we need.
    # The node we are looking for is the node which
    # is the child of the incoming edge to the cycle
    found = false
    key_node = -1
    for (i, node) in enumerate(cycle)
        for cycle_rep in considered_reps[i]
            if in(cycle_rep,keys(final_edges))
                key_node = node
                found = true
                break
            end
        end
        if found
            break
        end
    end


    # break the cycle
    previous = parents[key_node]
    while previous != key_node
        child = old_output[parents[previous], previous]
        parent = old_input[parents[previous], previous]
        final_edges[child] = parent
        previous = parents[previous]
    end
end             

function find_cycle(parents, N, current_nodes)
    # Returns:
    #         has_cycle: Whether the graph has at least one cycle
    #         cycle: A list of nodes that form a cycle in the graph
    
    # Visited array
    visited = [false for i in 1:N]
    visited[1] = true
    cycle = Set()
    has_cycle = false
    for i in 2:N
        if has_cycle
            break
        end
        
        # Avoid redoing nodes we are not considering
        # or nodes we have visited and processed before
        if visited[i] || !current_nodes[i]
            continue
        end
        # Initialize a new possible cycle
        this_cycle = Set()
        push!(this_cycle, i)
        visited[i] = true
        has_cycle = true
        next_node = i
        while !in(parents[next_node], this_cycle)
            next_node = parents[next_node]
            # If we see a node we've already processed,
            # we can stop since the node we are processing
            # would have been in that cycle if it existed
            if visited[next_node]
                has_cycle = false
                break
            end
            visited[next_node] = true
            push!(this_cycle, next_node)
        end
        
        if has_cycle
            origin = next_node
            push!(cycle, origin)
            next_node = parents[origin]
            while next_node != origin
                push!(cycle, next_node)
                next_node = parents[next_node]
            end
            # Break the for loop and return, we already found a cycle
            break
        end
    end
    # Sorted is purely to mimic the Sets in python
    # Not needed, but for consistency I put it here
    has_cycle, sort([el for el in cycle])
end
        

function decode_mst_with_corefs(energy, coreference, N, has_labels=true)
    # energy: a tensor with shape (Ty, Ty, num_labels) if has_labels is true
    # (Ty, Ty) otherwise. Contains the score of each edge
    # Since the graph is directed: energy[i, j, k] should denote the score for the edge
    #                              from node i to node j, when the edge is of type k
    # coreferences: A list mapping a node to its first precedent
    # N: The length of this sequence. This is useful for when the 
    # input is from a batch operation and has been padded
    # has_labels: default value is true, whether 
    # Returns:
    #         The maximum scoring arborescence in the graph
    #         - Heads: An array of size length, where the element at index i is the parent of the ith node
    #         - Head_labels: An array of size length, where the label of the ith node is at index i.
    #                        If no labels, an array of 1s
    @assert (!has_labels && length(size(energy))==2) || (has_labels && length(size(energy))==3) 
    @assert size(energy)[1]==size(energy)[2]
    @assert size(energy, 1)<=N
    
    max_length = size(energy, 2)
    
    # Clip the energy matrix to size NxN
    # Extract the label of each edge
    energy = energy[1:N, 1:N, :]
    label_matrix = (x->x[3]).(argmax(energy, dims=3))
    # Convert to two dimensional matrix, only take the highest possible score for each edge among
    # the different scores for different labels
    energy = energy[argmax(energy, dims=3)]
    energy = reshape(energy, N, N)
    
    
    # get original score matrix
    original_score_matrix = energy
    # initialize a copy of the score matrix
    score_matrix = copy(original_score_matrix)
    
    old_input = zeros(Int, N, N)
    old_output = zeros(Int, N, N)
    current_nodes = [true for i in 1:N]
    representatives = []
    
    for node1 in 1:N
        original_score_matrix[node1, node1] = 0
        score_matrix[node1, node1] = 0
        push!(representatives, Set(node1))
        
        for node2 in node1+1:N
            old_input[node1, node2] = node1
            old_output[node1, node2] = node2
            
            old_input[node2, node1] = node2
            old_output[node2, node1] = node1
        end
    end
    
    final_edges = Dict{Int, Int}()
    
    # Algorithm operates in place
    adapted_chu_liu_edmonds!(N, score_matrix, coreference, current_nodes, final_edges, old_input, old_output, representatives)

    
    # Modify edges which are invalid according to coreference
    validate!(final_edges, N, original_score_matrix, coreference)

    
    heads = zeros(Int, N)
    head_type = ones(Int, N)
    
    for (child,parent) in final_edges
        heads[child]=parent
        if parent!=-1
            head_type[child]=label_matrix[parent, child]
        end
    end
    
    heads, head_type
end

function adapted_chu_liu_edmonds!(N, score_matrix, coreference, current_nodes, final_edges, old_input, old_output, representatives)
    # Apply the chu-liu-edmonds algorithm recursively to a graph with edge scores
    # according to score_matrix. Inplace function, modifies input variables.
    # TODO: There is an iterative approach, but for now this should suffice
    # Node '1' is always the root node
    parents = [-1]
    for node1 in 2:N
        # Init the parent of each node to be the root
        push!(parents, 1)
        if current_nodes[node1]
        # If the node is a representative, find the max outgoing edge to other
        # non-root representative, and update its parent
            max_score = score_matrix[1, node1]
            for node2 in 2:N
                if node1 == node2 || !current_nodes[node2]
                    continue
                end
                
                # Exclude edges formed by two coreferred nodes
                _parent = old_input[node1, node2]
                _child = old_output[node1, node2]
                if _parent == 0 || _child == 0 || coreference[_child] == coreference[_parent]
                    continue
                end
                
                
                new_score = score_matrix[node2, node1]
                if new_score > max_score
                    max_score = new_score
                    parents[node1] = node2
                end
            end
        end
    end
    # Check if there is a cycle in the current solution
    has_cycle, cycle = find_cycle(parents, N, current_nodes)
    
    # if no cycles, find all edges and return
    if !has_cycle
        final_edges[1] = -1
        for node in 2:N
            if !current_nodes[node]
                continue
            end

            # Translate the indices back to the original indices
            parent = old_input[parents[node], node]
            child = old_output[parents[node], node]
            final_edges[child] = parent
        end
        return
    end

    # Otherwise, remove an edge to get rid of the cycle
    # Contraction stage
    cycle_score = 0
    # Find the score of the cycle
    for node in cycle
        cycle_score += score_matrix[parents[node], node]
    end

    # For each node in the graph, find the maximum incoming weight
    # and outgoing edge into the cycle
    cycle_rep = cycle[1]
    for node in 1:N
        # We want to look at nodes not in the cycle
        if !current_nodes[node] || node in cycle
            continue
        end

        in_edge_score = -Inf
        in_edge = -1
        out_edge_score = -Inf
        out_edge = -1

        for node_in_cycle in cycle
            # Exclude edges formed by two coreferred nodes
            _parent = old_input[node_in_cycle, node]
            _child = old_output[node_in_cycle, node]
            if coreference[_parent] != coreference[_child]
                if score_matrix[node_in_cycle, node] > in_edge_score
                    in_edge_score = score_matrix[node_in_cycle, node]
                    in_edge = node_in_cycle
                end
            end

            # Exclude edges formed by two coreferred nodes
            _parent = old_input[node, node_in_cycle]
            _child = old_output[node, node_in_cycle]
            if coreference[_parent] != coreference[_child]
                # Add the new edge score to the cycle score
                # Subtract the edge we're considering removing

                score = (cycle_score + 
                         score_matrix[node, node_in_cycle] -
                         score_matrix[parents[node_in_cycle], node_in_cycle])

                if score > out_edge_score
                    out_edge_score = score
                    out_edge = node_in_cycle    
                end
            end
        end

        score_matrix[cycle_rep, node] = in_edge_score
        old_input[cycle_rep, node] = old_input[in_edge, node]
        old_output[cycle_rep, node] = old_output[in_edge, node]

        score_matrix[node, cycle_rep] = out_edge_score
        old_input[node, cycle_rep] = old_input[node, out_edge]
        old_output[node, cycle_rep] = old_output[node, out_edge]
    end

    # For the next recursive iteration, we want to collapse this cycle
    # and treat is as a single node. This first node is arbitrarily chosen.
    # All other nodes will not be considered in the next iteration. We
    # also need to keep track of which representatives we are considering
    # this iteration because we need them below to check if we're done
    considered_reps = []
    for (i, node_in_cycle) in enumerate(cycle)
        push!(considered_reps, Set())
        # Choose the first as the representative, disable the rest
        if i>1
            current_nodes[node_in_cycle]=false
        end

        for node in representatives[node_in_cycle]
            push!(considered_reps[i], node)
            if i>1
                push!(representatives[cycle_rep], node)
            end
        end
    end

    adapted_chu_liu_edmonds!(N, score_matrix, coreference, current_nodes, final_edges, old_input, old_output, representatives)
    
    # Expansion stage
    # Check each node in cycle, if one of its reps is a key
    # in the final edges, it is the one we need.
    # The node we are looking for is the node which
    # is the child of the incoming edge to the cycle
    found = false
    key_node = -1
    for (i, node) in enumerate(cycle)
        for cycle_rep in considered_reps[i]
            if in(cycle_rep,keys(final_edges))
                key_node = node
                found = true
                break
            end
        end
        if found
            break
        end
    end


    # break the cycle
    previous = parents[key_node]
    while previous != key_node
        child = old_output[parents[previous], previous]
        parent = old_input[parents[previous], previous]
        final_edges[child] = parent
        previous = parents[previous]
    end
end             

function validate!(final_edges, N, original_score_matrix, coreference)
    # Count how many edges have been modified by this function
    modified = 0
    
    # Make a constant used by find_cycle
    current_nodes = [true for i in 1:N]
    
    # Group nodes by coreference
    group_by_precedent = Dict{Int, Any}()
    for (node, precedent) in enumerate(coreference)
        if !haskey(group_by_precedent, precedent)
            group_by_precedent[precedent] = Set()
        end
        push!(group_by_precedent[precedent], node)
    end
    
    # Validate the parents of nodes in each group
    for group in values(group_by_precedent)
        # Skip if only one node in the group
        if length(group) == 1
            continue
        end
        
        # Group conflicting nodes by parent
        conflicts_by_parent = Dict{Int, Array{Any, 1}}()
        for child in group
            parent = final_edges[child]
            if !haskey(conflicts_by_parent,parent)
                conflicts_by_parent[parent] = []
            end
            push!(conflicts_by_parent[parent],child)
        end
        
        # Keep the parents which have already been taken
        reserved_parents = Set(keys(conflicts_by_parent))
        
        for (parent, conflicts) in conflicts_by_parent
            # Skip if no conflict
            if length(conflicts) == 1
                continue
            end
            
            # Find the node that has the maximum edge with the parent
            highest_score = maximum(x->original_score_matrix[parent, x], conflicts)
            
            # Find the node corresponding to the highest score
            winner = conflicts[findall(x->original_score_matrix[parent,x]==highest_score, conflicts)][1]
            
            # Modify other nodes' parents
            for child in conflicts
                # Skip the winner
                if child == winner
                    continue
                end
                
                # Sort its candidate parents by score
                parent_scores = original_score_matrix[:, child]
                for _parent in sort([i for i in 1:N], by=x->parent_scores[x], rev=true)
                    # Skip current parent and reserved parents
                    if _parent == parent || in(_parent, reserved_parents)
                        continue
                    end
                    
                    # Check if there's any cycle if we use this parent
                    parents = copy(final_edges)
                    parents[child] = _parent
                    has_cycle, _ = find_cycle(parents, N, current_nodes)
                    if has_cycle
                        continue
                    end
                    
                    # Add it to the reserved parents
                    push!(reserved_parents, _parent) 
                    
                    # Update its parent
                    final_edges[child] = _parent
                    
                    # Update the counter
                    modified += 1
                    break
                end
            end
        end
    end
    modified         
end
