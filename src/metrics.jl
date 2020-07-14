mutable struct PointerGeneratorMetrics
    n_words 
    n_correct
    n_source_copies
    n_correct_source_copies
    n_correct_source_points 
    n_target_copies
    n_correct_target_copies
    n_correct_target_points
    generatorloss
end

mutable struct GraphMetrics
    unlabeled_correct     
    exact_unlabeled_correct 
    labeled_correct
    exact_labeled_correct
    total_sentences 
    total_words 
    total_loss
    total_edgenode_loss 
    total_edgelabel_loss 
end



function (gm::GraphMetrics)(pred_edgeheads, pred_edgelabels, gold_edgeheads, gold_edgelabels, parsermask, graphloss, edgeheadloss, edgelabelloss)
    #TODO: other metrics s.t. unlabeled exact match, labeled exact match etc., move metrics part into a function
    #TODO: ignore label class mask?
    
    correct_indices = (pred_edgeheads .== gold_edgeheads)  .*  (1 .- parsermask)   
    correct_labels = (pred_edgelabels .== gold_edgelabels) .*  (1  .- parsermask)    
    correct_labels_and_indices = correct_indices .* correct_labels             
    gm.unlabeled_correct += sum(correct_indices)               
    gm.labeled_correct += sum(correct_labels_and_indices)
    gm.total_sentences += size(pred_edgeheads,1)
    gm.total_words +=  sum(1 .- parsermask)
    gm.total_loss += graphloss
    gm.total_edgenode_loss += edgeheadloss
    gm.total_edgelabel_loss += edgelabelloss
    #println("GraphDecoder batch metrics, UAS: $UAS, LAS: $LAS,  EL=$EL, total_words: ", gm.total_words)
end



function (pm::PointerGeneratorMetrics)(predictions, targets, non_pad_mask, non_tgtcopy_mask, srccopy_mask, tgtcopy_mask, vocabsize, src_dynamic_vocabsize, generatorloss)
    targets = targets .* non_pad_mask                      
    pred_eq = (predictions .== targets) .* non_pad_mask     
    
    pm.n_words += sum(non_pad_mask .== 1)
    pm.n_correct += sum(pred_eq .== 1)
    pm.n_target_copies +=  sum((tgtcopy_mask .*   non_pad_mask) .==1)
    pm.n_correct_target_copies +=  sum((pred_eq .* tgtcopy_mask) .==1)
    pm.n_correct_target_points += sum( ((predictions .>= vocabsize+src_dynamic_vocabsize+1) .* tgtcopy_mask .* non_pad_mask) .==1)
    pm.n_source_copies +=  sum((srccopy_mask .* non_tgtcopy_mask .*  non_pad_mask) .==1)
    pm.n_correct_source_copies +=  sum((pred_eq .* non_tgtcopy_mask .* srccopy_mask) .==1)
    pm.n_correct_source_points += sum( ((predictions .>= vocabsize+1) .* (predictions .< vocabsize+src_dynamic_vocabsize+1)  .*non_tgtcopy_mask .*srccopy_mask .* non_pad_mask) .== 1)
    pm.generatorloss += generatorloss
    #println("PointerGeneratormetrics batch metrics, loss=$generatorloss all_acc=$accuracy src_acc=$srccopy_accuracy tgt_acc=$tgtcopy_accuracy, pm.n_words: ", pm.n_words)
end

function calculate_graphdecoder_metrics(gm::GraphMetrics)
    if gm.total_words > 0
        unlabeled_attachment_score = float(gm.unlabeled_correct) / float(gm.total_words)
        labeled_attachment_score = float(gm.labeled_correct) / float(gm.total_words)
        edgeloss = float(gm.total_loss) / float(gm.total_words)
        edgenode_loss = float(gm.total_edgenode_loss) / float(gm.total_words)
        edgelabel_loss = float(gm.total_edgelabel_loss) / float(gm.total_words)
    end

    UAS = unlabeled_attachment_score
    LAS = labeled_attachment_score
    EL  = value(edgeloss)
    return UAS, LAS, EL
end


function calculate_pointergenerator_metrics(pm::PointerGeneratorMetrics)
    accuracy = 100 * pm.n_correct / pm.n_words
    xent = pm.generatorloss / pm.n_words
    ppl = exp(min(pm.generatorloss / pm.n_words, 100))
    srccopy_accuracy = tgtcopy_accuracy = 0
    if pm.n_source_copies == 0 
        srccopy_accuracy = -1
    else 
        srccopy_accuracy = 100 * (pm.n_correct_source_copies / pm.n_source_copies); 
    end
    if pm.n_target_copies == 0
        tgtcopy_accuracy = -1
    else 
        tgtcopy_accuracy = 100 * (pm.n_correct_target_copies / pm.n_target_copies); 
    end
    return accuracy, xent, ppl, srccopy_accuracy, tgtcopy_accuracy, pm.n_words
end



function reset(gm::GraphMetrics)
    gm.unlabeled_correct = 0  
    gm.exact_unlabeled_correct  = 0
    gm.labeled_correct =  0
    gm.exact_labeled_correct = 0 
    gm.total_sentences = 0 
    gm.total_words = 0
    gm.total_loss = 0
    gm.total_edgenode_loss = 0
    gm.total_edgelabel_loss = 0
    gm.total_loss = 0
end

function reset(pm::PointerGeneratorMetrics)
    pm.n_words = 0
    pm.n_correct = 0
    pm.n_source_copies = 0
    pm.n_correct_source_copies = 0 
    pm.n_correct_source_points = 0 
    pm.n_target_copies = 0 
    pm.n_correct_target_copies = 0
    pm.n_correct_target_points = 0
    pm.generatorloss = 0
end

