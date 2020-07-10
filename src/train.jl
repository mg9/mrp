include("data.jl")
include("basemodel.jl")

import Base: length, iterate
using YAML, Knet, IterTools, Random
using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random, CuArrays, IterTools
using HDF5

# Model configs
BERT_EMB_DIM = 768
TOKEN_EMB_DIM = 300 #glove
POS_EMB_DIM = 100
CNN_EMB_DIM = 100
MUSTCOPY_EMB_DIM = 50
COREF_EMB_DIM = 50
POSTAG_VOCAB_SIZE = 52
ENCODER_CHAR_VOCAB_SIZE = 125
MUSTCOPY_VOCAB_SIZE = 3
DECODER_CHAR_VOCAB_SIZE = 87
DECODER_COREF_VOCAB_SIZE = 500
Ex = TOKEN_EMB_DIM 
Ey = TOKEN_EMB_DIM
H, L, Pdrop = 512, 2, 0.33
vocabsize = 12202
edgenode_hiddensize = 256
edgelabel_hiddensize = 128
num_edgelabels = 141
epochs = 50


train_path = "../data/AMR/amr_2.0/train.txt.features.preproc"
dev_path = "../data/AMR/amr_2.0/dev.txt.features.preproc"
#test_path = "../data/AMR/amr_2.0/test.txt.features.preproc"

trnamrs = read_preprocessed_files(train_path) # 36519 instances
devamrs = read_preprocessed_files(dev_path)    # 1368 instances
#testamrs = read_preprocessed_files(test_path) # 1371 instances

_trnamrs=  sort(collect(trnamrs),by=x->length(x.graph.variables()), rev=false)
trn  = AMRDataset(_trnamrs[1:5000], 8) 
dev  = AMRDataset(devamrs, 8)
#test = AMRDataset(testamrs, 16)


function trainmodel(epochs)
    model = BaseModel(H,Ex,Ey,L,vocabsize, edgenode_hiddensize, edgelabel_hiddensize, num_edgelabels; bidirectional=true, dropout=Pdrop)
    (ctrn, cdev) = collect(trn), collect(dev) # use dev for now, since preparing train instances takes longer time
    println("Trn created with path $train_path", " with ", trn.ninstances, " instances")
    println("Dev created with path $dev_path", " with ", dev.ninstances, " instances")

    epoch = adam(model, 
           ((srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, parsermask, edgeheads, edgelabels, encodervocab_pad_idx, decodervocab_pad_idx)
            for (srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, parsermask, edgeheads, edgelabels, encodervocab_pad_idx, decodervocab_pad_idx) 
            in ctrn))
   
    progress!(adam(model, ctrn), seconds=60) do y
        
        ## Trn
        trnloss = 0.0
        resetmetrics(model)

        for (srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, parsermask, edgeheads, edgelabels,encodervocab_pad_idx, decodervocab_pad_idx) in ctrn
            batchloss = model(srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, parsermask, edgeheads, edgelabels, encodervocab_pad_idx, decodervocab_pad_idx)
            trnloss +=batchloss
        end
        accuracy = 100 * model.metrics.n_correct / model.metrics.n_words
        xent = trnloss / model.metrics.n_words
        ppl = exp(min(trnloss / model.metrics.n_words, 100))
        srccopy_accuracy = tgtcopy_accuracy = 0
        if model.metrics.n_source_copies == 0; srccopy_accuracy = -1;
        else srccopy_accuracy = 100 * (model.metrics.n_correct_source_copies / model.metrics.n_source_copies); end
        if model.metrics.n_target_copies == 0; tgtcopy_accuracy = -1; 
        else tgtcopy_accuracy = 100 * (model.metrics.n_correct_target_copies / model.metrics.n_target_copies); end
        println("PointerGenerator trn metrics: loss=$trnloss all_acc=$accuracy src_acc=$srccopy_accuracy tgt_acc=$tgtcopy_accuracy")

        if model.g.metrics.total_words > 0
            unlabeled_attachment_score = float(model.g.metrics.unlabeled_correct) / float(model.g.metrics.total_words)
            labeled_attachment_score = float(model.g.metrics.labeled_correct) / float(model.g.metrics.total_words)
            edgeloss = float(model.g.metrics.total_loss) / float(model.g.metrics.total_words)
            edgenode_loss = float(model.g.metrics.total_edgenode_loss) / float(model.g.metrics.total_words)
            edgelabel_loss = float(model.g.metrics.total_edgelabel_loss) / float(model.g.metrics.total_words)
        end
        println("GraphDecoder trn metrics: EL=$edgeloss UAS=$unlabeled_attachment_score LAS=$labeled_attachment_score")


        ## Dev
        devloss = 0.0
        resetmetrics(model)

        for (srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, parsermask, edgeheads, edgelabels, encodervocab_pad_idx, decodervocab_pad_idx) in cdev
            batchloss = model(srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, parsermask, edgeheads, edgelabels, encodervocab_pad_idx, decodervocab_pad_idx)
            devloss +=batchloss
        end
        accuracy = 100 * model.metrics.n_correct / model.metrics.n_words
        xent = trnloss / model.metrics.n_words
        ppl = exp(min(trnloss / model.metrics.n_words, 100))
        srccopy_accuracy = tgtcopy_accuracy = 0
        if model.metrics.n_source_copies == 0; srccopy_accuracy = -1;
        else srccopy_accuracy = 100 * (model.metrics.n_correct_source_copies / model.metrics.n_source_copies); end
        if model.metrics.n_target_copies == 0; tgtcopy_accuracy = -1; 
        else tgtcopy_accuracy = 100 * (model.metrics.n_correct_target_copies / model.metrics.n_target_copies); end
        println("PointerGenerator dev metrics: loss=$devloss all_acc=$accuracy src_acc=$srccopy_accuracy tgt_acc=$tgtcopy_accuracy")

        if model.g.metrics.total_words > 0
            unlabeled_attachment_score = float(model.g.metrics.unlabeled_correct) / float(model.g.metrics.total_words)
            labeled_attachment_score = float(model.g.metrics.labeled_correct) / float(model.g.metrics.total_words)
            edgeloss = float(model.g.metrics.total_loss) / float(model.g.metrics.total_words)
            edgenode_loss = float(model.g.metrics.total_edgenode_loss) / float(model.g.metrics.total_words)
            edgelabel_loss = float(model.g.metrics.total_edgelabel_loss) / float(model.g.metrics.total_words)
        end
        println("GraphDecoder dev metrics: EL=$edgeloss UAS=$unlabeled_attachment_score LAS=$labeled_attachment_score")

    end
end
trainmodel(epochs)



