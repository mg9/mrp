include("data.jl")
include("basemodel.jl")

import Base: length, iterate
using YAML, Knet, IterTools, Random
using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random, CuArrays, IterTools

# Model configs
TOKEN_EMB_DIM = 300 #glove
POS_EMB_DIM = 100
COREF_EMB_DIM = 50
CHARCNN_EMB_DIM = 100 # The CharCNN output size, this is equal to length(Ngram_sizes) * Num_Filters
CHARCNN_EMB_SIZE = 100 # This is for the internal character embedding size
CHARCNN_NUM_FILTERS = 100
CHARCNN_NGRAM_SIZES = [3]
ENCODER_VOCAB_SIZE = 12202
DECODER_VOCAB_SIZE = 18002
POSTAG_VOCAB_SIZE = 52
DECODER_COREF_VOCAB_SIZE = 500
Ex = TOKEN_EMB_DIM + POS_EMB_DIM + CHARCNN_EMB_DIM
Ey = TOKEN_EMB_DIM + POS_EMB_DIM + CHARCNN_EMB_DIM + COREF_EMB_DIM 
H, L, Pdrop = 512, 2, 0.33
edgenode_hiddensize = 256
edgelabel_hiddensize = 128
num_edgelabels = 141
epochs = 50


## Not used yet
BERT_EMB_DIM = 768
CNN_EMB_DIM = 100
MUSTCOPY_EMB_DIM = 50
MUSTCOPY_VOCAB_SIZE = 3
ENCODER_CHAR_VOCAB_SIZE = 125
DECODER_CHAR_VOCAB_SIZE = 87


amrvocab = AMRVocab()
train_path = "../data/AMR/amr_2.0/train.txt.features.preproc"
dev_path = "../data/AMR/amr_2.0/dev.txt.features.preproc"
#test_path = "../data/AMR/amr_2.0/test.txt.features.preproc" 

trn  = @time AMRDataset(train_path, amrvocab, 1, init_vocab=true)  # 36519 instances
dev  = @time AMRDataset(dev_path, amrvocab, 1)    # 1368 instances
#test  = AMRDataset(test_path, amrvocab, 32)   # 1371 instances

ctrn = @time collect(trn)
cdev = @time collect(dev) 

println("Trn created with path $train_path", " with ", trn.ninstances, " instances")
println("Dev created with path $dev_path", " with ", dev.ninstances, " instances")


function train(m, epochs)
    for i in 1:10
        trnstart=time()
        println("epoch $i......")
        ##Trn
        trnloss =  0.0 
        for (i, (srcpostags, tgtpostags, corefs, srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, parsermask, edgeheads, edgelabels, encodervocab_pad_idx, decodervocab_pad_idx)) in enumerate(ctrn)
            println("iteration $i")
            J = @diff m(srcpostags, tgtpostags, corefs, srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, parsermask, edgeheads, edgelabels, encodervocab_pad_idx, decodervocab_pad_idx)
            for par in params(m)
                g = grad(J, par)
                if g===nothing; 
                    #println("par: $par, g: $g")
                    #continue
                else 
                    update!(value(par), g, par.opt); 
                end
            end
            trnloss += value(J)
            println("iteration $i, iterationloss: $trnloss")
        end
        trnend=time()
        elapsed_time = trnend - trnstart

        accuracy, xent, ppl, srccopy_accuracy, tgtcopy_accuracy, n_words = calculate_pointergenerator_metrics(m.p.metrics)
        uas, las, el = calculate_graphdecoder_metrics(m.g.metrics)

        println("--Train Metrics--")
        println("trnloss: $trnloss") #,  trnpgenloss: $trnpgenloss, trngraphloss: $trngraphloss")
        println("PointerGeneratormetrics, all_acc=$accuracy src_acc=$srccopy_accuracy tgt_acc=$tgtcopy_accuracy, ppl=$ppl, n_words: ", n_words)
        println("GraphDecoder metrics, UAS=$uas LAS=$las EL=$el")
        println("Epoch elapsed time: $elapsed_time sec.")

        reset(m.p.metrics)
        reset(m.g.metrics)
        
        ##Dev
        devloss =  0.0 
        for (srcpostags, tgtpostags, corefs, srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, parsermask, edgeheads, edgelabels, encodervocab_pad_idx, decodervocab_pad_idx) in cdev
            batchloss= m(srcpostags, tgtpostags, corefs, srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, parsermask, edgeheads, edgelabels, encodervocab_pad_idx, decodervocab_pad_idx)
            devloss += batchloss
        end
        accuracy, xent, ppl, srccopy_accuracy, tgtcopy_accuracy, n_words = calculate_pointergenerator_metrics(m.p.metrics)
        uas, las, el = calculate_graphdecoder_metrics(m.g.metrics)

        println("--Dev Metrics--")
        println("devloss: $devloss")
        println("PointerGeneratormetrics, all_acc=$accuracy src_acc=$srccopy_accuracy tgt_acc=$tgtcopy_accuracy, n_words: ", n_words)
        println("GraphDecoder metrics, UAS=$uas LAS=$las EL=$el")
        reset(m.p.metrics)
        reset(m.g.metrics)
    end
end

function initopt!(model::BaseModel, optimizer="Adam()")
    for par in params(model); 
        par.opt = eval(Meta.parse(optimizer)); 
        println("inited par: $par")
    end
end

m = BaseModel(H,Ex,Ey,L, DECODER_VOCAB_SIZE, edgenode_hiddensize, edgelabel_hiddensize, num_edgelabels, amrvocab.srcvocab, amrvocab.srccharactervocab, amrvocab.tgtvocab, amrvocab.tgtcharactervocab; bidirectional=true, dropout=Pdrop)
initopt!(m)
train(m, epochs) 

