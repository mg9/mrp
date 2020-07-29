include("data.jl"); include("basemodel.jl")
using Knet, CUDA

# Prepare/save data once, then load from a file for saving time.
#amrvocab =  AMRVocab()
#train_path = "../data/AMR/amr_2.0/train.txt.features.preproc"
#dev_path = "../data/AMR/amr_2.0/dev.txt.features.preproc"
#test_path = "../data/AMR/amr_2.0/test.txt.features.preproc"
#trn   = AMRDataset(train_path, amrvocab, 64, init_vocab=true)  # 36519 instances
#dev   = @time AMRDataset(dev_path, amrvocab, 32)               # 1368 instances
#test  = @time AMRDataset(test_path, amrvocab, 32)              # 1371 instances
#Knet.save("amrdata.jld2", "trn",trn, "dev", dev, "test", test, "vocab", amrvocab)

amrvocab = Knet.load("amrdata.jld2", "vocab")
trn = Knet.load("amrdata.jld2", "trn")
dev = Knet.load("amrdata.jld2", "dev")
ctrn = @time collect(trn)
cdev = @time collect(dev)


function train(m; epochs=1)
    for i in 1:epochs
        trnstart=time()
        println("epoch $i......")
        ##Trn
        trnloss =  0.0 
        for (i, batch) in enumerate(ctrn)
            println("iteration $i")
            J = @diff m(batch)
            for par in params(m)
                g = grad(J, par)
                update!(par.value,g, par.opt)
            end
            trnloss += value(J)
            #println("iteration $i, iterationloss: ", value(J))
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
        for (i,batch) in enumerate(cdev)
            devloss += m(batch)
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


function initopt!(m::BaseModel, optimizer="Adam()")
    for par in params(m); 
        par.opt = eval(Meta.parse(optimizer)); 
        par.opt.gclip= 0.2
        #println("inited par: $par")
    end
end


model =nothing; GC.gc(true);
model = BaseModel(H,Ex,Ey,L, DECODER_VOCAB_SIZE, edgenode_hiddensize, edgelabel_hiddensize, num_edgelabels, amrvocab.srcvocab, amrvocab.srccharactervocab, amrvocab.tgtvocab, amrvocab.tgtcharactervocab; bidirectional=true, dropout=Pdrop)
initopt!(model)
train(model, epochs=100) 


#= TODO: Change training with high-level functions
traindata = (collect(flatten(shuffle!(ctrn) for i in 1:epochs)))
progress!(adam(model, traindata), seconds=10) do y
    println("test")
end
=#
