include("data.jl")
include("basemodel.jl")

using CUDA, Knet

amrvocab = Knet.load("amrdata.jld2", "vocab")
trn = Knet.load("amrdata.jld2", "trn")
dev = Knet.load("amrdata.jld2", "dev")
ctrn = @time collect(trn)
cdev = @time collect(dev)


function initopt!(m::BaseModel, optimizer="Adam()")
    for par in params(m); 
        par.opt = eval(Meta.parse(optimizer)); 
        par.opt.gclip= 0.2
        #println("inited par: $par")
    end
end


model = BaseModel(H,Ex,Ey,L, DECODER_VOCAB_SIZE, edgenode_hiddensize, edgelabel_hiddensize, num_edgelabels, amrvocab.srcvocab, amrvocab.srccharactervocab, amrvocab.tgtvocab, amrvocab.tgtcharactervocab; bidirectional=true, dropout=Pdrop)
initopt!(model)
#a=forwback(model,first(ctrn))


function forw(model, batch)
    #srcpostags, tgtpostags, corefs, srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, parsermask, edgeheads, edgelabels, encodervocab_pad_idx, decodervocab_pad_idx = batch
    model(batch)
end

function forwback(model, batch)
    @diff model(batch)
end


function bytes(a)
    v = Knet.knetptrs(a)
    if isempty(v)
        return 0
    else
        sum(x->x.len, v)
    end
end

function fieldbytes(a)
    for n in fieldnames(typeof(a))
        println((n, bytes(getfield(a,n))))
    end
end


function alldata(bs)
    trn.batchsize= bs
    ctrn = collect(trn)
    for (i,batch) in enumerate(ctrn)
      println(i)
      #if i==54 return batch; end
      #if i==8 return batch end
      forw(model,batch)
      GC.gc(true)
      CUDA.memory_status()
    end
end

function alldataback(bs)
    trn.batchsize= bs
    ctrn = collect(trn)
    for batch in ctrn
       forwback(model,batch)
       GC.gc(true)
       CUDA.memory_status()
    end
end

