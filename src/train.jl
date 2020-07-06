include("data.jl")
include("basemodel.jl")
import Base: length, iterate
using YAML, Knet, IterTools, Random
using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random, CuArrays, IterTools

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
epochs = 10


train_path = "../data/AMR/amr_2.0/train.txt.features.preproc"
dev_path = "../data/AMR/amr_2.0/dev.txt.features.preproc"
test_path = "../data/AMR/amr_2.0/test.txt.features.preproc"

trnamrs = read_preprocessed_files(train_path) 
devamrs = read_preprocessed_files(dev_path)
testamrs = read_preprocessed_files(test_path)

trn  = AMRDataset(devamrs, 4)
dev  = AMRDataset(devamrs, 8)
test = AMRDataset(testamrs, 16)



dev_path = "../data/AMR/amr_2.0/dev.txt.features.preproc"
devamrs = read_preprocessed_files(dev_path)
dev  = AMRDataset(devamrs, 8)


function trainmodel(epochs)
    model = BaseModel(H,Ex,Ey,L,vocabsize, edgenode_hiddensize, edgelabel_hiddensize, num_edgelabels; bidirectional=true, dropout=Pdrop)
    batches = collect(dev)
    println("Dataset created with path $dev_path", " with ", dev.ninstances, " instances")
    epoch = adam(model, 
           ((srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, parsermask, edgeheads, edgelabels)
            for (srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets, parsermask, edgeheads, edgelabels) 
            in batches))
    progress!(adam(model, batches), seconds=10) do y
    end
end
trainmodel(epochs)


