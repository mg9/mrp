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
ENCODER_TOKEN_VOCAB_SIZE = 18002
ENCODER_CHAR_VOCAB_SIZE = 125
MUSTCOPY_VOCAB_SIZE = 3
DECODER_TOKEN_VOCAB_SIZE = 12202
DECODER_CHAR_VOCAB_SIZE = 87
DECODER_COREF_VOCAB_SIZE = 500
Ex = TOKEN_EMB_DIM 
Ey = TOKEN_EMB_DIM
H, L, Pdrop = 512, 2, 0.33
vocabsize = 12202
edgenode_hiddensize = 256
edgelabel_hiddensize = 128
num_labels = 141
epochs = 10
encodervocab_pad_idx =ENCODER_TOKEN_VOCAB_SIZE          ## DO not confuse about vocab_pad_idx, only srctokens and tgttokens pads have been changed for embedding layers
decodervocab_pad_idx =DECODER_TOKEN_VOCAB_SIZE

function iterate(d::Dataset, state=collect(1:d.totalbatch))
    new_state = copy(state)
    new_state_len = length(new_state) 
    if new_state_len == 0 ; return nothing; end
    batch = d.batches[new_state[1]]
    srctokens = batch["encoder_tokens"]'                                                # -> (B,Tx)
    tgttokens = batch["decoder_tokens"][1:end-1,:]'                                     # -> (B,Ty)
    srctokens[srctokens.==0] .= encodervocab_pad_idx                                    # replace pad idx 0 with the latest index of array
    tgttokens[tgttokens.==0] .= decodervocab_pad_idx                                    # replace pad idx 0 with the latest index of array
    srcattentionmaps = permutedims(batch["src_copy_map"], [2,1,3])[2:end-1,:,:]         # -> (Tx,Tx+2, B)
    tgtattentionmaps = permutedims(batch["tgt_copy_map"], [2,1,3])[2:end,:,:]           # -> (Ty,Ty+1, B)
    generatetargets = batch["decoder_tokens"][2:end,:]   ## DO not confuse vocab_pad_idx thing, for embeddings src and tgttokens pads have been changed
    srccopytargets = batch["src_copy_indices"][2:end,:]  
    tgtcopytargets = batch["tgt_copy_indices"][2:end,:]
    deleteat!(new_state, 1)
    return  ((srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets) , new_state)
end
function length(d::Dataset); return d.totalbatch; end


function trainmodel(epochs)
    batches = read_batches_from_h5("../data/data-100batches.h5")  # TODO: prepare batches dynamically, for now load them
    trn = Dataset(batches)
    trnbatches = collect(trn)
    model = BaseModel(H,Ex,Ey,L,vocabsize, edgenode_hiddensize, edgelabel_hiddensize, num_labels; bidirectional=true, dropout=Pdrop)
    epoch = adam(model, ((srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets) for (srctokens, tgttokens, srcattentionmaps, tgtattentionmaps, generatetargets, srccopytargets, tgtcopytargets) in trnbatches))
    ctrn = collect(trn)
    traindata = (collect(flatten(shuffle!(ctrn) for i in 1:epochs)))
    progress!(adam(model,traindata), seconds=10) do y
        #println("hello seker")
    end
end
trainmodel(epochs)
