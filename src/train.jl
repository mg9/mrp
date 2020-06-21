include("data.jl")
include("basemodel.jl")

# Model configs
BERT_EMB_DIM = 768
TOKEN_EMB_DIM = 300 #glove
POS_EMB_DIM = 100
CNN_EMB_DIM = 100
MUSTCOPY_EMB_DIM = 50
COREF_EMB_DIM = 50
POSTAG_VOCAB_SIZE = 51
ENCODER_TOKEN_VOCAB_SIZE = 18002
ENCODER_CHAR_VOCAB_SIZE = 125
MUSTCOPY_VOCAB_SIZE = 3
DECODER_TOKEN_VOCAB_SIZE = 12202
DECODER_CHAR_VOCAB_SIZE = 87
DECODER_COREF_VOCAB_SIZE = 500
Ex = BERT_EMB_DIM + TOKEN_EMB_DIM + POS_EMB_DIM + MUSTCOPY_EMB_DIM + CNN_EMB_DIM
Ey = TOKEN_EMB_DIM + POS_EMB_DIM + COREF_EMB_DIM + CNN_EMB_DIM
H, L, Pdrop = 512, 2, 0.33
vocabsize = 12202
edgenode_hiddensize = 256
edgelabel_hiddensize = 128
num_labels = 141
num_epochs = 10


function train(model, num_epochs)
    batches = read_batches_from_h5("../data/data.h5")  # TODO: prepare batches dynamically, for now load them
    num_batches = length(batches)
    for i in 1:num_epochs
        epoch_loss =  0.0 
        for j in 1:length(batches) 
            batch = batches[j]
            J = @diff model(batch)
            for par in params(model)
                g = grad(J, par)
                if g===nothing; continue
                else; update!(value(par), g, par.opt); end
            end
            instance_loss = value(J)
            epoch_loss += instance_loss
        end
        println("epoch: $i, totalbatchsize: $num_batches, trnloss: $epoch_loss")
    end
end

function initopt!(model::BaseModel, optimizer="Adam()")
    for par in params(model); par.opt = eval(Meta.parse(optimizer)); end
end

model = BaseModel(H,Ex,Ey,L,vocabsize, edgenode_hiddensize, edgelabel_hiddensize, num_labels; bidirectional=true, dropout=Pdrop)
initopt!(model)
train(model, num_epochs)