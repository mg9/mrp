include("data.jl")
include("s2s.jl")
include("pointer_generator.jl")
include("deep_biaffine_graph_decoder.jl")


batches = read_batches_from_h5("../data/data.h5")
batch = batches[1]

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
H, Ex, Ey, L, Dx, Dy, Pdrop = 512, Ex, Ey, 2, 2, 1, 0.33


## S2S
m = S2S(H,Ex,Ey; layers=L, bidirectional=(Dx==2), dropout=Pdrop)
enc_inputs, dec_inputs, generator_inputs, parser_inputs = prepare_batch_input(batch)
encoder_input, encoder_mask = prepare_encode_input(enc_inputs)
mem = encode(m, encoder_input, encoder_mask)
(key,val) = mem
decoder_input = prepare_decode_input(enc_inputs, dec_inputs, parser_inputs)
cell = randn!(similar(key, size(key,1), size(key,3), 2)) # -> (H,B,Ty)
hiddens, src_attn_vector, srcalignments, tgt_attn_vector, tgtalignments = decode(m, decoder_input, (key,val), cell)

## PointerGenerator
vocabsize = 12202
pointergenerator = PointerGenerator(H, vocabsize, 1e-20)
source_attentions = srcalignments
source_attention_maps = generator_inputs["copy_attention_maps"]
target_attentions = tgtalignments
target_attention_maps = generator_inputs["coref_attention_maps"]
pointergenerator_loss = pointergenerator(hiddens, source_attentions, source_attention_maps, target_attentions, target_attention_maps)

## DeepBiaffineGraphDecoder
edgenode_hidden_size = 256
edgelabel_hidden_size = 128
num_labels = 141
g = DeepBiaffineGraphDecoder(H, edgenode_hidden_size, edgelabel_hidden_size, num_labels)
edge_heads = parser_inputs["edge_heads"]     
edgelabels = parser_inputs["edge_labels"]   
corefs = parser_inputs["corefs"]            
mask = parser_inputs["mask"] 
graph_decoder_loss = g(hiddens, edge_heads, edgelabels, corefs, mask)

loss = pointergenerator_loss + graph_decoder_loss
