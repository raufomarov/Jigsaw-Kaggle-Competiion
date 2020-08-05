
#------------------------------BERT--------------------------------

Sys.setenv(TF_KERAS=1)
# make sure we use python 3
reticulate::py_install('keras-bert', pip=T)
# to see python version
reticulate::py_config()

#conda install keras-bert
reticulate::py_module_available('keras_bert')
tensorflow::tf_version()

#pip install keras-bert
#tensorflow::install_tensorflow(version = "1.15")

#Model structure
# bert_model.ckpt, which is for loading the weights from the TensorFlow checkpoint
# bert_config.json, which is a configuration file
# vocab.txt, which is for text tokenization
library(keras)
pretrained_path = 'multi_cased_L-12_H-768_A-12'
config_path = file.path(pretrained_path, 'bert_config.json')
checkpoint_path = file.path(pretrained_path, 'bert_model.ckpt')
vocab_path = file.path(pretrained_path, 'vocab.txt')

#Import Keras-Bert module via reticulate
library(reticulate)
k_bert = import('keras_bert')
token_dict = k_bert$load_vocabulary(vocab_path)
tokenizer = k_bert$Tokenizer(token_dict)

#Define model parameters and column names
seq_length = 60L
bch_size = 60
epochs = 1
learning_rate = 1e-4

DATA_COLUMN = 'comment_text'
LABEL_COLUMN = 'target'

#Bert model
model = k_bert$load_trained_model_from_checkpoint(
  config_path,
  checkpoint_path,
  training=T,
  trainable=T,
  seq_len=seq_length)

#Data structure, reading, preparation
# tokenize text
tokenize_fun = function(dataset) {
  c(indices, target, segments) %<-% list(list(),list(),list())
  for ( i in 1:nrow(dataset)) {
    c(indices_tok, segments_tok) %<-% tokenizer$encode(dataset[[DATA_COLUMN]][i],
                                                       max_len=seq_length)
    indices = indices %>% append(list(as.matrix(indices_tok)))
    target = target %>% append(dataset[[LABEL_COLUMN]][i])
    segments = segments %>% append(list(as.matrix(segments_tok)))
  }
  return(list(indices,segments, target))
}
# read data
dt_data = function(dir, rows_to_read){
  data = data.table::fread(dir, nrows=rows_to_read)
  c(x_train, x_segment, y_train) %<-% tokenize_fun(data)
  return(list(x_train, x_segment, y_train))
}

#load dataset
c(x_train,x_segment, y_train) %<-%
  dt_data('https://raw.githubusercontent.com/QSS-Analytics/Datasets/master/bert1.csv',3000)

#Matrix format for Keras-Bert
train = do.call(cbind,x_train) %>% t()
segments = do.call(cbind,x_segment) %>% t()
targets = do.call(cbind,y_train) %>% t()

concat = c(list(train ),list(segments))

#Calculate decay and warmup steps
c(decay_steps, warmup_steps) %<-% k_bert$calc_train_steps(
  targets %>% length(),
  batch_size=bch_size,
  epochs=epochs
)

#Determine inputs and outputs, then concatenate them
library(keras)

input_1 = get_layer(model,name = 'Input-Token')$input
input_2 = get_layer(model,name = 'Input-Segment')$input
inputs = list(input_1,input_2)

dense = get_layer(model,name = 'NSP-Dense')$output

outputs = dense %>% layer_dense(units=1L, activation='sigmoid',
                                kernel_initializer=initializer_truncated_normal(stddev = 0.02),
                                name = 'output')

model = keras_model(inputs = inputs,outputs = outputs)

#Compile model and begin training
model %>% compile(
  k_bert$AdamWarmup(decay_steps=decay_steps,
                    warmup_steps=warmup_steps, lr=learning_rate),
  loss = 'binary_crossentropy',
  metrics = tensorflow::tf$keras$metrics$AUC()
)

model %>% fit(
  concat,
  targets,
  epochs=epochs,
  batch_size=bch_size, validation_split=0.2)


## validation

c(x_train2,x_segment2, y_train2) %<-%
  dt_data('https://raw.githubusercontent.com/QSS-Analytics/Datasets/master/bert2.csv',5000)


train2 = do.call(cbind,x_train2) %>% t()
segments2 = do.call(cbind,x_segment2) %>% t()
targets2 = do.call(cbind,y_train2) %>% t()
concat2 = c(list(train2 ),list(segments2))

result = predict(model, concat2)

Metrics::auc(targets2,result)








