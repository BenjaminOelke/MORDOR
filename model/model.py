import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Input, BatchNormalization,Dropout,Concatenate,AlphaDropout,Lambda
from keras.models import Model
from keras import regularizers
import pickle
from .data_processing import *
from .losses import *
#import mordor as mo


'''
_____________________
|necessary arguments| 
---------------------

expr = path to expr data
cna = path to cna data
mut = path to mutation data
rsp = path to response data
hist = path to which history will be saved (has to be already set up)
pred = path to which predictions will be saved (has to be already set up)
emb = path to which the embeddings will be saved (has to be already set up)

____________________________
|default values for callbacks|
------------------------------
RoP_f = 0.5
RoP_p = 20
RoP_m_lr = 0.00000001
ES_m_d = 0.001
ES_p = 20

_______________________
|set default arguments|
-----------------------
e = epochs = 200
bs = batchsize = 32
tw = triplet loss weight = 1.0
ovs = oversampling = True
cw = class weights = False
src = data source = GDSC
a_f = activation function = selu

class weights may only be set true for single output models . It is not supported for multi output models

______________________
|How to use this code|
----------------------

import mordor as mo
import tensorflow as tf
import pandas as pd

train_val_paths = [tvexpr,tvcna,tvmut,tvrsp]
test_paths = [tstexpr,tstcna,tstmut,tstrsp]
o = [hist,pred,emb]

#this reads in and preprocesses the input data,builds the model graph and prepares the Dataset generator for training and validation data with default parameters
#code for reading in,preprocessing and the generator can be found in model.data_processing.FullData() 
model = mo.model.model.FullModel(train_val_paths,test_paths,o)

# setting up the metrics
model.set_metrics([tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

#setting up the callbacks
model.set_callbacks([tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                               factor=RoP_f,                       
                                                               patience=RoP_p,
                                                               min_lr=RoP_m_lr,
                                                               verbose=2,),
                         tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        min_delta=ES_m_d,
                        patience=ES_p,
                        verbose=0,
                        mode="auto",
                        baseline=None,
                        restore_best_weights=True,
                        )])
                        
#compiling the model
model.compile_model()

# fit the model and save the history to self.hist
model.fit_model()




'''

class FullModel(tf.keras.Model):
    
    def __init__(self,tv_paths,tst_paths,out_paths,
                 e: int=200,
                 learn_r: float=0.001,
                 drop: float=0.1,
                 l1: float=0.1,
                 l2: float=0.1,
                 gamma: float=0.0,
                 bs: int=32,
                 tw: float=1.0 ,
                 scope: str="all",
                 select_feat: bool=True,
                 integration: str="late",
                 mining = "hard",
                 c_loss: str='bce',
                 ovs: bool= True,
                 src: str="GDSC",
                 a_f: str='selu',
                 eager:bool = False,
                 no_classifier_loss: bool= False,
                 seed: int=0,
                 sq_euc_dist: bool= False,
                 *args, **kwargs):  
        
        super().__init__(*args, **kwargs)
        self.train_val_paths = tv_paths
        self.test_paths = tst_paths
        self.out_paths = out_paths
        self.ep = e
        self.scope = scope
        self.integration = integration
        self.seed = seed
        self.select_feat = select_feat
        #self.data = mo.model.data_processing.FullData(tv_paths,tst_paths,src,ovs,bs,integration,self.seed,self.scope,self.select_feat)
        self.data = FullData(tv_paths, tst_paths, src, ovs, bs, integration, self.seed,
                                                      self.scope, self.select_feat)
        self.train_data = self.data.train_data
        self.val_data = self.data.eval_data
        self.val_split = None
        self.b_s = bs
        self.cbaks = None
        self.mtrcs = None
        self.hist = None
        self.source = src
        self.triplet_weight = tw
        self.act_fct = a_f
        self.learn_r = learn_r
        self.drop = drop
        self.gamma = gamma
        self.l1 = l1
        self.l2 = l2
        self.no_classifier_loss = False
        self.eager = eager
        self.activation_function = a_f
        self.classifier_loss = c_loss
        self.model = self.build_model_graph()
        self.out_enc_names = None
        self.is_squared = sq_euc_dist
        self.mining = mining
        
        if self.is_squared:
            print("squared euclidian")
        else:
            print("euclidian")
        print(integration)
            
    
        

    def call(self,x):
        return super.__call__(x)
    
    def add_dense_block(self,dim,a_fct,k_init,prev_layer,key,cur_value):
        
        
        hidden = Dense(dim, activation=a_fct, kernel_initializer=k_init,kernel_regularizer=tf.keras.regularizers.L1L2(
    l1=self.l1, l2=self.l2))(prev_layer)
        checknan_hid_ol = Lambda(lambda x: self.check_nan(x,"hidden layer output ("+key+","+str(cur_value)+") for nan"))(hidden)
        batch_norm = BatchNormalization(center=True, scale=True)(checknan_hid_ol)
        checknan_bn_ol = Lambda(lambda x: self.check_nan(x,"hidden layer batch norm output ("+key+","+str(cur_value)+") for nan"))(batch_norm)
        dropout = AlphaDropout(self.drop, noise_shape=None, seed=None)(checknan_bn_ol)
        checknan_dr_ol = Lambda(lambda x: self.check_nan(x,"hidden layer batch norm output ("+key+","+str(cur_value)+") for nan"))(dropout)
        
        return checknan_dr_ol
    

        
    def build_model_graph(self):     # to do add option for activation function
    
        #init = {"selu":"lecun_normal","relu":"glorot_uniform"}
        init = {"selu":"lecun_normal","relu":"he_normal"}
        activ_f = self.activation_function
        kernel_init = init[activ_f]
        print("dense layers use "+activ_f+",kernel initialised with "+kernel_init)
        print(self.data.full_data[0]["train"].shape[1])
        
        if self.scope == "all":
            enc_arch = {"enc_expr":[self.data.full_data[0]["train"].shape[1],512],"enc_cna":[self.data.full_data[1]["train"].shape[1],128],"enc_mut":[self.data.full_data[2]["train"].shape[1],256]}
        else:
            enc_arch = {"enc_expr":[self.data.full_data[0]["train"].shape[1],512]}

        
        if self.integration == "late":
        
            emb_in_layers = []
            emb_out_layers = []

            for key,value in enc_arch.items():

                cur_value= value[0]
                input_layer = Input(shape = (value[0]))
                checknan_inp_ol = Lambda(lambda x: self.check_nan(x,"checking input layer output ("+key+","+str(cur_value)+") for nan"))(input_layer)
                emb_in_layers.append(input_layer)
                value.pop(0)
                pr_layer = checknan_inp_ol

                for v in value:
                    layer = self.add_dense_block(v,activ_f,kernel_init,pr_layer,key,v)
                    pr_layer = layer

                emb_out_layers.append(pr_layer)
            
            if self.scope == "all":
        
                concat = Concatenate()(emb_out_layers)
                checknan_l2_inp = Lambda(lambda x: self.check_nan(x,"l2_norm_input"))(concat)
                concatnorm = Lambda(lambda x: tf.math.l2_normalize(x),name = "conc")(checknan_l2_inp)
                classifier_dropout = AlphaDropout(self.drop, noise_shape=None, seed=None)(concatnorm)
                checknan_sigma = Lambda(lambda x: self.check_nan(x,"sigma input"))(classifier_dropout)
                output = Dense(1, activation = 'sigmoid',name="output")(checknan_sigma)


                if self.no_classifier_loss:                      # diagnostic mode (under construction)
                    outputs = [concatnorm]
                    inputs = [emb_in_layers]
                    model = Model(inputs=inputs, outputs=outputs)
                else:
                    outputs = [concatnorm,output]
                    inputs = [emb_in_layers]
                    model = Model(inputs=inputs, outputs=outputs)

            else:
                
                expression_embedding = emb_out_layers[0]
                checknan_l2_inp = Lambda(lambda x: self.check_nan(x,"l2_norm_input"))(expression_embedding)
                concatnorm = Lambda(lambda x: tf.math.l2_normalize(x),name = "conc")(checknan_l2_inp)   # the layer name is not changed here so the loss declaration futher downstream hasnt to be changed
                classifier_dropout = AlphaDropout(self.drop, noise_shape=None, seed=None)(concatnorm)
                checknan_sigma = Lambda(lambda x: self.check_nan(x,"sigma input"))(classifier_dropout)
                output = Dense(1, activation = 'sigmoid',name="output")(checknan_sigma)
                
                if self.no_classifier_loss:                      # diagnostic mode (under construction)
                    outputs = [concatnorm]
                    inputs = [emb_in_layers]
                    model = Model(inputs=inputs, outputs=outputs)
                else:
                    outputs = [concatnorm,output]
                    inputs = [emb_in_layers]
                    model = Model(inputs=inputs, outputs=outputs)

                
        if self.integration == "early":
            
            if self.scope=="only_expression":
                print("not implemented. If you want to run expression only, set integration to late")
                return None
        
            emb_in_layers = []
            emb_out_layers = []
            enc_out_layer_names =[]
            output_layers=[]

            for key,value in enc_arch.items():

                cur_value= value[0]
                input_layer = Input(shape = (value[0]))
                checknan_inp_ol = Lambda(lambda x: self.check_nan(x,"checking input layer output ("+key+","+str(cur_value)+") for nan"))(input_layer)

                emb_in_layers.append(input_layer)
                value.pop(0)
                pr_layer = checknan_inp_ol

                for v in value:
                    layer = self.add_dense_block(v,activ_f,kernel_init,pr_layer,key,v)
                    pr_layer = layer
                
                norm_emb = Lambda(lambda x: tf.math.l2_normalize(x),name=key)(pr_layer) 

                enc_out_layer_names.append(key)
                emb_out_layers.append(norm_emb)
                output_layers.append(norm_emb)

            concat = Concatenate()(emb_out_layers)
            checknan_sigma = Lambda(lambda x: self.check_nan(x,"sigma input"))(concat) # add dropout layer , check SuperFELT code again
            output = Dense(1, activation='sigmoid',name="output")(checknan_sigma)
            output_layers.append(output)
            
            outputs = output_layers
            inputs = [emb_in_layers]
            model = Model(inputs=inputs, outputs=outputs)
            self.enc_out_layer_names = enc_out_layer_names
        
        return model
    
    
    
    def compile_model(self):
        
        if self.eager:
            print("running eagerly")
        else:
            print("not running eagerly")
        
        
        if self.integration == "late":
            
            if self.no_classifier_loss:
                print("noBCE")
                self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learn_r),
                  #loss={"conc":mo.model.losses.TripletLoss(weight=self.triplet_weight,mining=self.mining)},
                  loss={"conc": TripletLoss(weight=self.triplet_weight,
                                                                             mining=self.mining)},
                  run_eagerly=self.eager
                  #metrics={"output":self.mtrcs}
                 )   
            else:
                
                if self.classifier_loss=="keras_fbce":
                    
                    print("keras focal BCE")
                    
                    self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learn_r),
                            #loss={"conc":mo.model.losses.TripletLoss(weight=self.triplet_weight,squared=self.is_squared,mining=self.mining),
                            loss = {"conc": TripletLoss(weight=self.triplet_weight,
                                                                              squared=self.is_squared,
                                                                              mining=self.mining),
                                  "output":tf.keras.losses.BinaryFocalCrossentropy(gamma=self.gamma)},
                                       run_eagerly=self.eager)
                
                if self.classifier_loss=="clipped_fbce":
                    
                    print("clipped focal BCE")
                    
                    self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learn_r),
                            loss={"conc":TripletLoss(weight=self.triplet_weight,squared=self.is_squared,mining=self.mining),
                                  "output":BinaryFocalLoss(gamma=self.gamma)},
                                       run_eagerly=self.eager)

                    
                if self.classifier_loss=="bce":
                
                    print("BCE")
                
                    self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learn_r),
                            loss={"conc":TripletLoss(weight=self.triplet_weight,squared=self.is_squared,mining=self.mining),
                                  "output":tf.keras.losses.BinaryCrossentropy()},
                                       run_eagerly=self.eager)
                    
        if self.integration == "early":
            
            if self.no_classifier_loss:
                print("not implemented. Model was not compiled")
                
            else:
                
                if self.classifier_loss=="keras_fbce":
                    
                    print("keras focal BCE")
                    
                    self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learn_r),
                            loss={self.enc_out_layer_names[0]:TripletLoss(weight=self.triplet_weight,squared=self.is_squared,mining=self.mining),
                                  self.enc_out_layer_names[1]:TripletLoss(weight=self.triplet_weight,squared=self.is_squared,mining=self.mining),
                                  self.enc_out_layer_names[2]:TripletLoss(weight=self.triplet_weight,squared=self.is_squared,mining=self.mining),
                                  "output":tf.keras.losses.BinaryFocalCrossentropy(gamma=self.gamma)},
                                       run_eagerly=self.eager)
                
                if self.classifier_loss=="clipped_fbce":
                    
                    print("clipped focal BCE")
                    
                    self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learn_r),
                            loss={self.enc_out_layer_names[0]:TripletLoss(weight=self.triplet_weight,squared=self.is_squared,mining=self.mining),
                                  self.enc_out_layer_names[1]:TripletLoss(weight=self.triplet_weight,squared=self.is_squared,mining=self.mining),
                                  self.enc_out_layer_names[2]:TripletLoss(weight=self.triplet_weight,squared=self.is_squared,mining=self.mining),
                                  "output":BinaryFocalLoss(gamma=self.gamma)},
                                       run_eagerly=self.eager)
                    
                if self.classifier_loss=="bce":
                
                    print("BCE")
                
                    self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learn_r),
                            loss={self.enc_out_layer_names[0]:TripletLoss(weight=self.triplet_weight,squared=self.is_squared,mining=self.mining),
                                  self.enc_out_layer_names[1]:TripletLoss(weight=self.triplet_weight,squared=self.is_squared,mining=self.mining),
                                  self.enc_out_layer_names[2]:TripletLoss(weight=self.triplet_weight,squared=self.is_squared,mining=self.mining),
                                  "output":tf.keras.losses.BinaryFocalCrossentropy(gamma=self.gamma)},
                                       run_eagerly=self.eager)
                
                
               
        
    def fit_model(self):

        print("")
        history = self.model.fit(
                      self.train_data,
                      callbacks=self.cbaks,
                      batch_size=self.b_s,
                      epochs=self.ep,
                      steps_per_epoch=int(len(self.data.full_data[0]["train"])/self.b_s),
                      validation_data=self.val_data,
                      validation_steps=int(len(self.data.full_data[0]["train"])/self.b_s) 
                       )
        self.hist = history
        
    
    def set_metrics(self,metrics_list):
        self.mtrcs = metrics_list
        
    def set_callbacks(self, callbacks_list):
        self.cbaks = callbacks_list
        
    def predict(self,x):
        return self.model.predict(x)
    
    def check_nan(self,t,what):
        tf.debugging.check_numerics(t, "checking "+what+" for nan", name=None)
        return t
    def save_results(self,name):
        
        if self.no_classifier_loss:                                      # diagnostic mode
            hist_path = self.out_paths[0]+"history_"+name+".csv"
            results = pd.DataFrame(columns=(self.hist.history.keys()))
            for key in self.hist.history.keys():
                results[key] = self.hist.history[key]
            results.to_csv(hist_path)
            
        else:
            hist_path = self.out_paths[0]+"history_"+name+".csv"
            results = pd.DataFrame(columns=(self.hist.history.keys()))
            for key in self.hist.history.keys():
                results[key] = self.hist.history[key]
            results.to_csv(hist_path)
        
        # this is where a threshold parameter can be introduced for different class discrimination, currently threshold is hard coded to 0.5
        def binarize (input):
            if input < 0.5: return 0
            else: return 1
        
        def process_prediction(pred,truth,which_slice):
            list_r = list()

            for i in range(0,len(pred[1])):
                
                p = binarize(pred[1][i,0])
                
                if p == 0 and truth[i,0] == 0:
                    list_r.append("TN")
                if p == 0 and truth[i,0] == 1:
                    list_r.append("FN")
                if p == 1 and truth[i,0] == 1:
                    list_r.append("TP")
                if p == 1 and truth[i,0] == 0:
                    list_r.append("FP")
             
            if which_slice == "val":

                emb_path = self.out_paths[2]+"emb_val_"+name
                embedding_file = open(emb_path,'wb')
                pickle.dump(pred[0],embedding_file)
                embedding_file.close()
                
                path = self.out_paths[1]+"pred_val_"+name
                
                if self.integration == "late":
                    results = pd.DataFrame(pred[1][:,0],columns=["pred"])
                    results.index = self.data.full_data[3]["val"].index
                    results["truth"] = self.data.full_data[3]["val"]
                    results["metric"] = list_r
                    results["smpl"] = range(len(results))
                    results.to_csv(path)

                if self.integration == "early":
                    results = pd.DataFrame(pred[3][:,0],columns=["pred"])
                    results.index = self.data.full_data[3]["val"].index
                    results["truth"] = self.data.full_data[3]["val"]
                    results["metric"] = list_r
                    results["smpl"] = range(len(results))
                    results.to_csv(path)
               
            if which_slice == "test":

                emb_path = self.out_paths[2]+"emb_test_"+name
                embedding_file = open(emb_path,'wb')
                pickle.dump(pred[0],embedding_file)
                embedding_file.close()
                
                path = self.out_paths[1]+"pred_test_"+name
                
                if self.integration == "late":
                    results = pd.DataFrame(pred[1][:,0],columns=["pred"])
                    results.index = self.data.full_data[9].index
                    results["truth"] = self.data.full_data[9]
                    results["metric"] = list_r
                    results["smpl"] = range(len(results))
                    results.to_csv(path)

                if self.integration == "early":
                    results = pd.DataFrame(pred[3][:,0],columns=["pred"])
                    results.index = self.data.full_data[9].index
                    results["truth"] = self.data.full_data[9]
                    results["metric"] = list_r
                    results["smpl"] = range(len(results))
                    results.to_csv(path)
                
        
        #train_pred = self.model.predict((self.data.full_data[0]["train"].to_numpy(), 
                            #self.data.full_data[1]["train"].to_numpy(),
                            #self.data.full_data[2]["train"].to_numpy()))
        #train_truth = self.data.full_data[3]["train"].to_numpy()
        
        #process_prediction(train_pred,train_truth,"train")
      
    
        if self.scope=="all":
        
            val_pred = self.model.predict((self.data.full_data[0]["val"].to_numpy(), 
                                self.data.full_data[1]["val"].to_numpy(),
                                self.data.full_data[2]["val"].to_numpy()))
            val_truth = self.data.full_data[3]["val"].to_numpy()

            process_prediction(val_pred,val_truth,"val")

            test_pred = self.model.predict((self.data.full_data[5].to_numpy(), 
                                self.data.full_data[6].to_numpy(),
                                self.data.full_data[7].to_numpy()))
            test_truth = self.data.full_data[9].to_numpy()
           

            process_prediction(test_pred,test_truth,"test")
            
        else:
            
            val_pred = self.model.predict(self.data.full_data[0]["val"].to_numpy())
            val_truth = self.data.full_data[3]["val"].to_numpy()

            process_prediction(val_pred,val_truth,"val")

            test_pred = self.model.predict(self.data.full_data[5].to_numpy())
            test_truth = self.data.full_data[9].to_numpy()

            process_prediction(test_pred,test_truth,"test")
        
        
        
        
    