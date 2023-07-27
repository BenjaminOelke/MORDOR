import tensorflow as tf
import sys
from . import model

expr_train_val = sys.argv[1]
cna_train_val = sys.argv[2]
mut_train_val = sys.argv[3]
rsp_train_val = sys.argv[4]
expr_test = sys.argv[5]
cna_test = sys.argv[6]
mut_test = sys.argv[7]
rsp_test = sys.argv[8]
hist = sys.argv[9]
pred = sys.argv[10]
emb = sys.argv[11]
epochs = int(sys.argv[12])
batch_size = int(sys.argv[13])
lrn_r = float(sys.argv[14])
tr_weight = float(sys.argv[15])
scp = sys.argv[16]
feat_select=eval(sys.argv[17])
intgrt = sys.argv[18]                               
tr_mine = sys.argv[19]
RoP_f = float(sys.argv[20])
RoP_p = int(sys.argv[21])
RoP_m_lr = float(sys.argv[22])
ES_m_d = float(sys.argv[23])
ES_p = int(sys.argv[24])
ovrs = eval(sys.argv[25])
squared = eval(sys.argv[26])
l1_reg = float(sys.argv[27])
l2_reg = float(sys.argv[28])
fbce_gamma = float(sys.argv[29])
cl_loss = sys.argv[30]
drp = float(sys.argv[31])
source = sys.argv[32]
conf = sys.argv[33]


#expr,cna,mut,response data
paths_train_val = [expr_train_val,cna_train_val,mut_train_val,rsp_train_val]
paths_test = [expr_test,cna_test,mut_test,rsp_test]
# path to save history and predictions 
o = [hist,pred,emb]

for i in range(3):
    
    #getting some random range
    j = 5
    rand_seed = i*(i+j)
    j=i+j
    
    
    model = model.FullModel(paths_train_val,paths_test,o,
                                          e=epochs,
                                          learn_r=lrn_r,
                                          drop=drp,
                                          l1=l1_reg,
                                          l2=l2_reg,
                                          gamma=fbce_gamma,
                                          bs=batch_size,
                                          tw=tr_weight,
                                          scope = scp,
                                          select_feat=feat_select,
                                          integration=intgrt,
                                          mining= tr_mine,
                                          c_loss=cl_loss,
                                          ovs=ovrs,
                                          src=source,
                                          seed = rand_seed,
                                          sq_euc_dist = squared
                                            
                                            )


    model.set_metrics([tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
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

    model.compile_model()
    model.fit_model()
    name = conf+"_run_"+str(i+1)
    model.save_results(name)    
#print("done")
