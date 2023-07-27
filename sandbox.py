import os
from model.model import *
'''
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
feat_select = eval(sys.argv[17])
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
'''

# this section sets up the input and output directories

#home = r"C:\Users\49176\PycharmProjects\Mordor\Data".replace("\\\\","\\")

expr_train_val_path = "GDSC_exprs.Paclitaxel.tsv"
cna_train_val_path = "GDSC_CNA.Paclitaxel.tsv"
mut_train_val_path = "GDSC_mutations.Paclitaxel.tsv"
rsp_train_val_path = "GDSC_response.Paclitaxel.tsv"

expr_test_path = "PDX_exprs.Paclitaxel.tsv"
cna_test_path = "PDX_CNA.Paclitaxel.tsv"
mut_test_path = "PDX_mutations.Paclitaxel.tsv"
rsp_test_path = "PDX_response.Paclitaxel.tsv"

expr_train_val = expr_train_val_path
cna_train_val = cna_train_val_path
mut_train_val = mut_train_val_path
rsp_train_val = rsp_train_val_path

expr_test = expr_test_path
cna_test = cna_test_path
mut_test = mut_test_path
rsp_test = rsp_test_path


hist = ""
pred = ""
emb = ""

RoP_f = 0.5
RoP_p = 50
RoP_m_lr = 0.00000001
ES_m_d = 0.001
ES_p = 100


# expr,cna,mut,response data
paths_train_val = [expr_train_val, cna_train_val, mut_train_val, rsp_train_val]
paths_test = [expr_test, cna_test, mut_test, rsp_test]
# path to save history and prediction
o = [hist, pred, emb]



classifier = FullModel(paths_train_val, paths_test, o)

classifier.set_metrics([tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
classifier.set_callbacks([tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                          factor=RoP_f,
                                                          patience=RoP_p,
                                                          min_lr=RoP_m_lr,
                                                          verbose=2, ),
                     tf.keras.callbacks.EarlyStopping(
                         monitor="val_loss",
                         min_delta=ES_m_d,
                         patience=ES_p,
                         verbose=0,
                         mode="auto",
                         baseline=None,
                         restore_best_weights=True,
                     )])

classifier.compile_model()
classifier.fit_model()
name = "test"
classifier.save_results(name)