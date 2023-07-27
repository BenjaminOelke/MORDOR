import pandas as pd
import os
import math
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import tensorflow as tf

    
def get_scores(pred_file):
    
    preds = pd.read_csv(pred_file)
    
    # compute the entries of the confusion matrix
    tn = len(preds.loc[preds["metric"]=="TN"])
    tp = len(preds.loc[preds["metric"]=="TP"])
    fn = len(preds.loc[preds["metric"]=="FN"])
    fp = len(preds.loc[preds["metric"]=="FP"])
    
    #accuracy
    acc = float(tn+tp)/float(tn+tp+fn+fp)
    
    #precision
    # todo replace -2 by nan or document what this means and why it is important
    try:
        pr = float(tp)/float(tp+fp)
    except ZeroDivisionError:
        pr = -2
    
    #recall
    try:
        rc = float(tp)/float(tp+fn)
    except ZeroDivisionError:
        rc = -2
    
    #mathews correlation coefficient
    try:
        mcc = (float(tn*tp) - float(fn*fp))/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    except ZeroDivisionError:
        mcc = -2
    
    # roc-auc and pr-auc
    y_scores = preds["pred"]
    y_true = preds["truth"]
    
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recalls,precisions)
    
    fprs, tprs, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fprs,tprs)
    
    #auc
    
    
    #dominance
    try:
        tpr = float(tp)/float(tp+fn)
    except ZeroDivisionError:
        tpr = -2
    try:
        tnr = float(tn)/float(tn+fp)
    except ZeroDivisionError:
        tnr = 2 
    dominance = tpr-tnr
    sign = np.sign(dominance)
    dominance = abs(dominance)
    
    #Binary Cross Entropy
    '''
    summed_loss = 0
    for i in range(len(preds)):
        y_pred = preds.loc[i]["pred"]
        y_truth = preds.loc[i]["truth"]
        loss = y_truth*np.log(max(y_pred,1e-08))+((1-y_truth)*np.log(max(1-y_pred,1e-08)))
        summed_loss = summed_loss+loss
        i = i+1
    bce = (summed_loss*(-1))/len(preds)
    '''
    y_pred = preds["pred"]
    y_true = preds["truth"]

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    bce = bce(y_true, y_pred).numpy()
    
    return [acc,pr,rc,mcc,bce,roc_auc,pr_auc,dominance,sign,tpr,tnr]




    
def write_configs():
    home = "/dss/dsshome1/lxc0F/ga73jiv2/"
    in_dir = "/dss/dsshome1/lxc0F/ga73jiv2/Mordor/depmap/data/base/dl_data/"
    expr_dir = "exprs_homogenized/"
    cna_dir = "CNA/"
    mut_dir = "mutations/"
    rsp_dir = "response/"

    out_dir = "/dss/dsshome1/lxc0F/ga73jiv2/experiments/square_bce_late_expr_plus/results/"
    hist_dir = "histories/"
    pred_dir = "predictions/"
    emb_dir = "embeddings/"

    epochs = [1000]
    batch_size = [32,64]
    learning_rate =  [0.001,0.0001]
    RoP_factor = [0.5]
    RoP_patience = [50]
    RoP_min_lr = [0.00000001]
    ES_min_d = [0.001]
    ES_patience = [100]
    Triplet_weight = [0,0.1,0.5,1]
    train_val = ["GDSC"]
    test = ["PDX"]
    drug = ["Paclitaxel"]
    l1_kernel_reg = [0,1e-4,1e-2,1]
    l2_kernel_reg = [0,1e-4,1e-2,1]
    ovs = ["True"] 
    dropout = [0,0.1,0.2,0.5]
    #gamma = [0.5,1,2]
    gamma = [0]
    integration = "late"
    c_loss = "bce"
    squared = ["True"]                           # if ovs and squared have more then one entry,they need to be integrated into the for loop cascade
    
    i = 0
    with open('/home/debian/Mordor/configs/square_bce_late_expr_plus/square_bce_late_expr_plus.txt','a') as f:
        for tv in train_val:
            for tst in test:
                for d in drug:
                    for e in epochs:
                        for b in batch_size:
                            for l in learning_rate:
                                for trw in Triplet_weight:
                                    for dr in dropout:
                                        for l1r in l1_kernel_reg:
                                            for l2r in l2_kernel_reg:
                                                for rf in RoP_factor:
                                                    for rp in RoP_patience:
                                                        for rlr in RoP_min_lr:
                                                            for emd in ES_min_d:
                                                                for ep in ES_patience:
                                                                    for g in gamma:

                                                                        tvxpr = in_dir+expr_dir+tv+"_exprs."+d+".eb_with."+tst+"_exprs."+d+".tsv"
                                                                        tvcna = in_dir+cna_dir+tv+"_CNA."+d+".tsv"
                                                                        tvmut = in_dir+mut_dir+tv+"_mutations."+d+".tsv"
                                                                        tvrsp = in_dir+rsp_dir+tv+"_response."+d+".tsv"

                                                                        tstxpr = in_dir+expr_dir+tst+"_exprs."+d+".eb_with."+tv+"_exprs."+d+".tsv"
                                                                        tstcna = in_dir+cna_dir+tst+"_CNA."+d+".tsv"
                                                                        tstmut = in_dir+mut_dir+tst+"_mutations."+d+".tsv"
                                                                        tstrsp = in_dir+rsp_dir+tst+"_response."+d+".tsv"

                                                                        hist = out_dir+hist_dir
                                                                        pred = out_dir+pred_dir
                                                                        emb = out_dir+emb_dir

                                                                        src = str(tv)
                                                                        drg = str(d)

                                                                        epoch = str(e)
                                                                        lr = str(l) 
                                                                        batchsize = str(b)
                                                                        drop = str(dr)
                                                                        t_w = str(trw)
                                                                        l1 = str(l1r)
                                                                        l2 = str(l2r)

                                                                        r_f = str(rf)
                                                                        r_p = str(rp)
                                                                        r_m_l = str(rlr)

                                                                        e_m_d = str(emd)
                                                                        e_p = str(ep)
                                                                        g = str(g)

                                                                        conf = str(i)


                                                                        i = i+1

                                                                        f.write("bash "+home+"gridwrapper.sh "+out_dir+"taskstats/taskstat"+conf+".txt "+
                                                                                    tvxpr +" "+
                                                                                    tvcna +" "+
                                                                                    tvmut +" "+
                                                                                    tvrsp +" "+
                                                                                    tstxpr +" "+
                                                                                    tstcna +" "+
                                                                                    tstmut +" "+
                                                                                    tstrsp +" "+
                                                                                    hist +" "+
                                                                                    pred +" "+
                                                                                    emb +" "+
                                                                                    epoch +" "+
                                                                                    batchsize +" "+
                                                                                    lr +" "+
                                                                                    t_w +" "+
                                                                                    integration +" "+
                                                                                    r_f +" "+
                                                                                    r_p +" "+
                                                                                    r_m_l +" "+
                                                                                    e_m_d +" "+
                                                                                    e_p +" "+
                                                                                    ovs[0] +" "+ # if ovs and squared have more then one entry,integrate into the for loop cascade
                                                                                    squared[0]+" "+
                                                                                    l1+" "+
                                                                                    l2+" "+
                                                                                    g+" "+
                                                                                    c_loss+" "+
                                                                                    drop +" "+
                                                                                    src +" "+
                                                                                    conf +"\n")
    f.close()
                                                    

                                       
                            
                                                
                                               
                                               
                                       
                            