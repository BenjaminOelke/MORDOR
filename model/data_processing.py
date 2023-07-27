import numpy as np
import pandas as pd
import tensorflow as tf 
from kneed import KneeLocator
import os
from decimal import Decimal
class FullData():
    
    def __init__(self,tv_paths,tst_paths,data_source, ovsmpl,batch_size,integration,seed,scope,select_feat, *args, **kwargs): #paths is list/array gives the path to exp,                                                                              cna, mut and response data df's in that order 
        self.train_val_paths = tv_paths
        self.test_paths = tst_paths
        self.data_source = data_source
        self.ovs = ovsmpl
        self.bs = batch_size
        self.seed = seed
        self.integration = integration
        self.s_samples_index = None
        self.r_samples_index = None
        self.select_feat = select_feat
        self.scope = scope
        self.full_data =[[],[],[],[],[],[],[],[],[],[]]
        self.train_data = None
        self.eval_data = None
        self.ccle_seed = 13
        self.get_input()
        #self.expr_var_genes = None
        #self.mut_var_genes = None
        #self.cna_var_genes = None

    ''' 
    # to log normalize expression data. to do implement switch. atm a different normalization is done during input parsing
    # to do banish into utils not used in mordor 0.0.0
    def log_normalize(df):
    
        expr_nt = pd.DataFrame(index=expr.columns)
        for smpl in df.index:
            a = expr.loc[smpl]
            scaling_factor = 100  # this is the total counts per sample after normalization and prior to log transformation
            total_counts = a.sum(axis=0)[..., np.newaxis]
            normalized_values = a * scaling_factor / total_counts  # divide each sample by its total counts ( -> total counts scaling_factor)
            log_vals = np.log1p(normalized_values)
            log_vals = np.expm1(normalized_values)
            a = pd.DataFrame(log_vals)
            expr_nt = pd.concat((expr_nt,a),axis=1)     
        return expr_nt.T
    '''
    # to binarize prediction output
    def binarize (input):
        if input < 0.5: return 0
        else: return 1

    def rank_feature(self,df,feature):
    
        if feature == "variance":
            df_c = df.copy()
            df_var = pd.DataFrame(pd.DataFrame.var(df_c))
            df_var  = df_var [0].sort_values(ascending = False)
            #push the gene ids into the dataframe
            df_var = df_var.reset_index()
            #push the ranks into the dataframe
            df_var = df_var.reset_index()
            df_var.columns=["rank","genes","variance"]

            return df_var

        if feature == "counts":

            counts = {}
            for gene in df.columns:
                g_sum = df[gene].sum()
                counts[gene] = g_sum

            df_counts = pd.DataFrame.from_dict(counts,orient="index")
            df_counts = df_counts.sort_values(0,ascending=False)
            df_counts = df_counts.reset_index()
            df_counts = df_counts.reset_index()
            df_counts.columns = ["rank","genes","counts"]

            return df_counts

    
    
    def feature_selection_gdsc(self,tv_expr,tv_cna,tv_mut,tst_expr,tst_cna,tst_mut):


        expr_gene_overlap = set(tv_expr.columns).intersection(set(tst_expr.columns))
        expr = tv_expr[list(expr_gene_overlap)]
        cna_gene_overlap = set(tv_cna.columns).intersection(set(tst_cna.columns))
        cna = tv_cna[list(cna_gene_overlap)]
        mut_gene_overlap = set(tv_mut.columns).intersection(set(tst_mut.columns))
        mut = tv_mut[list(mut_gene_overlap)]
        
        expr_var = self.rank_feature(expr,"variance")
        kneedle = KneeLocator(expr_var["rank"],expr_var["variance"], curve="convex", direction="decreasing",online=True)
        expr_feat_genes = set(expr_var.loc[expr_var["rank"]<=kneedle.knee]["genes"])
        
        cna_counts = self.rank_feature(cna,"counts")
        cna_feat_genes = set(cna_counts.loc[cna_counts["counts"] >=int(cna_counts["counts"].mean())]["genes"])
        
        mut_counts = self.rank_feature(mut,"counts")
        mut_feat_genes = set(mut_counts.loc[mut_counts["counts"] >=5]["genes"])
        
        return(expr_feat_genes,cna_feat_genes,mut_feat_genes)
        
        
    
    def get_ccle_data(self,scope_all):
        
            
            if scope_all:
                
                ccle_expr = pd.read_csv(self.train_val_paths[0],sep="\t",index_col=0)
                ccle_expr.index.name = 'CCLE Name / ENTREZID'
                
                 
                
                ccle_cna = pd.read_csv(self.train_val_paths[1],sep="\t")
                ccle_cna = ccle_cna.rename(columns={"Unnamed: 0":'CCLE Name / ENTREZID'})
                ccle_cna = ccle_cna.set_index("CCLE Name / ENTREZID")

                
                ccle_mut = pd.read_csv(self.train_val_paths[2],sep="\t",index_col = 0)
                ccle_mut.index.name = 'CCLE Name / ENTREZID'
                
                
               
                ccle_rsp = pd.read_csv(self.train_val_paths[3],sep="\t",index_col=0)
                ccle_bin_rsp = pd.DataFrame(ccle_rsp["response"])
                ccle_bin_rsp["response"] = 0


                # response binarization loop
                for sample in ccle_rsp.index:
                    if ccle_rsp.at[sample,"response"] == "S":
                        ccle_bin_rsp.at[sample,"response"] = 1
                    else:
                        ccle_bin_rsp.at[sample,"response"] = 0

                ccle_rsp["bin"] = ccle_bin_rsp["response"]
                ccle_bin_rsp.columns = ["rsp"]
                
                self.r_samples_index= ccle_bin_rsp.loc[ccle_bin_rsp["rsp"]==0].index
                self.s_samples_index= ccle_bin_rsp.loc[ccle_bin_rsp["rsp"]==1].index
                print(len(self.r_samples_index)+len(self.s_samples_index))
                print(len(set(ccle_bin_rsp.index).intersection(set(ccle_expr.index))))
                print(len(set(ccle_expr.index)))
                
                n_s_smpl_tst = len(self.s_samples_index) - int(len(self.s_samples_index)*0.8) # nmbr val smpl, possibly here one could introduce a variable to change train/vall split ratio
                n_r_smpl_tst = len(self.r_samples_index) - int(len(self.r_samples_index)*0.8)
                np.random.seed(self.ccle_seed)  # this is to make sure the "same random split" is generated all times to compare models
                tst_split_s = np.random.choice(a=self.s_samples_index, size=n_s_smpl_tst,replace=False)
                np.random.seed(self.ccle_seed) 
                tst_split_r = np.random.choice(a=self.r_samples_index, size=n_r_smpl_tst,replace=False)

                tv_split_s = np.array([i for i in self.s_samples_index if i not in tst_split_s])
                tv_split_r = np.array([i for i in self.r_samples_index if i not in tst_split_r])

                tst_expr = ccle_expr.loc[np.concatenate((tst_split_s,tst_split_r))].sort_index()
                tst_cna = ccle_cna.loc[np.concatenate((tst_split_s,tst_split_r))].sort_index()
                tst_mut = ccle_mut.loc[np.concatenate((tst_split_s,tst_split_r))].sort_index()
                tst_bin_rsp = ccle_bin_rsp.loc[np.concatenate((tst_split_s,tst_split_r))].sort_index()
                tst_rsp = ccle_rsp.loc[np.concatenate((tst_split_s,tst_split_r))].sort_index()

                tv_expr = ccle_expr.loc[np.concatenate((tv_split_s,tv_split_r))].sort_index()
                tv_cna = ccle_cna.loc[np.concatenate((tv_split_s,tv_split_r))].sort_index()
                tv_mut = ccle_mut.loc[np.concatenate((tv_split_s,tv_split_r))].sort_index()
                tv_bin_rsp = ccle_bin_rsp.loc[np.concatenate((tv_split_s,tv_split_r))].sort_index()
                tv_rsp = ccle_rsp.loc[np.concatenate((tv_split_s,tv_split_r))].sort_index()
                
                self.r_samples_index= tv_bin_rsp.loc[ccle_bin_rsp["rsp"]==0].index
                self.s_samples_index= tv_bin_rsp.loc[ccle_bin_rsp["rsp"]==1].index

                
                return(tv_expr,tv_cna,tv_mut,tv_rsp,tv_bin_rsp,tst_expr,tst_cna,tst_mut,tst_bin_rsp,tst_rsp)
            
            else:
                
                ccle_expr = pd.read_csv(self.train_val_paths[0],sep="\t",index_col=0)
                ccle_expr.index.name = 'CCLE Name / ENTREZID'
                
                ccle_rsp = pd.read_csv(self.train_val_paths[3],sep="\t",index_col=0)
                ccle_bin_rsp = pd.DataFrame(ccle_rsp["response"])
                ccle_bin_rsp["response"] = 0

                # response binarization loop
                for sample in ccle_rsp.index:
                    if ccle_rsp.at[sample,"response"] == "S":
                        ccle_bin_rsp.at[sample,"response"] = 1
                    else:
                        ccle_bin_rsp.at[sample,"response"] = 0

                ccle_rsp["bin"] = ccle_bin_rsp["response"]
                ccle_bin_rsp.columns = ["rsp"]

                # get indices for different response classes
                self.r_samples_index= ccle_bin_rsp.loc[ccle_bin_rsp["rsp"]==0].index
                self.s_samples_index= ccle_bin_rsp.loc[ccle_bin_rsp["rsp"]==1].index
                
                n_s_smpl_tst = len(self.s_samples_index) - int(len(self.s_samples_index)*0.8) # nmbr val smpl, possibly here one could introduce a variable to change train/vall split ratio
                n_r_smpl_tst = len(self.r_samples_index) - int(len(self.r_samples_index)*0.8)

                nsmpl = len(self.s_samples_index)+len(self.r_samples_index) # nmbr of all samples
               

                np.random.seed(self.ccle_seed)  # this is to make sure the "same random split" is generated all times to compare models
                tst_split_s = np.random.choice(a=self.s_samples_index, size=n_s_smpl_tst,replace=False)
                np.random.seed(self.ccle_seed) 
                tst_split_r = np.random.choice(a=self.r_samples_index, size=n_r_smpl_tst,replace=False)
                
                tv_split_s = np.array([i for i in self.s_samples_index if i not in tst_split_s])
                tv_split_r = np.array([i for i in self.r_samples_index if i not in tst_split_r])

                tst_expr = ccle_expr.loc[np.concatenate((tst_split_s,tst_split_r))].sort_index()
                tst_bin_rsp = ccle_bin_rsp.loc[np.concatenate((tst_split_s,tst_split_r))].sort_index()
                tst_rsp = ccle_rsp.loc[np.concatenate((tst_split_s,tst_split_r))].sort_index()

                tv_expr = ccle_expr.loc[np.concatenate((tv_split_s,tv_split_r))].sort_index()
                tv_bin_rsp = ccle_bin_rsp.loc[np.concatenate((tv_split_s,tv_split_r))].sort_index()
                tv_rsp = ccle_rsp.loc[np.concatenate((tv_split_s,tv_split_r))].sort_index()
                
                self.r_samples_index= tv_bin_rsp.loc[tv_bin_rsp["rsp"]==0].index
                self.s_samples_index= tv_bin_rsp.loc[tv_bin_rsp["rsp"]==1].index

      
                
                return(tv_expr,tv_rsp,tv_bin_rsp,tst_expr,tst_rsp,tst_bin_rsp)



    def get_gdsc_data(self,scope_all):



            if scope_all:
                tv_expr = pd.read_csv(self.train_val_paths[0],sep="\t",decimal=".",index_col=0).T
                tv_expr.index = tv_expr.index.astype("int")
                #tv_expr = df[decimal_columns].applymap(to_decimal)


                # this is where switch for different normalizations goes
                tv_expr = tv_expr.div(tv_expr.sum(axis=1), axis=0)   # rowwise normalizaion per cell line
                
                tv_cna = pd.read_csv(self.train_val_paths[1],sep="\t",index_col=0).T
                tv_cna.index = tv_cna.index.astype("int")

                tv_mut = pd.read_csv(self.train_val_paths[2],sep="\t",index_col=0).T
                tv_mut.index = tv_mut.index.astype("int")

                tv_rsp = pd.read_csv(self.train_val_paths[3],sep="\t")
                tv_rsp = tv_rsp.set_index(tv_rsp["sample_name"])
                tv_bin_rsp = pd.DataFrame(tv_rsp["response"])
                tv_bin_rsp["response"] = 0

                # response binarization loop
                for sample in tv_rsp.index:
                    if tv_rsp.at[sample,"response"] == "S":
                        tv_bin_rsp.at[sample,"response"] = 1
                    else:
                        tv_bin_rsp.at[sample,"response"] = 0

                tv_rsp["bin"] = tv_bin_rsp["response"]
                tv_bin_rsp.columns = ["rsp"]

                # get indices for different response classes
                self.r_samples_index= tv_bin_rsp.loc[tv_bin_rsp["rsp"]==0].index
                self.s_samples_index= tv_bin_rsp.loc[tv_bin_rsp["rsp"]==1].index


                return(tv_expr,tv_cna,tv_mut,tv_rsp,tv_bin_rsp)
            
            else:
                
                tv_expr = pd.read_csv(self.train_val_paths[0],sep="\t",decimal=".",index_col=0).T
                tv_expr.index = tv_expr.index.astype("int")
                tv_expr = tv_expr.div(tv_expr.sum(axis=1), axis=0)
                
                tv_rsp = pd.read_csv(self.train_val_paths[3],sep="\t")
                tv_rsp = tv_rsp.set_index(tv_rsp["sample_name"])
                tv_bin_rsp = pd.DataFrame(tv_rsp["response"])
                tv_bin_rsp["response"] = 0

                # response binarization loop
                for sample in tv_rsp.index:
                    if tv_rsp.at[sample,"response"] == "S":
                        tv_bin_rsp.at[sample,"response"] = 1
                    else:
                        tv_bin_rsp.at[sample,"response"] = 0

                tv_rsp["bin"] = tv_bin_rsp["response"]
                tv_bin_rsp.columns = ["rsp"]

                # get indices for different response classes
                self.r_samples_index= tv_bin_rsp.loc[tv_bin_rsp["rsp"]==0].index
                self.s_samples_index= tv_bin_rsp.loc[tv_bin_rsp["rsp"]==1].index
                
                return(tv_expr,tv_rsp,tv_bin_rsp)
                
        
    def get_pdx_data(self,scope_all):
        
        if scope_all:
    
            tst_expr = pd.read_csv(self.test_paths[0],sep="\t",decimal=".",index_col=0).T
            #different normalization switch goes here
            tst_expr = tst_expr.div(tst_expr.sum(axis=1), axis=0)

            tst_cna = pd.read_csv(self.test_paths[1],sep="\t",index_col=0).T
            tst_mut = pd.read_csv(self.test_paths[2],sep="\t",index_col=0).T
            tst_rsp = pd.read_csv(self.test_paths[3],sep="\t")

            tst_rsp = tst_rsp.set_index(tst_rsp["sample_name"])
            tst_bin_rsp = pd.DataFrame(tst_rsp["response"])
            tst_bin_rsp["response"] = 0
                                            # binarization loop
            for sample in tst_rsp.index:
                if tst_rsp.at[sample,"response"] == "S":
                    tst_bin_rsp.at[sample,"response"] = 1
                else:
                    tst_bin_rsp.at[sample,"response"] = 0

            tst_rsp["bin"] = tst_bin_rsp["response"]
            tst_bin_rsp.columns = ["rsp"]

            return(tst_expr,tst_cna,tst_mut,tst_rsp,tst_bin_rsp)
        
        else:
            
            tst_expr = pd.read_csv(self.test_paths[0],sep="\t",decimal=".",index_col=0).T
            #different normalization switch goes here
            tst_expr = tst_expr.div(tst_expr.sum(axis=1), axis=0)
            
            tst_rsp = pd.read_csv(self.test_paths[3],sep="\t")

            tst_rsp = tst_rsp.set_index(tst_rsp["sample_name"])
            tst_bin_rsp = pd.DataFrame(tst_rsp["response"])
            tst_bin_rsp["response"] = 0
                                            # binarization loop
            for sample in tst_rsp.index:
                if tst_rsp.at[sample,"response"] == "S":
                    tst_bin_rsp.at[sample,"response"] = 1
                else:
                    tst_bin_rsp.at[sample,"response"] = 0

            tst_rsp["bin"] = tst_bin_rsp["response"]
            tst_bin_rsp.columns = ["rsp"]
            
            return(tst_expr,tst_rsp,tst_bin_rsp)

    def generate_train_val_split_scope_all(self):
        
        if self.data_source == "GDSC":
            n_s_smpl_val = len(self.s_samples_index) - int(len(self.s_samples_index)*0.7) # nmbr val smpl, possibly here one could introduce a variable to change train/vall split ratio
            n_r_smpl_val = len(self.r_samples_index) - int(len(self.r_samples_index)*0.7)
        if self.data_source == "CCLE":
            n_s_smpl_val = len(self.s_samples_index) - int(len(self.s_samples_index)*0.8) # nmbr val smpl, possibly here one could introduce a variable to change train/vall split ratio
            n_r_smpl_val = len(self.r_samples_index) - int(len(self.r_samples_index)*0.8)
            
        nsmpl = len(self.s_samples_index)+len(self.r_samples_index) # nmbr of all samples
        

        np.random.seed(self.seed)  # this is to make sure the "same random split" is generated all times to compare models
        val_split_s = np.random.choice(a=self.s_samples_index, size=n_s_smpl_val,replace=False)
        np.random.seed(self.seed) 
        val_split_r = np.random.choice(a=self.r_samples_index, size=n_r_smpl_val,replace=False)
        train_split_s = np.array([i for i in self.s_samples_index if i not in val_split_s])
        train_split_r = np.array([i for i in self.r_samples_index if i not in val_split_r])

        train_expr = self.full_data[0].loc[np.concatenate((train_split_s,train_split_r))].sort_index()
        train_cna = self.full_data[1].loc[np.concatenate((train_split_s,train_split_r))].sort_index()
        train_mut = self.full_data[2].loc[np.concatenate((train_split_s,train_split_r))].sort_index()
        train_bin_rsp = self.full_data[3].loc[np.concatenate((train_split_s,train_split_r))].sort_index()
        train_rsp = self.full_data[4].loc[np.concatenate((train_split_s,train_split_r))].sort_index()

        val_expr = self.full_data[0].loc[np.concatenate((val_split_s,val_split_r))].sort_index()
        val_cna = self.full_data[1].loc[np.concatenate((val_split_s,val_split_r))].sort_index()
        val_mut = self.full_data[2].loc[np.concatenate((val_split_s,val_split_r))].sort_index()
        val_bin_rsp = self.full_data[3].loc[np.concatenate((val_split_s,val_split_r))].sort_index()
        val_rsp = self.full_data[4].loc[np.concatenate((val_split_s,val_split_r))].sort_index()

        train_bin_rsp_s = train_bin_rsp.loc[train_split_s] 
        train_rsp_s = train_rsp.loc[train_split_s]

        train_expr_c = train_expr.copy()
        train_mut_c = train_mut.copy()
        train_cna_c = train_cna.copy()


        # a variable to scale oversampling could be introduced here, or just write a dedicated function
        for i in range(1):

            train_expr = pd.concat([train_expr,train_expr_c.loc[train_split_s]])
            train_mut= pd.concat([train_mut,train_mut_c.loc[train_split_s]])
            train_cna = pd.concat([train_cna,train_cna_c.loc[train_split_s]])
            train_bin_rsp = pd.concat([train_bin_rsp,train_bin_rsp_s.loc[train_split_s]])
            train_rsp = pd.concat([train_rsp,train_rsp_s])

        train_expr.index = range(len(train_expr))
        train_cna.index = range(len(train_cna))
        train_mut.index = range(len(train_mut))
        train_bin_rsp.index = range(len(train_bin_rsp))
        train_rsp.index = range(len(train_rsp))

        self.train_data = self.generate_scope_all(train_expr.to_numpy(),train_cna.to_numpy(),train_mut.to_numpy(),train_bin_rsp.to_numpy())
        self.eval_data = self.generate_scope_all(val_expr.to_numpy(),val_cna.to_numpy(),val_mut.to_numpy(),val_bin_rsp.to_numpy())
        
        expr = {"train":train_expr,"val":val_expr}
        cna = {"train":train_cna,"val":val_cna}
        mut = {"train":train_mut,"val":val_mut}
        bin_rsp = {"train":train_bin_rsp,"val":val_bin_rsp}
        rsp = {"train":train_rsp,"val":val_rsp}
        
        self.full_data[0] = expr
        self.full_data[1] = cna
        self.full_data[2] = mut
        self.full_data[3] = bin_rsp
        self.full_data[4] = rsp
        
    def generate_train_val_split_scope_only_expression(self):
    
        if self.data_source == "GDSC":
            n_s_smpl_val = len(self.s_samples_index) - int(len(self.s_samples_index)*0.7) # nmbr val smpl, possibly here one could introduce a variable to change train/vall split ratio
            n_r_smpl_val = len(self.r_samples_index) - int(len(self.r_samples_index)*0.7)
        if self.data_source == "CCLE":
            n_s_smpl_val = len(self.s_samples_index) - int(len(self.s_samples_index)*0.8) # nmbr val smpl, possibly here one could introduce a variable to change train/vall split ratio
            n_r_smpl_val = len(self.r_samples_index) - int(len(self.r_samples_index)*0.8)

        nsmpl = len(self.s_samples_index)+len(self.r_samples_index) # nmbr of all samples


        np.random.seed(self.seed)  # this is to make sure the "same random split" is generated all times to compare models
        val_split_s = np.random.choice(a=self.s_samples_index, size=n_s_smpl_val,replace=False)
        np.random.seed(self.seed) 
        val_split_r = np.random.choice(a=self.r_samples_index, size=n_r_smpl_val,replace=False)
        train_split_s = np.array([i for i in self.s_samples_index if i not in val_split_s])
        train_split_r = np.array([i for i in self.r_samples_index if i not in val_split_r])

        train_expr = self.full_data[0].loc[np.concatenate((train_split_s,train_split_r))].sort_index()
        train_bin_rsp = self.full_data[3].loc[np.concatenate((train_split_s,train_split_r))].sort_index()
        train_rsp = self.full_data[4].loc[np.concatenate((train_split_s,train_split_r))].sort_index()

        val_expr = self.full_data[0].loc[np.concatenate((val_split_s,val_split_r))].sort_index()
        val_bin_rsp = self.full_data[3].loc[np.concatenate((val_split_s,val_split_r))].sort_index()
        val_rsp = self.full_data[4].loc[np.concatenate((val_split_s,val_split_r))].sort_index()

        train_bin_rsp_s = train_bin_rsp.loc[train_split_s] 
        train_rsp_s = train_rsp.loc[train_split_s]

        train_expr_c = train_expr.copy()

        # a variable to scale oversampling could be introduced here
        for i in range(1):
            train_expr= train_expr.append(train_expr_c.loc[train_split_s])
            train_bin_rsp = train_bin_rsp.append(train_bin_rsp_s)
            train_rsp = train_rsp.append(train_rsp_s)

        train_expr.index = range(len(train_expr))
        train_bin_rsp.index = range(len(train_bin_rsp))
        train_rsp.index = range(len(train_rsp))

        self.train_data = self.generate_scope_only_expression(train_expr.to_numpy(),train_bin_rsp.to_numpy())
        self.eval_data = self.generate_scope_only_expression(val_expr.to_numpy(),val_bin_rsp.to_numpy())
        
        expr = {"train":train_expr,"val":val_expr}
        bin_rsp = {"train":train_bin_rsp,"val":val_bin_rsp}
        rsp = {"train":train_rsp,"val":val_rsp}
        
        self.full_data[0] = expr
        self.full_data[3] = bin_rsp
        self.full_data[4] = rsp
        
        
    
    # main function to prepare read in data and split into test/val data with or without oversampling
    def get_input(self):
    
        print("getting "+self.data_source+" data")
        # switch for different input data sources
        
        if self.data_source == "CCLE":
            
            if self.scope == "all":
            
                tv_expr,tv_cna,tv_mut,tv_rsp,tv_bin_rsp,tst_expr,tst_cna,tst_mut,tst_bin_rsp,tst_rsp = self.get_ccle_data(True)
                
                self.full_data[0] = tv_expr
                self.full_data[1] = tv_cna
                self.full_data[2] = tv_mut
                self.full_data[3] = tv_bin_rsp
                self.full_data[4] = tv_rsp
                
                self.generate_train_val_split_scope_all()

                self.full_data[5] = tst_expr
                self.full_data[6] = tst_cna
                self.full_data[7] = tst_mut
                self.full_data[8] = tst_rsp
                self.full_data[9] = tst_bin_rsp
            
            if self.scope == "only_expression":
                
                tv_expr,tv_rsp,tv_bin_rsp,tst_expr,tst_rsp,tst_bin_rsp = self.get_ccle_data(False)
                
                self.full_data[0] = tv_expr
                self.full_data[3] = tv_bin_rsp
                self.full_data[4] = tv_rsp
                
                self.generate_train_val_split_scope_only_expression()
                
                self.full_data[5] = tst_expr
                self.full_data[8] = tst_rsp
                self.full_data[9] = tst_bin_rsp
            
        if self.data_source == "GDSC":
          
            if self.scope == "all":
                tv_expr,tv_cna,tv_mut,tv_rsp,tv_bin_rsp = self.get_gdsc_data(True)
                tst_expr,tst_cna,tst_mut,tst_rsp,tst_bin_rsp = self.get_pdx_data(True)
                
                # selec the genes that overlap between train_val and test set
                expr_genes = tv_expr.columns.intersection(tst_expr.columns)
                cna_genes = tv_cna.columns.intersection(tst_cna.columns)
                mut_genes = tv_mut.columns.intersection(tst_mut.columns)

                #feature selection switch goes here
                if self.select_feat:
                    expr_genes,cna_genes,mut_genes = self.feature_selection_gdsc(tv_expr,tv_cna,tv_mut,tst_expr,tst_cna,tst_mut)
                                                 
                
                else:
                    # selec the genes that overlap between train_val and test set
                    expr_genes = tv_expr.columns.intersection(tst_expr.columns)
                    cna_genes = tv_cna.columns.intersection(tst_cna.columns)
                    mut_genes = tv_mut.columns.intersection(tst_mut.columns)
                
                tv_expr = tv_expr[list(expr_genes)]
                tv_cna = tv_cna[list(cna_genes)]
                tv_mut = tv_mut[list(mut_genes)]
                

                self.full_data[0] = tv_expr
                self.full_data[1] = tv_cna
                self.full_data[2] = tv_mut
                self.full_data[3] = tv_bin_rsp
                self.full_data[4] = tv_rsp

                self.generate_train_val_split_scope_all()

                self.full_data[5] = tst_expr[list(expr_genes)]
                self.full_data[6] = tst_cna[list(cna_genes)]
                self.full_data[7] = tst_mut[list(mut_genes)]
                self.full_data[8] = tst_rsp
                self.full_data[9] = tst_bin_rsp
        
            if self.scope == "only_expression":
            
                tv_expr,tv_rsp,tv_bin_rsp = self.get_gdsc_data(False)
                tst_expr,tst_rsp,tst_bin_rsp = self.get_pdx_data(False)
                
                expr_overlap = tv_expr.columns.intersection(tst_expr.columns)
                #feature selection switch goes here
                tv_expr = tv_expr[list(expr_overlap)]
                
                self.full_data[0] = tv_expr
                self.full_data[3] = tv_bin_rsp
                self.full_data[4] = tv_rsp
                
                self.generate_train_val_split_scope_only_expression()
                
                self.full_data[5] = tst_expr[list(expr_overlap)]
                self.full_data[8] = tst_rsp
                self.full_data[9] = tst_bin_rsp
                
       
                
    def generate_scope_all(self,xprs,cn,mu,b_rsp):
        batch_size=self.bs   
        prefetch = 1
        
        if self.integration=="late":
            output_types = ((tf.float32,)*3,(tf.float32,tf.float32))
            def generator():
                for i in range(xprs.shape[0]):

                    yield (xprs[i],cn[i],mu[i]),(b_rsp[i],b_rsp[i])

            model_input_shape=((self.full_data[0].shape[1],),(self.full_data[1].shape[1],),(self.full_data[2].shape[1],))
            model_triplet_label_shape=(self.full_data[3].shape[1],)
            model_binxentr_shape=(self.full_data[3].shape[1],)
            dataset = tf.data.Dataset.from_generator(
                generator=generator,
                output_types=output_types,
                output_shapes=(model_input_shape, (model_triplet_label_shape,model_binxentr_shape))
            )
            dataset = dataset.repeat().shuffle(
                # use at least dataset size for perfect shuffling and to avoid periodicity in loss decay
                buffer_size=xprs.shape[0],
                seed=None,
                reshuffle_each_iteration=True
                ).batch(batch_size).prefetch(prefetch)
            
        if self.integration=="early":
            output_types = ((tf.float32,)*3,(tf.float32,tf.float32,tf.float32,tf.float32))
            
            def generator():
                for i in range(xprs.shape[0]):
                    

                    yield (xprs[i],cn[i],mu[i]),(b_rsp[i],b_rsp[i],b_rsp[i],b_rsp[i])

            model_input_shape=((self.full_data[0].shape[1],),(self.full_data[1].shape[1],),(self.full_data[2].shape[1],))
            
            
            out_sh = (model_input_shape, (((self.full_data[3].shape[1],),(self.full_data[3].shape[1],),(self.full_data[3].shape[1],)),(self.full_data[3].shape[1],)))
            
            dataset = tf.data.Dataset.from_generator(
                generator=generator,
                output_types=output_types,
                output_shapes=(model_input_shape, ((self.full_data[3].shape[1],),(self.full_data[3].shape[1],),(self.full_data[3].shape[1],),(self.full_data[3].shape[1],)))
            )
            dataset = dataset.repeat().shuffle(
                # use at least dataset size for perfect shuffling and to avoid periodicity in loss decay
                buffer_size=xprs.shape[0],
                seed=None,
                reshuffle_each_iteration=True
                ).batch(batch_size).prefetch(prefetch)

        return dataset
    
    def generate_scope_only_expression(self,xprs,b_rsp):
        batch_size=self.bs   
        prefetch = 1
        
        if self.integration=="late":
            output_types = (tf.float32,(tf.float32,tf.float32))
            def generator():
                for i in range(xprs.shape[0]):

                    yield xprs[i],(b_rsp[i],b_rsp[i])

            model_input_shape=(self.full_data[0].shape[1],)
            model_triplet_label_shape=(self.full_data[3].shape[1],)
            model_binxentr_shape=(self.full_data[3].shape[1],)
            dataset = tf.data.Dataset.from_generator(
                generator=generator,
                output_types=output_types,
                output_shapes=(model_input_shape, (model_triplet_label_shape,model_binxentr_shape))
            )
            dataset = dataset.repeat().shuffle(
                # use at least dataset size for perfect shuffling and to avoid periodicity in loss decay
                buffer_size=xprs.shape[0],
                seed=None,
                reshuffle_each_iteration=True
                ).batch(batch_size).prefetch(prefetch)
            
        if self.integration=="early":
            print("not_implemented, if you want to run only expression, set integration to late")
            
        return dataset
                


    
                
                               

      



