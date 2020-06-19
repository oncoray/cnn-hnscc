library(survcomp)
library(compareC)

compute_cindex = function(data, pred_col, time_col, event_col){
  x = data[[pred_col]]
  time = data[[time_col]]
  event = data[[event_col]]
  #weights = rep(1, dim(data)[1])
  #strat = rep(1, dim(data)[1])
  
  cindex = concordance.index(x, surv.time=time, surv.event=event, 
                             #cl=,
                             #weights=weights,
                             #strat=strat,
                             alpha=0.05, method="noether")
  return(cindex)
}

workdir = "~/dev/R/oncoray/"

# C-index for clinical data
clinical_pred_test = read.csv(file.path(workdir, "clinical_model_predictions_ln(GTVtu_from_mask)_zscore_test.csv"))
c_index_clinical_test = compute_cindex(clinical_pred_test, pred_col="pred_per_pat.log_hazard.",
                                       time_col="LRCtime_truth", event_col="LRC_truth")



files = list.files(workdir)
cindices = list()
for(f in files){
  if(!grepl('ensemble_predictions', f)){next}
  p = file.path(workdir, f)
  
  data = read.csv(p)
  pred_col = colnames(data)[2]
  cindex = compute_cindex(data, pred_col=pred_col, time_col="LRCtime", event_col="LRC")
  cindices[[f]] = cindex
  
  l = round(cindex$lower, 2)
  u = round(cindex$upper, 2)
  pval = round(cindex$p.value, 3)
  
  # test if c-index is significantly different from the clinical c-index 
  # using two different approaches
  comp_against_clinical = cindex.comp(cindex, c_index_clinical_test)
  p_clin = round(comp_against_clinical$p.value, 3)
  
  comp_against_clinical_2 = compareC(cindex$data$surv.time, cindex$data$surv.event, cindex$data$x, c_index_clinical_test$data$x)
  p_clin2 = round(comp_against_clinical_2$pval, 3)
  
  print(paste(f, ":", "c=", round(1. - cindex$c.index, 2), "(", 1-u, "-", 1-l, ")", "| p(c==0.5)=", pval, "| Differ from clinical p=", p_clin, p_clin2))
}


# compare if Concordance indices are different between clinical model and dl models on the test cohort
# cindex.comp(c_index_clinical_test, c_index_dl_test)

# a different library to compare c indices

#compareC(timeX=, statusX=clinical_pred_test)