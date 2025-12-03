#setwd(WORKING_DIRECTORY) # set the working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
rm(list=ls())

#### load data
filename = 'radiomics_feature_values.csv'   ## 这里是输入的影像组学特征数据！！
dat <- read.csv(filename,encoding = 'UTF-8')

pre_var <- colnames(dat)[-c(1)]  ##第一列为ID，后续列为各个特征
data=as.matrix(dat[,pre_var])

filename = 'radiomics_coef.csv'
cofs <- read.csv(filename,encoding = 'UTF-8',header = T)
ftcofs = cofs[,2]
ftnames = cofs[,1];ftnames

#### calculate rad_score (各个参数权重coefficient已经由训练集确定，输出为表格csv.因此这里直接读取权重)
radscore = ftcofs[1] # (intercept)，这个参数第一行为常数项，后续行为各个特征的权重
for (i in 2:length(ftnames)) {
  radscore = radscore + (data[, ftnames[i]] * ftcofs[i])
}
ft_c <- cbind(ftnames,ftcofs)
write.csv(ft_c,row.names = F,file = "radscore_feat_coef.csv")
write.csv(radscore,row.names = F,file = "Rad_score.csv")



############# calculate Dose_score  ##########

filename = 'Dose_feature_values.csv' ## 这里是输入的剂量特征数据！！
dat <- read.csv(filename,encoding = 'UTF-8')

pre_var <- colnames(dat)[-c(1)]  ##第一列为ID，后续列为各个特征
data=as.matrix(dat[,pre_var])

filename = 'Dose_coef.csv'
cofs <- read.csv(filename,encoding = 'UTF-8',header = T)
ftcofs = cofs[,2]
ftnames = cofs[,1];ftnames

#### calculate rad_score (各个参数权重coefficient已经由训练集确定，输出为表格csv.因此这里直接读取权重)
radscore = ftcofs[1] # (intercept)，这个参数第一行为常数项，后续行为各个特征的权重
for (i in 2:length(ftnames)) {
  radscore = radscore + (data[, ftnames[i]] * ftcofs[i])
}
ft_c <- cbind(ftnames,ftcofs)
write.csv(ft_c,row.names = F,file = "Dosescore_feat_coef.csv")
write.csv(radscore,row.names = F,file = "Dose_score.csv")
