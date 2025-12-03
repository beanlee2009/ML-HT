getwd()
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# setwd('E:/SYSUCC/Research/MyCodes/LearnR/nomogram/HT-646')
#=================================LR to select features========================
library(plyr)
#可进行logistic回归的包
library(rms)#可实现逻辑回归模型（lrm）
#1.清理工作环境
rm(list = ls()) 
#2.数据放入工作目录，读取
fname = 'HT2'
filename = paste0(fname,".csv")
dd <- read.csv(filename,encoding = 'UTF-8')


#### 第1步：使用function(x)将glm( )手动提取结果的函数构建循环
Uni_glm_model<- 
  function(x){
    #拟合结局和变量
    FML<-as.formula(paste0("status==1~",x))
    #glm()逻辑回归
    glm1<-glm(FML,data=dd,family = binomial(link = "logit"))
    #提取所有回归结果放入glm2中
    glm2<-summary(glm1)
    #1-计算OR值保留两位小数
    OR<-round(exp(coef(glm1)),2)
    #2-提取SE
    SE<-glm2$coefficients[,2]
    #3-计算CI保留两位小数并合并
    CI5<-round(exp(coef(glm1)-1.96*SE),2)
    CI95<-round(exp(coef(glm1)+1.96*SE),2)
    CI<-paste0(CI5,'-',CI95)
    #4-提取P值
    P<-round(glm2$coefficients[,4],2)
    #5-将变量名、OR、CI、P合并为一个表，删去第一行
    Uni_glm_model <- data.frame('Characteristics'=x,
                                'OR' = OR,
                                'CI' = CI,
                                'P' = P)[-1,]
    #返回循环函数继续上述操作                     
    return(Uni_glm_model)
  }  

###第2步：挑选自己想进行单因素logistic回归的变量带入循环
#把它们放入variable.names中
variable.names<- colnames(dd)[c(2:length(dd))];variable.names 
###第3步：将变量带入循环函数运行，批量输出结果
#变量带入循环函数
Uni_glm <- lapply(variable.names, Uni_glm_model)
#批量输出结果并合并在一起
Uni_glm<- ldply(Uni_glm,data.frame);
#Uni_glm
#最后，将P值=0的变为p<0.0001
Uni_glm$P[Uni_glm$P==0]<-"0.0001"
#Uni_glm

#保存为Excel
oname = paste("单因素回归三线表结果",fname,sep="-")
outname = paste(oname,"csv",sep=".")
write.csv(Uni_glm,outname,fileEncoding='GBK')


####=======================Nomogram modeling=============================
#1.加载R包
#library(rms)
#清理工作环境
#rm(list = ls()) 
#2.载入数据，status=1为有毒性
#aa<- read.csv("E:\\SYSUCC\\Research\\MyCodes\\LearnR\\nomogram\\5010-4A-test2.csv")
#aa<- read.csv("./5010-4A-test2.csv")
sind <- which((Uni_glm$P < 0.05) & (Uni_glm$OR != 1))
sind = sind + 1 #与表格列号对应需+1
aa <- dd [,c(1,sind)]
sname = paste("LR_select_features",fname,sep="-")
selectname = paste(sname,"csv",sep=".")
write.csv(aa,selectname,row.names = F,fileEncoding='GBK')
#3. 查看数据类型，本例结局为status，1为有毒性
str(aa)
##!!!!!!!!!!如果不用LR筛选特征！！！！！！！！！！！
aa <-dd
################ pearson correlation 
# 计算相关矩阵
aa_temp <- aa[,2:length(aa)]
corr_matrix <- cor(aa_temp)
write.csv(corr_matrix,paste0('PCC_',fname,'.csv'),row.names = F,fileEncoding='GBK')
# 设置相关系数阈值
threshold <- 0.7
# 剔除高相关变量
high_cor_vars <- which(corr_matrix > threshold, arr.ind = TRUE)
high_cor_vars <- high_cor_vars[high_cor_vars[,1] != high_cor_vars[,2],]
aa_temp <- aa_temp[,-unique(high_cor_vars[,2])]
# 显示相似矩阵
heatmap(corr_matrix)

status <- aa$status
ss <- cbind(status,aa_temp)
sname = paste("PCC_select_features",fname,sep="-")
selectname = paste(sname,"csv",sep=".")
write.csv(ss,selectname,row.names = F,fileEncoding='GBK')
