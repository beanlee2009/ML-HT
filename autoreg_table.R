#autoReg不仅可以完成基线表的制作,还可输出回归分析（支持线性模型和比例风险模型）的表格
getwd()
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# install.packages("devtools") 
# install.packages("remotes") # 如果devtools包是旧有的，可能需要更新，否则有可能报错
# library(remotes) 
# update(package_deps("devtools")) #更新devtools的依赖包
# devtools::install_github("cardiomoon/autoReg") #从github上下载autoReg包
# install.package("survival") #获取survival包中内置示例数据

library(autoReg)
# library(survival)

#读取输入文件
rm(list = ls()) 
fname = 'HT2_zz645_select20250513'
filename = paste(fname,"csv",sep =".")
data <- read.csv(filename,encoding = 'UTF-8')
data$status<-as.factor(data$status)#结局事件因子

## z-score scale数据标准化
x_dat <- as.data.frame(scale(data[,-1]))
dat <- cbind(data$status,x_dat)
colnames(dat)[1] <- "status"
write.csv(dat,file = paste0("./scale_",fname,".csv"))
# dat <- na.omit(data)


baseline_table1=gaze(status~.,data=dat) 
print(baseline_table1)


fit=glm(status~.,data = dat, family = "binomial")
autoReg(fit) #只显示多因素回归

fit2=glm(status~.,data = dat, family = "binomial")
autoReg(fit2, uni=TRUE) #uni=TRUE, 显示单因素
#先进行单因素挑选统计意义显著的解释变量进入多因素分析

#当然也可以设定所有的因素全部进入多变量回归分析,设置参数threshold=1
autoReg(fit2, uni=TRUE, threshold=1) %>% myft()  #myft()函数生成发表级别图片

#install.packages("rrtable")
library(rrtable)
result=autoReg(fit2, uni=TRUE, threshold=1) %>% myft()
table2pptx(result)  #导出到ppt，可编辑数据
table2docx(result)  #导出到docx，可编辑数据
#多因素回归统计森林图
modelPlot(fit2)
#modelPlot(fit2,uni=TRUE,threshold=1,show.ref=FALSE)

#将图片导出至ppt编辑
p1=modelPlot(fit2)
rrtable::plot2pptx(print(p1))
