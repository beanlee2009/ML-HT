setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()


#读取输入文件
rm(list = ls()) 
fname = 'HT2_train_ZZ'
filename = paste(fname,"csv",sep =".")
data <- read.csv(filename,encoding = 'UTF-8')
data$status<-as.factor(data$status)#结局事件因子

## z-score scale数据标准化
x_dat <- as.data.frame(scale(data[,-1]))
dat <- cbind(data$status,x_dat)
colnames(dat)[1] <- "status"
write.csv(dat,file = paste0("./scale_",fname,".csv"),row.names = FALSE)

## 对验证集进行标准化
train_mean <- attr(scale(data[,-1]), "scaled:center")
train_sd <- attr(scale(data[,-1]), "scaled:scale")
fname1 = 'HT2_valid_ZL'
filename1 = paste0(fname1,".csv")
test_data_1 <- read.csv(filename1,encoding = 'UTF-8')
test_scaled_1 <- scale(test_data_1[,-1], center = train_mean, scale = train_sd)
test1 <- cbind(test_data_1$status,test_scaled_1)
colnames(test1)[1] <- "status"
write.csv(test1,file = paste0("./scale_",fname1,".csv"),row.names = FALSE)

fname2 = 'HT2_valid_JL'
filename2 = paste0(fname2,".csv")
test_data_2 <- read.csv(filename2,encoding = 'UTF-8')
test_scaled_2 <- scale(test_data_2[,-1], center = train_mean, scale = train_sd)
test2 <- cbind(test_data_2$status,test_scaled_2)
colnames(test2)[1] <- "status"
write.csv(test2,file = paste0("./scale_",fname2,".csv"),row.names = FALSE)
