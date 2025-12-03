# === 15种机器学习-DCA-SHAP解释-性能折线-ROC诊断-AUC森林图-临床影响曲线-校准曲线-残差图 ===
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
library(caret)    # 机器学习框架
library(pROC)     # ROC曲线分析
library(ggplot2)  # 可视化
library(shapviz)  # SHAP 分析
library(kernelshap)
library(dplyr)    # 数据处理
library(rmda)
# library(catboost)
library(lightgbm)
library(DALEX)
#1.清理工作环境
rm(list = ls()) 
# === 用户自定义区域 ============================================
# 1. 数据加载
# train_data <- read.csv('train_data.csv', header = T, row.names = 1)
# test_data <- read.csv('test_data.csv', header = T, row.names = 1)

train_data <- read.csv('train_HT2_ZZ.csv',encoding = 'UTF-8')

test_data <- read.csv('test_HT2_ZZ.csv', encoding = 'UTF-8')

# 2. 指定响应变量名称
response_var <- "Group"

# 预处理标签
# 这里很多朋友都有疑问，这是在干吗？其实只是从名字里提取分类标签。
# 如果你的分类信息标签不在名字里的话，就直接赋值就可以了。
# 训练集最后一列是分类变量，正常设定为"Control", 疾病设定为"DN”，一定要是DN，
#  否在后面也得跟着改，所以别设置成其他的。
# 训练集最后一列不要有标签。把分类信息给到test_group就好了，保证只有0 和 1.
# 
# train_data$Group <- gsub("(.+)\\_(.+)\\_(.+)", '\\3', rownames(train_data))
# table(train_data$Group)
# test_group <- gsub("(.+)\\_(.+)\\_(.+)", '\\3', rownames(test_data))

train_data$Group <-ifelse(train_data$Group =='X0','Control','DN')
test_data$Group <-ifelse(test_data$Group =='X0','Control','DN')
test_group  <- test_data$Group
test_group  <- ifelse(test_group  == 'Control', '0','1')
table(test_group)
set.seed(12345)


# 3. 设置交叉验证参数。样本越少，交叉折数越少，否则会报错。

cv_control <- trainControl(
  method = "repeatedcv",
  number = 5,
  savePredictions = "final",
  classProbs = TRUE
)

# === 性能指标计算函数 ==========================================
calculate_metrics <- function(truth, probs, threshold = 0.5) {
  # 将概率转换为预测标签
  pred <- factor(ifelse(probs >= threshold, "Positive", "Negative"), 
                 levels = c("Negative", "Positive"))
  
  # 转换真实标签为因子
  truth_fct <- factor(ifelse(truth == 1, "Positive", "Negative"), 
                      levels = c("Negative", "Positive"))
  
  # 计算混淆矩阵
  cm <- confusionMatrix(pred, truth_fct, positive = "Positive")
  
  # 提取各项指标
  metrics <- c(
    cm$byClass["Sensitivity"],  # 灵敏度 = TP/(TP+FN)
    cm$byClass["Specificity"],  # 特异度 = TN/(TN+FP)
    cm$overall["Accuracy"],       # 准确度 = (TP+TN)/(TP+FP+TN+FN)
    cm$byClass["Pos Pred Value"],     # 阳性预测值 = TP/(TP+FP)
    cm$byClass["Neg Pred Value"],      # 阴性预测值 = TN/(TN+FN)
    cm$byClass["F1"]     
  )
  metrics["Youden"] <- metrics["Sensitivity"] + metrics["Specificity"] - 1  # 约登指数
  return(metrics)
}
# ===============================================================



# 验证数据格式
if(!response_var %in% colnames(train_data)) {
  stop(paste("响应变量", response_var, "不存在于训练数据中"))
}

# 确保响应变量为因子
train_data[[response_var]] <- factor(train_data[[response_var]], 
                                     levels = c("Control", "DN"))
formula <- as.formula(paste(response_var, "~ ."))

## 13种模型列表（添加了XGBoost）
model_settings <- data.frame(
  AlgorithmName = c("RandomForest", "GradientBoosting", "SVM_Kernel",
                    "LogisticModel", "NeighborMethod","PLSModel", 
                    "BoostingMethod", "NeuralNet", "BayesMethod", 
                    "DiscriminantModel",'Lasso',"AdaptiveBoosting", "XGBoost"),
  Implementation = c("rf", "xgbTree", "svmRadial", 
                     "glm", "knn","pls", 
                     "gbm", "nnet", "nb", 
                     "lda",'glmnet',"AdaBoost.M1", "xgbLinear")
)

# 创建存储结果的列表

# 模型性能信息储存：
train_metrics <- list()
test_metrics <- list()

# 模型ROC信息储存：
train_roc_list <- list()  
test_roc_list <- list() 

modelContainer <- list()
AUCresults <- c()
training_times <- numeric()

# 为DCA创建存储预测概率的列表
train_probs_list <- list()
test_probs_list <- list()

# 在模型训练循环中增加残差计算列表
residuals_train_list <- list()
residuals_test_list <- list()

#2025-07-30更新，添加记录网格超参数的列表。
best_tune_params <- list() # 记录网格超参数
# 开始总计时
total_start <- Sys.time()

cat("===== 开始模型训练 ", format(total_start, "%Y-%m-%d %H:%M:%S"), " =====\n")

# 模型训练和评估
for (idx in seq_len(nrow(model_settings))) {
  algoName <- model_settings$AlgorithmName[idx]
  algoImpl <- model_settings$Implementation[idx]
  
  # 开始模型计时
  start_time <- Sys.time()
  cat(sprintf("\n[%d/%d] 开始训练 %s: %s\n", idx, nrow(model_settings), algoName, format(start_time, "%H:%M:%S")))
  
  tryCatch({
    # 特殊模型的注意事项
    slow_models <- c("SVM_Kernel", "NeuralNet", "GradientBoosting", "AdaptiveBoosting", "XGBoost")
    if (algoName %in% slow_models) {
      cat("  ! 注意: 此模型可能需要较长时间...\n")
    }
    
    # 模型训练
    cat("  - 开始模型训练...\n")
    
    # 根据不同算法应用特定设置
    if (algoName == "SVM_Kernel") {
      # SVM 调优
      svm_tuneGrid <- expand.grid(
        sigma = c(0.01, 0.1, 1),
        C = c(0.25, 0.5, 1, 2)
      )
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   prob.model = TRUE, trControl = cv_control,
                                   tuneGrid = svm_tuneGrid)
    } else if (algoName == "GradientBoosting") {
      # XGBoost 调优
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = expand.grid(nrounds=100, max_depth=3, eta=0.1,
                                                          gamma=0, colsample_bytree=0.7,
                                                          min_child_weight=1, subsample=0.7))
    } else if (algoName == "RandomForest") {
      # 随机森林调优
      rf_tuneGrid <- expand.grid(mtry = c(2, 4, 6, 8, 10))
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = rf_tuneGrid)
    } else if (algoName == "NeighborMethod") {
      # KNN 调优参数
      knn_tuneGrid <- expand.grid(k = c(3, 5, 7, 9, 11, 13, 15))
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = knn_tuneGrid)
    } else if (algoName == "PLSModel") {
      # PLS 调优参数
      pls_tuneGrid <- expand.grid(ncomp = c(1, 2, 3, 4, 5))
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = pls_tuneGrid)
    } else if (algoName == "BoostingMethod") {
      # GBM 调优参数
      gbm_tuneGrid <- expand.grid(
        n.trees = c(50, 100, 150),
        interaction.depth = c(1, 3, 5),
        shrinkage = c(0.01, 0.1),
        n.minobsinnode = c(5, 10)
      )
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = gbm_tuneGrid)
    } else if (algoName == "NeuralNet") {
      # 神经网络调优参数
      nnet_tuneGrid <- expand.grid(
        size = c(3, 5, 7),
        decay = c(0, 0.001, 0.01)
      )
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = nnet_tuneGrid)
    } else if (algoName == "BayesMethod") {
      # 朴素贝叶斯调优参数
      nb_tuneGrid <- expand.grid(
        fL = c(0, 0.5, 1),
        usekernel = c(TRUE, FALSE),
        adjust = c(0.5, 1, 1.5)
      )
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = nb_tuneGrid)
    } else if (algoName == "Lasso") {
      # Lasso/弹性网络调优参数
      glmnet_tuneGrid <- expand.grid(
        alpha = c(0, 0.5, 1),  # 0=岭回归, 1=lasso, 0.5=弹性网络
        lambda = c(0.001, 0.01, 0.1, 1)
      )
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = glmnet_tuneGrid)
    } else if (algoName == "AdaptiveBoosting") {
      # AdaBoost调优参数
      adaboost_tuneGrid <- expand.grid(
        mfinal = c(50, 100, 150),  # 树的数量
        maxdepth = c(1, 3, 5),     # 树的最大深度
        coeflearn = "Breiman"      # 系数学习算法
      )
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = adaboost_tuneGrid)
    } else if (algoName == "XGBoost") {
      # XGBoost线性模型调优参数
      xgb_tuneGrid <- expand.grid(
        nrounds = c(50, 100, 150),  # boosting迭代次数
        lambda = c(0, 0.1, 1),      # L2正则化
        alpha = c(0, 0.1, 1),       # L1正则化
        eta = c(0.01, 0.1, 0.3)     # 学习率
      )
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = xgb_tuneGrid)
    } else {
      # 对于没有调优参数的模型（如glm, lda，正常运行就可以）
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control)
    }
    
    # 计算训练时间
    model_time <- round(as.numeric(difftime(Sys.time(), start_time, units = "secs")), 1)
    cat(sprintf("  - 训练完成 (耗时: %s秒)\n", model_time))
    
    # +++ 提取并输出最佳参数 +++
    cat("  - 最佳参数组合:\n")
    print(trainedModel$bestTune)
    
    # 存储最佳参数
    best_tune_params[[algoName]] <- trainedModel$bestTune
    
    # 添加评估信息
    cat("  - 开始模型评估...\n")
    
    # === 训练集性能指标 =========================================
    # 获取交叉验证预测概率
    cv_probs <- trainedModel$pred[trainedModel$pred$rowIndex, "DN"]
    cv_indices <- trainedModel$pred[trainedModel$pred$rowIndex, "rowIndex"]
    
    # 提取真实标签
    truth_train <- ifelse(train_data[[response_var]][cv_indices] == "DN", 1, 0)
    
    # 计算训练集指标
    train_metrics[[algoName]] <- calculate_metrics(truth_train, cv_probs)
    
    # 创建训练集ROC曲线对象
    roc_train <- roc(truth_train, cv_probs)
    train_roc_list[[algoName]] <- roc_train
    
    # === 测试集性能指标 =========================================
    test_probs <- predict(trainedModel, newdata = test_data, type = "prob")[, "DN"]
    
    # 计算测试集指标
    test_metrics[[algoName]] <- calculate_metrics(test_group, test_probs)
    
    # 创建测试集ROC曲线对象
    roc_test <- roc(test_group, test_probs)
    test_roc_list[[algoName]] <- roc_test
    
    # 存储模型/AUC结果
    modelContainer[[algoImpl]] <- trainedModel
    AUCresults <- c(AUCresults, paste0(algoName, ": ", 
                                       sprintf("%.03f", roc_test$auc)))  # 保存AUC
    
    # 为DCA存储预测概率
    # 训练集：创建完整概率向量
    train_probs_full <- rep(NA, nrow(train_data))
    train_probs_full[trainedModel$pred$rowIndex] <- trainedModel$pred$DN
    train_probs_list[[algoName]] <- train_probs_full
    
    # 测试集：直接存储
    test_probs_list[[algoName]] <- test_probs
    
    # 性能输出
    cat(sprintf("  - 训练集AUC: %.3f\n", auc(roc_train)))
    cat(sprintf("  - 测试集AUC: %.3f\n", auc(roc_test)))
    
    # 记录训练时间
    training_times[algoName] <- model_time
    
  }, error = function(e) {
    # 记录错误信息和耗时
    error_time <- round(as.numeric(difftime(Sys.time(), start_time, units = "secs")), 1)
    cat(sprintf("  ! 错误: %s (耗时: %s秒)\n", e$message, error_time))
    
    # 记录错误信息
    train_metrics[[algoName]] <- rep(NA, 7)
    test_metrics[[algoName]] <- rep(NA, 7)
    training_times[algoName] <- error_time
    # 存储NA表示参数提取失败
    best_tune_params[[algoName]] <- NA
  })
  
  # 显示单个模型完成状态
  cat(sprintf("[%d/%d] 完成 %s | 累计耗时: %.1f秒\n", idx, nrow(model_settings), algoName, 
              as.numeric(difftime(Sys.time(), total_start, units = "secs"))))
  
  # 添加分隔线
  cat("-----------------------------------------\n")
}
# 最终报告
total_time <- round(as.numeric(difftime(Sys.time(), total_start, units = "mins")), 1)
cat("\n===== 13种模型训练完成! 总耗时: ", total_time, "分钟 =====\n")

# 2个特殊的模型需要单独训练。
# === 在模型训练循环之后，单独训练CatBoost模型 ========================
# cat("\n===== 开始单独训练 CatBoost 模型 =====\n")
# 
# # 1. 准备CatBoost所需的数据格式
# train_features <- train_data[, -which(names(train_data) == response_var)]
# train_labels <- ifelse(train_data[[response_var]] == "DN", 1, 0)
# 
# test_features <- test_data
# test_labels <- test_group  # 已经在前面预处理为0/1
# 
# # 创建CatBoost Pool对象
# train_pool <- catboost.load_pool(data = train_features, label = train_labels)
# test_pool <- catboost.load_pool(data = test_features, label = as.numeric(test_labels))
# 
# # 2. 定义参数网格
# catboost_grid <- expand.grid(
#   depth = c(4, 6, 8),          # 树深度
#   learning_rate = c(0.01, 0.05, 0.1),  # 学习率
#   l2_leaf_reg = c(1, 3, 5)     # L2正则化系数
# )
# 
# # 初始化最佳参数和性能
# best_auc <- 0
# best_params <- NULL
# best_iter <- NULL
# 
# # 3. 网格搜索
# cat("-- 开始网格搜索寻找最佳参数 --\n")
# for (i in 1:nrow(catboost_grid)) {
#   params <- as.list(catboost_grid[i, ])
#   
#   # 设置基本参数
#   cv_params <- list(
#     iterations = 500,
#     learning_rate = params$learning_rate,
#     depth = params$depth,
#     l2_leaf_reg = params$l2_leaf_reg,
#     loss_function = 'Logloss',
#     eval_metric = 'AUC',
#     early_stopping_rounds = 50,
#     random_seed = 42
#   )
#   
#   # 执行交叉验证
#   cv_results <- catboost.cv(
#     pool = train_pool,
#     params = cv_params,
#     fold_count = 5,
#     partition_random_seed = 42,
#     shuffle = TRUE
#   )
#   
#   # 获取最佳迭代次数和AUC
#   iter <- which.max(cv_results$test.AUC.mean)
#   auc_value <- max(cv_results$test.AUC.mean)
#   
#   cat(sprintf("参数组合 %d/%d: depth=%d, lr=%.3f, l2=%.1f | 最佳迭代: %d, AUC: %.4f\n",
#               i, nrow(catboost_grid), params$depth, params$learning_rate, 
#               params$l2_leaf_reg, iter, auc_value))
#   
#   # 更新最佳参数
#   if (auc_value > best_auc) {
#     best_auc <- auc_value
#     best_params <- cv_params
#     best_iter <- iter
#   }
# }
# 
# # 4. 使用最佳参数训练最终模型
# cat(sprintf("\n最佳参数: depth=%d, lr=%.3f, l2=%.1f | 最佳迭代: %d, AUC: %.4f\n",
#             best_params$depth, best_params$learning_rate, 
#             best_params$l2_leaf_reg, best_iter, best_auc))
# # 将CatBoost的最佳参数添加到参数汇总列表
# catboost_best_params <- data.frame(
#   depth = best_params$depth,
#   learning_rate = best_params$learning_rate,
#   l2_leaf_reg = best_params$l2_leaf_reg,
#   iterations = best_iter
# )
# best_tune_params[["CATBoost"]] <- catboost_best_params
# 
# # 设置最终模型参数
# catboost_params <- best_params
# catboost_params$iterations <- best_iter
# 
# # 5. 训练CatBoost模型并计时
# start_time_catboost <- Sys.time()
# cat(sprintf("开始训练 CatBoost: %s\n", format(start_time_catboost, "%H:%M:%S")))
# 
# catboost_model <- catboost.train(
#   learn_pool = train_pool,
#   test_pool = test_pool,
#   params = catboost_params
# )
# 
# # 计算训练时间
# catboost_train_time <- round(as.numeric(difftime(Sys.time(), start_time_catboost, units = "secs")), 1)
# cat(sprintf("CatBoost 训练完成 (耗时: %s秒)\n", catboost_train_time))
# 
# # 4. 获取预测概率
# # 训练集预测概率
# train_catboost_preds <- catboost.predict(catboost_model, train_pool, prediction_type = "Probability")
# # 测试集预测概率
# test_catboost_preds <- catboost.predict(catboost_model, test_pool, prediction_type = "Probability")
# 
# # 5. 计算性能指标并整合到结果列表中
# algoName <- "CATBoost"
# 
# # 训练集指标
# train_truth <- ifelse(train_data[[response_var]] == "DN", 1, 0)
# train_metrics[[algoName]] <- calculate_metrics(train_truth, train_catboost_preds)
# 
# # 测试集指标
# test_metrics[[algoName]] <- calculate_metrics(as.numeric(test_labels), test_catboost_preds)
# 
# # 创建ROC对象
# roc_train_catboost <- roc(train_truth, train_catboost_preds)
# roc_test_catboost <- roc(as.numeric(test_labels), test_catboost_preds)
# 
# # 添加到ROC列表
# train_roc_list[[algoName]] <- roc_train_catboost
# test_roc_list[[algoName]] <- roc_test_catboost
# 
# # 添加到模型容器和AUC结果
# modelContainer[["catboost"]] <- catboost_model
# AUCresults <- c(AUCresults, paste0(algoName, ": ", sprintf("%.03f", roc_test_catboost$auc)))
# 
# # 添加到概率列表（用于DCA）
# train_probs_list[[algoName]] <- train_catboost_preds
# test_probs_list[[algoName]] <- test_catboost_preds
# 
# # === 添加CatBoost残差计算 ===
# # 训练集残差 = 实际值 - 预测概率
# train_residuals_catboost <- train_truth - train_catboost_preds
# # 测试集残差 = 实际值 - 预测概率
# test_residuals_catboost <- as.numeric(test_labels) - test_catboost_preds
# 
# # 存储结果
# residuals_train_list[[algoName]] <- train_residuals_catboost
# residuals_test_list[[algoName]] <- test_residuals_catboost
# 
# # 记录训练时间
# training_times[algoName] <- catboost_train_time
# 
# # 性能输出
# cat(sprintf("  - CatBoost 训练集AUC: %.3f\n", auc(roc_train_catboost)))
# cat(sprintf("  - CatBoost 测试集AUC: %.3f\n", auc(roc_test_catboost)))
# 

# === 在模型训练循环之后，单独训练LightGBM模型 ======================
cat("\n===== 开始单独训练 LightGBM 模型 =====\n")
# 1. 准备LightGBM所需的数据格式
train_features_lgb <- train_data[, -which(names(train_data) == response_var)]
train_labels_lgb <- ifelse(train_data[[response_var]] == "DN", 1, 0)

test_features_lgb <- test_data[, -which(names(test_data) == response_var)]
test_labels_lgb <- as.numeric(test_group)

# 处理分类变量
categorical_cols <- names(train_features_lgb)[sapply(train_features_lgb, is.factor)]

# 转换为矩阵格式
train_matrix_lgb <- as.matrix(train_features_lgb)
test_matrix_lgb <- as.matrix(test_features_lgb)

# 2. 创建LightGBM数据集
dtrain_lgb <- lgb.Dataset(
  data = train_matrix_lgb,
  label = train_labels_lgb,
  categorical_feature = categorical_cols
)

dtest_lgb <- lgb.Dataset(
  data = test_matrix_lgb,
  label = test_labels_lgb,
  reference = dtrain_lgb,
  categorical_feature = categorical_cols
)

# 3. 定义参数网格
lgb_grid <- expand.grid(
  num_leaves = c(15, 31, 63),          # 叶子节点数量
  learning_rate = c(0.01, 0.05, 0.1),  # 学习率
  min_data_in_leaf = c(5, 10, 20),     # 叶子节点最小数据量 
  lambda_l2 = c(0, 0.1, 0.5),          # L2正则化系数
  feature_pre_filter= F
)

# 初始化最佳参数和性能
best_auc <- 0
best_params <- NULL
best_iter <- NULL

# 4. 网格搜索
cat("-- 开始网格搜索寻找最佳参数 --\n")
for (i in 1:nrow(lgb_grid)) {
  params <- as.list(lgb_grid[i, ])
  
  lgb_params <- list(
    objective = "binary",
    metric = "auc",
    num_leaves = params$num_leaves,
    learning_rate = params$learning_rate,
    min_data_in_leaf = params$min_data_in_leaf,
    lambda_l2 = params$lambda_l2,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq = 5,
    verbosity = -1,
    seed = 42,
    feature_pre_filter = FALSE  # 关键设置在这里
  )
  
  # 交叉验证
  cv_model <- lgb.cv(
    params = lgb_params,
    data = dtrain_lgb,
    nrounds = 500,
    nfold = 5,
    stratified = TRUE,
    early_stopping_rounds = 50,
    eval_freq = 10
  )
  
  # 获取最佳迭代次数和AUC
  iter <- cv_model$best_iter
  auc_value <- max(unlist(cv_model$record_evals$valid$auc$eval))
  
  cat(sprintf("参数组合 %d/%d: leaves=%d, lr=%.3f, min_data=%d, l2=%.1f | 最佳迭代: %d, AUC: %.4f\n",
              i, nrow(lgb_grid), params$num_leaves, params$learning_rate, 
              params$min_data_in_leaf, params$lambda_l2, iter, auc_value))
  
  # 更新最佳参数
  if (auc_value > best_auc) {
    best_auc <- auc_value
    best_params <- lgb_params
    best_iter <- iter
  }
}

# 5. 使用最佳参数训练最终模型
cat(sprintf("\n最佳参数: leaves=%d, lr=%.3f, min_data=%d, l2=%.1f | 最佳迭代: %d, AUC: %.4f\n",
            best_params$num_leaves, best_params$learning_rate, 
            best_params$min_data_in_leaf, best_params$lambda_l2, best_iter, best_auc))

# 将LightGBM的最佳参数添加到参数汇总列表
lightgbm_best_params <- data.frame(
  num_leaves = best_params$num_leaves,
  learning_rate = best_params$learning_rate,
  min_data_in_leaf = best_params$min_data_in_leaf,
  lambda_l2 = best_params$lambda_l2,
  nrounds = best_iter
)
best_tune_params[["LightGBM"]] <- lightgbm_best_params

# 6. 训练LightGBM模型并计时
start_time_lgb <- Sys.time()
cat(sprintf("开始训练 LightGBM: %s\n", format(start_time_lgb, "%H:%M:%S")))

lgb_model <- lgb.train(
  params = lgb_params,
  data = dtrain_lgb,
  valids = list(test = dtest_lgb),
  nrounds = 500,
  early_stopping_rounds = 50,
  eval_freq = 10
)

# 计算训练时间
lgb_train_time <- round(as.numeric(difftime(Sys.time(), start_time_lgb, units = "secs")), 1)
cat(sprintf("LightGBM 训练完成 (耗时: %s秒)\n", lgb_train_time))

# 7. 获取预测概率
# 训练集预测概率
train_lgb_preds <- predict(lgb_model, train_matrix_lgb)
# 测试集预测概率
test_lgb_preds <- predict(lgb_model, test_matrix_lgb)

# 8. 计算性能指标并整合到结果列表中
algoName <- "LightGBM"

# 训练集指标
train_metrics[[algoName]] <- calculate_metrics(train_labels_lgb, train_lgb_preds)

# 测试集指标
test_metrics[[algoName]] <- calculate_metrics(test_labels_lgb, test_lgb_preds)

# 创建ROC对象
roc_train_lgb <- roc(train_labels_lgb, train_lgb_preds)
roc_test_lgb <- roc(test_labels_lgb, test_lgb_preds)

# 添加到ROC列表
train_roc_list[[algoName]] <- roc_train_lgb
test_roc_list[[algoName]] <- roc_test_lgb

# 添加到模型容器和AUC结果
modelContainer[["lightgbm"]] <- lgb_model
AUCresults <- c(AUCresults, paste0(algoName, ": ", sprintf("%.03f", roc_test_lgb$auc)))

# 添加到概率列表（用于DCA）
train_probs_list[[algoName]] <- train_lgb_preds
test_probs_list[[algoName]] <- test_lgb_preds

# === 添加LightGBM残差计算 ===
# 训练集残差 = 实际值 - 预测概率
train_residuals_lgb <- train_labels_lgb - train_lgb_preds
# 测试集残差 = 实际值 - 预测概率
test_residuals_lgb <- test_labels_lgb - test_lgb_preds

# 存储结果
residuals_train_list[[algoName]] <- train_residuals_lgb
residuals_test_list[[algoName]] <- test_residuals_lgb

# 记录训练时间
training_times[algoName] <- lgb_train_time

# 性能输出
cat(sprintf("  - LightGBM 训练集AUC: %.3f\n", auc(roc_train_lgb)))
cat(sprintf("  - LightGBM 测试集AUC: %.3f\n", auc(roc_test_lgb)))



# 显示训练时间表
cat("\n模型训练时间汇总:\n")
print(data.frame(
  Model = names(training_times),
  Train_time_seconds = unlist(training_times)
))

# 第一道菜：建模耗时统计
write.csv(data.frame(
  Model = names(training_times),
  Train_time_seconds = unlist(training_times)
),'1.1_ML_training_time_record.csv')


# 创建最佳参数汇总表格
best_params_df <- data.frame(
  Model = character(),
  Parameter = character(),
  Value = character(),
  stringsAsFactors = FALSE
)

# 遍历所有模型的最佳参数
for (model_name in names(best_tune_params)) {
  params <- best_tune_params[[model_name]]
  
  if (!is.null(params) && nrow(params) > 0) {
    for (param_name in names(params)) {
      new_row <- data.frame(
        Model = model_name,
        Parameter = param_name,
        Value = as.character(params[[param_name]])
      )
      best_params_df <- rbind(best_params_df, new_row)
    }
  } else {
    new_row <- data.frame(
      Model = model_name,
      Parameter = "No tuning parameters",
      Value = ""
    )
    best_params_df <- rbind(best_params_df, new_row)
  }
}

# 保存为CSV文件
write.csv(best_params_df, "1.2_Best_Tuning_Parameters.csv", row.names = FALSE)


# 如果你之前删过模型，这里也要把相应的颜色给删掉。
# 准备模型和颜色的对应关系
color_mapping <- c(
  AdaptiveBoosting = "#DC143C",
  BayesMethod = "#20B2AA",
  BoostingMethod = "#FFA500",
  # CATBoost = 'green',
  DiscriminantModel = "#9370DB",
  GradientBoosting = "#98FB98",
  Lasso = "#F08080",
  LightGBM = 'purple',
  LogisticModel = "#1E90FF",
  NeighborMethod = "#FFFF00", 
  NeuralNet = "#0000FF",
  PLSModel = "#808000",
  RandomForest = "#FF00FF",
  SVM_Kernel = "#FA8072"
)

allcolour <- data.frame(color_mapping)$color_mapping

# 第2道菜：
# === 性能指标表格 ==============================================
library(gridExtra)
library(grid)

# 整理训练集指标为数据框
train_results <- bind_rows(train_metrics, .id = "Model")

# 重命名训练集指标表格
colnames(train_results) <- c("Models", 
                             "Sensitivity", "Specificity", "Accuracy", 
                             "PPV", "NPV", 'F1', "Youden's index")

# 保存训练集性能指标
write.csv(train_results, "2a_Train_Performance_Metrics.csv", row.names = FALSE)

# 创建训练集性能表格PDF
pdf("2a_Train_Performance_Table.pdf", width = 12, height = 10)
grid.table(train_results, 
           rows = NULL,
           theme = ttheme_default(
             core = list(bg_params = list(fill = c("#F7F7F7", "#FFFFFF"), col = "gray"),
                         fg_params = list(cex = 0.8)),
             colhead = list(fg_params = list(cex = 0.9, fontface = "bold"))
           ))
dev.off()

# === 绘制训练集指标折线图 ===
library(ggplot2)
library(tidyr)

# 将数据转换为长格式
train_long <- gather(train_results, key = "Metric", value = "Value", -Models)

# 创建训练集指标折线图
mdsize <- nrow(train_results)
train_plot <- ggplot(train_long, aes(x = Metric, y = Value, group = Models, color = Models)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  labs(title = "Training Set Performance Metrics", 
       x = "Performance Metrics", 
       y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = mdsize),
        axis.text.y = element_text(size = mdsize),
        legend.position = "bottom",
        legend.title = element_text(size = mdsize, face = "bold"),
        legend.text = element_text(size = mdsize),
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold")) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(values = color_mapping)
train_plot 
# 保存训练集指标折线图
ggsave("3a_Train_Metrics_LinePlot.pdf", train_plot, width = 10, height = 8)
dev.off()
# 整理测试集指标为数据框
test_results <- bind_rows(test_metrics, .id = "Model")

# 重命名测试集指标表格
colnames(test_results) <- c("Models", 
                            "Sensitivity", "Specificity", "Accuracy", 
                            "PPV", "NPV", 'F1', "Youden's index")

# 保存测试集性能指标
write.csv(test_results, "2b_Test_Performance_Metrics.csv", row.names = FALSE)

# 创建测试集性能表格PDF
pdf("2b_Test_Performance_Table.pdf", width = 12, height = 10)
grid.table(test_results, 
           rows = NULL,
           theme = ttheme_default(
             core = list(bg_params = list(fill = c("#F7F7F7", "#FFFFFF"), col = "gray"),
                         fg_params = list(cex = 0.8)),
             colhead = list(fg_params = list(cex = 0.9, fontface = "bold"))
           ))
dev.off()



test_long <- gather(test_results, key = "Metric", value = "Value", -Models)

# 创建测试集指标折线图
test_plot <- ggplot(test_long, aes(x = Metric, y = Value, group = Models, color = Models)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  labs(title = "Validation Set Performance Metrics", 
       x = "Performance Metrics", 
       y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = mdsize),
        axis.text.y = element_text(size = mdsize),
        legend.position = "bottom",
        legend.title = element_text(size = mdsize, face = "bold"),
        legend.text = element_text(size = mdsize),
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold")) +
  scale_y_continuous(limits = c(-1, 1)) +
  scale_color_manual(values = color_mapping)
test_plot 
# 保存测试集指标折线图
ggsave("3b_Test_Metrics_LinePlot.pdf", test_plot, width = 10, height = 8)
dev.off()

# === AUC置信区间 ==============================================
# 计算训练集AUC置信区间
train_auc_ci <- data.frame()
for (model_name in names(train_roc_list)) {
  if (!is.null(train_roc_list[[model_name]])) {
    ci <- ci.auc(train_roc_list[[model_name]])
    train_auc_ci <- rbind(train_auc_ci, data.frame(
      Model = model_name,
      AUC = auc(train_roc_list[[model_name]]),
      CI_lower = ci[1],
      CI_upper = ci[3]
    ))
  }
}

# 格式化训练集AUC和置信区间
train_auc_ci$AUC_CI <- sprintf("%.3f (%.3f-%.3f)", 
                               train_auc_ci$AUC,
                               train_auc_ci$CI_lower,
                               train_auc_ci$CI_upper)

# 保存训练集AUC置信区间结果
write.csv(train_auc_ci, "4a_Train_AUC_Confidence_Intervals.csv", row.names = FALSE)

# 计算测试集AUC置信区间
test_auc_ci <- data.frame()
for (model_name in names(test_roc_list)) {
  if (!is.null(test_roc_list[[model_name]])) {
    ci <- ci.auc(test_roc_list[[model_name]])
    test_auc_ci <- rbind(test_auc_ci, data.frame(
      Model = model_name,
      AUC = auc(test_roc_list[[model_name]]),
      CI_lower = ci[1],
      CI_upper = ci[3]
    ))
  }
}

# 格式化测试集AUC和置信区间
test_auc_ci$AUC_CI <- sprintf("%.3f (%.3f-%.3f)", 
                              test_auc_ci$AUC,
                              test_auc_ci$CI_lower,
                              test_auc_ci$CI_upper)

# 保存测试集AUC置信区间结果
write.csv(test_auc_ci, "4b_Test_AUC_Confidence_Intervals.csv", row.names = FALSE)


# 第3道菜：AUC森林图
# === 森林图绘制: AUC比较 ======================================
# 准备森林图所需数据
train_forest_df <- train_auc_ci %>%
  arrange(AUC)%>% # 按AUC升序排列
  mutate(
    Model = factor(Model, levels = rev(Model)),  # 反转模型顺序，使首项在上
    AUC_label = sprintf("%.3f (%.3f-%.3f)", AUC, CI_lower, CI_upper)
  ) 




# 创建森林图
ggplot(train_forest_df, aes(x = AUC, y = Model)) +
  geom_point(aes(color = Model), size = 5, shape = 18) +  # 菱形点表示AUC值
  geom_errorbarh(aes(xmin = CI_lower, xmax = CI_upper, color = Model),
                 height = 0.1, size = 1.0) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "gray", size = 0.8) +  # 参考线
  geom_text(aes(label = AUC_label), 
            size = 4, hjust = 0.5, nudge_y = 0.3,check_overlap = TRUE, show.legend = FALSE) +  # AUC数值标签
  
  # 美化设置
  scale_x_continuous(limits = c(min(train_forest_df$CI_lower) * 0.95, max(train_forest_df$CI_upper) * 1.05),
                     breaks = seq(0.5, 1.0, by = 0.05), expand = c(0, 0)) +
  scale_y_discrete(expand = expansion(add = 0.6)) +
  scale_color_manual(values = color_mapping) +
  
  # 标题和标签
  labs(title = "Forest Plot of Each Model AUC Score in Trainset",
       x = "AUC Score (95% CI)",
       y = "Models") +
  
  # 主题设置
  theme_minimal(base_size = 14) 
# 保存森林图
ggsave("5a_Trainset_Forest_Plot_AUC.pdf", width = 12, height = 8, device = "pdf")

test_forest_df <- test_auc_ci %>%
  arrange(AUC)%>% # 按AUC升序排列
  mutate(
    Model = factor(Model, levels = rev(Model)),  # 反转模型顺序，使首项在上
    AUC_label = sprintf("%.3f (%.3f-%.3f)", AUC, CI_lower, CI_upper)
  ) 



# 创建森林图
ggplot(test_forest_df, aes(x = AUC, y = Model)) +
  geom_point(aes(color = Model), size = 5, shape = 18) +  # 菱形点表示AUC值
  geom_errorbarh(aes(xmin = CI_lower, xmax = CI_upper, color = Model),
                 height = 0.1, size = 1.0) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "gray", size = 0.8) +  # 参考线
  geom_text(aes(label = AUC_label), 
            size = 4, hjust = 0.5, nudge_y = 0.3,check_overlap = TRUE, show.legend = FALSE) +  # AUC数值标签
  
  # 美化设置
  scale_x_continuous(limits = c(min(test_forest_df$CI_lower) * 0.95, max(test_forest_df$CI_upper) * 1.05),
                     breaks = seq(0.5, 1.0, by = 0.05), expand = c(0, 0)) +
  scale_y_discrete(expand = expansion(add = 0.6)) +
  scale_color_manual(values = color_mapping) +
  
  # 标题和标签
  labs(title = "Forest Plot of Each Model AUC Score in Testset",
       x = "AUC Score (95% CI)",
       y = "Models") +
  
  # 主题设置
  theme_minimal(base_size = 14) 
# 保存森林图
ggsave("5b_Testset_Forest_Plot_AUC.pdf", width = 12, height = 8, device = "pdf")
dev.off()


# 第4道菜： ROC曲线比较
# === ROC曲线比较 ==============================================
# 训练集ROC曲线比较

str(train_roc_list$RandomForest)

pdf("6a_Train_ROC_Comparison.pdf", width = 7, height = 7)
first_index <- which(!sapply(train_roc_list, is.null))[1]
if(is.na(first_index)) {
  plot(0, 0, type="n", xlim=c(0,1), ylim=c(0,1), 
       main="No ROC curves available (Training Set)",
       xlab="False Positive Rate (1 - Specificity)", 
       ylab="True Positive Rate (Sensitivity)")
} else {
  # 动态计算曲线数量
  n_curves <- length(train_roc_list)
  
  # 从allcolour中提取与曲线数量匹配的颜色
  curve_colors <- allcolour[1:n_curves]  # 直接按索引顺序分配颜色
  
  # 绘制第一条曲线（使用自定义颜色）
  plot(train_roc_list[[first_index]], col = curve_colors[first_index], lwd=2, legacy.axes = T,
       xlab="False Positive Rate (1 - Specificity)", 
       ylab="True Positive Rate (Sensitivity)", 
       main="Training Set ROC Curves")
  
  # 添加其他曲线（使用自定义颜色）
  for (i in (1:n_curves)) {
    if(i != first_index && !is.null(train_roc_list[[i]])) {
      plot(train_roc_list[[i]], add=TRUE, col=curve_colors[i], lwd=2,legacy.axes = T,)
    }
  }
  
  # 添加对角线
  abline(a=0, b=1, lty=3, col="gray")
  
  # 创建图例文本（含AUC）
  model_names <- names(train_roc_list)
  auc_text <- sapply(train_roc_list, function(roc) {
    if(!is.null(roc)) sprintf("%.3f", auc(roc)) else "N/A"
  })
  legend_text <- paste0(model_names, ": AUC = ", auc_text)
  
  # 添加图例（颜色与曲线对应）
  legend("bottomright", legend=legend_text, 
         col=curve_colors, lwd=2, cex=0.8)  # 使用curve_colors确保颜色一致
}
dev.off()

# 测试集ROC曲线比较
pdf("6b_Test_ROC_Comparison.pdf", width = 7, height = 7)
first_index <- which(!sapply(test_roc_list, is.null))[1]
if(is.na(first_index)) {
  plot(0, 0, type="n", xlim=c(0,1), ylim=c(0,1), 
       main="No ROC curves available (Training Set)",
       xlab="False Positive Rate (1 - Specificity)", 
       ylab="True Positive Rate (Sensitivity)")
} else {
  # 动态计算曲线数量
  n_curves <- length(test_roc_list)
  
  # 从allcolour中提取与曲线数量匹配的颜色
  curve_colors <- allcolour[1:n_curves]  # 直接按索引顺序分配颜色
  
  # 绘制第一条曲线（使用自定义颜色）
  plot(test_roc_list[[first_index]], col = curve_colors[first_index], lwd=2, legacy.axes = T,
       xlab="False Positive Rate (1 - Specificity)", 
       ylab="True Positive Rate (Sensitivity)", 
       main="Test Set ROC Curves")
  
  # 添加其他曲线（使用自定义颜色）
  for (i in (1:n_curves)) {
    if(i != first_index && !is.null(test_roc_list[[i]])) {
      plot(test_roc_list[[i]], add=TRUE, col=curve_colors[i], lwd=2,legacy.axes = T,)
    }
  }
  
  # 添加对角线
  abline(a=0, b=1, lty=3, col="gray")
  
  # 创建图例文本（含AUC）
  model_names <- names(test_roc_list)
  auc_text <- sapply(test_roc_list, function(roc) {
    if(!is.null(roc)) sprintf("%.3f", auc(roc)) else "N/A"
  })
  legend_text <- paste0(model_names, ": AUC = ", auc_text)
  
  # 添加图例（颜色与曲线对应）
  legend("bottomright", legend=legend_text, 
         col=curve_colors, lwd=2, cex=0.8)  # 使用curve_colors确保颜色一致
}
dev.off()


# 第5道菜：决策分析曲线DCA
# === 决策曲线分析(DCA)模块 =====================================
# 准备训练集数据（转换为0/1格式）
train_truth <- ifelse(train_data$Group == "DN", 1, 0)

# 创建训练集的DCA数据框
dca_train_data <- data.frame(truth = train_truth)
for (model_name in names(train_probs_list)) {
  dca_train_data[[model_name]] <- train_probs_list[[model_name]]
}

# 创建验证集的DCA数据框
dca_test_data <- data.frame(truth = test_group)
for (model_name in names(test_probs_list)) {
  dca_test_data[[model_name]] <- test_probs_list[[model_name]]
}

# === 自定义DCA计算函数 ===
calculate_dca <- function(data, outcome, predictors) {
  thresholds <- seq(0.01, 0.99, by = 0.01)
  results <- data.frame()
  
  # 提取关键统计量
  outcome_vector <- data[[outcome]]
  n <- length(outcome_vector)
  n_positive <- sum(outcome_vector == 1)  # 实际阳性患者数
  n_negative <- sum(outcome_vector == 0)  # 实际阴性患者数
  
  for (pt in thresholds) {
    # 正确计算Treat all策略
    all_positive_nb <- (n_positive / n) - (n_negative / n) * (pt / (1 - pt))
    
    # Treat none策略 (永远是0)
    all_negative_nb <- 0
    
    # 各模型的NB计算
    model_nbs <- sapply(predictors, function(p) {
      pred <- data[[p]]
      if (all(is.na(pred))) return(NA)
      
      # 计算模型的TP和FP
      pred_positive <- pred >= pt  # 模型预测为阳性
      tp <- sum(outcome_vector == 1 & pred_positive, na.rm = TRUE)
      fp <- sum(outcome_vector == 0 & pred_positive, na.rm = TRUE)
      n_valid <- length(which(!is.na(pred)))
      
      (tp / n_valid) - (fp / n_valid) * (pt / (1 - pt))
    })
    
    # 合并结果
    res_row <- data.frame(
      threshold = pt,
      variable = c("Treat all", "Treat none", names(model_nbs)),
      net_benefit = c(all_positive_nb, all_negative_nb, model_nbs)
    )
    
    results <- rbind(results, res_row)
  }
  
  return(results)
}

# === 使用自定义函数进行DCA分析 ===

# 处理训练集数据
dca_train_data_clean <- na.omit(dca_train_data)
model_names_train <- setdiff(colnames(dca_train_data_clean), "truth")
dca_train_res <- calculate_dca(dca_train_data_clean, "truth", model_names_train)

# 处理测试集数据
dca_test_data_clean <- na.omit(dca_test_data)
model_names_test <- setdiff(colnames(dca_test_data_clean), "truth")
dca_test_res <- calculate_dca(dca_test_data_clean, "truth", model_names_test)

# 重构DCA数据为宽格式
convert_dca_to_wide <- function(dca_data) {
  # 提取唯一策略名称
  strategies <- unique(dca_data$variable)
  
  # 初始化数据框
  wide_data <- data.frame(threshold = unique(dca_data$threshold))
  
  # 为每个策略创建列
  for (strategy in strategies) {
    wide_data[[strategy]] <- sapply(wide_data$threshold, function(t) {
      dca_data$net_benefit[dca_data$threshold == t & dca_data$variable == strategy]
    })
  }
  
  return(wide_data)
}

# 转换训练和验证数据
dca_train_wide <- convert_dca_to_wide(dca_train_res)
colnames(dca_train_wide) <- gsub(" ", "_", colnames(dca_train_wide))
dca_test_wide <- convert_dca_to_wide(dca_test_res)
colnames(dca_test_wide) <- gsub(" ", "_", colnames(dca_test_wide))

# Modified DCA plotting function with English-only labels
plot_custom_dca <- function(dca_wide, title) {
  # Extract model names (exclude all and none)
  model_names <- setdiff(colnames(dca_wide), c("threshold", "Treat_all", "Treat_none"))
  
  # Create base plot
  p <- ggplot(dca_wide, aes(x = threshold)) +
    # Plot reference strategies
    geom_line(aes(y = Treat_all, color = "Treat_all"), linetype = "dashed", size = 0.8) +
    geom_line(aes(y = Treat_none, color = "Treat_none"), linetype = "dashed", size = 0.8)
  
  # Add lines for each model
  for (model_name in model_names) {
    p <- p + geom_line(aes_string(y = model_name, color = shQuote(model_name)), size = 1)
  }
  
  # Number of models for color palette
  n_colors <- length(model_names)
  
  # Set colors and labels
  p <- p +
    scale_color_manual(
      name = "Strategy",
      values = c(
        "Treat_all" = "gray",
        "Treat_none" = "black",
        setNames(rainbow(n_colors), model_names)
      ),
      breaks = c("Treat_all", "Treat_none", model_names)
    ) +
    labs(
      title = title,
      x = "High risk threshold",
      y = "Net benefit"
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10),
      legend.text = element_text(size = 9),
      legend.title = element_text(size = 10, face = "bold")
    ) +
    theme(panel.grid.major = element_line(color = "gray90", size = 0.2),
          panel.grid.minor = element_line(color = "gray95", size = 0.1)) +
    scale_x_continuous(breaks = seq(0, 1, by = 0.1), 
                       limits = c(0, 1)) +
    scale_y_continuous(limits = c(-0.1, max(dca_wide[model_names], na.rm = TRUE) * 1.1))
  
  return(p)
}

pdf("7a_DCA_Training_Set.pdf", width=10, height=7)
print(plot_custom_dca(dca_train_wide, "Decision Curve Analysis (Training Set)"))+scale_color_manual(values = color_mapping)
dev.off()

pdf("7b_DCA_Validation_Set.pdf", width=10, height=7)
print(plot_custom_dca(dca_test_wide, "Decision Curve Analysis (Validation Set)"))+scale_color_manual(values = color_mapping)
dev.off()

# 第六道菜：反向残差累积图
## 2025-07-10更新，增加了残差可视化
library(DALEX)
library(ggplot2)
library(pROC)

# 定义预测函数（输出阳性类别的概率）
p_fun <- function(object, newdata) {
  if (inherits(object, "train")) {
    # caret模型
    predict(object, newdata = newdata, type = "prob")[, "DN"]
  } else if (inherits(object, "catboost.Model")) {
    # CatBoost模型
    pool <- catboost.load_pool(newdata)
    catboost.predict(object, pool, prediction_type = "Probability")
  } else if (inherits(object, "lgb.Booster")) {
    # LightGBM模型
    predict(object, as.matrix(newdata))
  }
}

# 准备训练集和验证集的真实标签
yTrain <- ifelse(train_data$Group == "DN", 1, 0)
yTest <- as.numeric(test_group)

# 创建存储性能对象的列表
train_mp_list <- list()
test_mp_list <- list()

# === 为每个模型创建解释器并计算性能 ===

# 1. 处理caret训练的12个模型
for (idx in seq_len(nrow(model_settings))) {
  algoName <- model_settings$AlgorithmName[idx]
  model <- modelContainer[[model_settings$Implementation[idx]]]
  
  if (!is.null(model)) {
    # 训练集解释器和性能
    explainer_train <- DALEX::explain(
      model = model,
      data = train_data[, -which(names(train_data) == response_var)],
      y = yTrain,
      label = algoName,
      predict_function = p_fun,
      verbose = FALSE
    )
    train_mp_list[[algoName]] <- model_performance(explainer_train)
    # 测试集解释器和性能
    explainer_test <- DALEX::explain(
      model = model,
      data = test_data,
      y = yTest,
      label = algoName,
      predict_function = p_fun,
      verbose = FALSE
    )
    test_mp_list[[algoName]] <- model_performance(explainer_test)
  }
}

# 2. 处理CatBoost模型
if (!is.null(modelContainer[["catboost"]])) {
  algoName <- "CATBoost"
  model <- modelContainer[["catboost"]]
  
  # 训练集解释器和性能
  explainer_train <- DALEX::explain(
    model = model,
    data = train_data[, -which(names(train_data) == response_var)],
    y = yTrain,
    label = algoName,
    predict_function = p_fun,
    verbose = FALSE
  )
  train_mp_list[[algoName]] <- model_performance(explainer_train)
  
  # 测试集解释器和性能
  explainer_test <- DALEX::explain(
    model = model,
    data = test_data,
    y = yTest,
    label = algoName,
    predict_function = p_fun,
    verbose = FALSE
  )
  test_mp_list[[algoName]] <- model_performance(explainer_test)
}

# 3. 处理LightGBM模型
if (!is.null(modelContainer[["lightgbm"]])) {
  algoName <- "LightGBM"
  model <- modelContainer[["lightgbm"]]
  
  # 训练集解释器和性能
  explainer_train <- DALEX::explain(
    model = model,
    data = train_data[, -which(names(train_data) == response_var)],
    y = yTrain,
    label = algoName,
    predict_function = p_fun,
    verbose = FALSE
  )
  train_mp_list[[algoName]] <- model_performance(explainer_train)
  
  # 测试集解释器和性能
  explainer_test <- DALEX::explain(
    model = model,
    data = test_data[, -which(names(train_data) == response_var)],
    y = yTest,
    label = algoName,
    predict_function = p_fun,
    verbose = FALSE
  )
  test_mp_list[[algoName]] <- model_performance(explainer_test)
}



plot(train_mp_list[[1]],train_mp_list[[2]],train_mp_list[[3]],
     train_mp_list[[4]],train_mp_list[[5]],train_mp_list[[6]],
     train_mp_list[[7]],train_mp_list[[8]],train_mp_list[[9]],
     train_mp_list[[10]],train_mp_list[[11]],train_mp_list[[12]],
     train_mp_list[[13]])+
  scale_color_manual(values = color_mapping) + ggtitle('Reverse_cumulative_|residua|_distribution_trainset')
ggsave('8a_Reverse_cumulative_residual_distribution_train_set.pdf',width = 9.7, height = 6.1)

plot(test_mp_list[[1]],test_mp_list[[2]],test_mp_list[[3]],
     test_mp_list[[4]],test_mp_list[[5]],test_mp_list[[6]],
     test_mp_list[[7]],test_mp_list[[8]],test_mp_list[[9]],
     test_mp_list[[10]],test_mp_list[[11]],test_mp_list[[12]],
     test_mp_list[[13]])+ #,test_mp_list[[14]]
  scale_color_manual(values = color_mapping)+ ggtitle('Reverse_cumulative_|residua|_distribution_testset')
ggsave('8b_Reverse_cumulative_residual_distribution_test_set.pdf',width = 9.7, height = 6.1)

# 第7道菜：残差箱型图
plot(train_mp_list[[1]],train_mp_list[[2]],train_mp_list[[3]],
     train_mp_list[[4]],train_mp_list[[5]],train_mp_list[[6]],
     train_mp_list[[7]],train_mp_list[[8]],train_mp_list[[9]],
     train_mp_list[[10]],train_mp_list[[11]],train_mp_list[[12]],
     train_mp_list[[13]],geom = "boxplot")+ #,train_mp_list[[14]]
  scale_fill_manual(values = color_mapping)+ggtitle('Boxplot of |Residual| in traindata')
ggsave('9a_Boxplot_redisual_train_set.pdf',width = 9.42, height = 6.61)

plot(test_mp_list[[1]],test_mp_list[[2]],test_mp_list[[3]],
     test_mp_list[[4]],test_mp_list[[5]],test_mp_list[[6]],
     test_mp_list[[7]],test_mp_list[[8]],test_mp_list[[9]],
     test_mp_list[[10]],test_mp_list[[11]],test_mp_list[[12]],
     test_mp_list[[13]],geom = "boxplot")+ #,test_mp_list[[14]]
  scale_fill_manual(values = color_mapping)+ggtitle('Boxplot of |Residual| in testdata')
ggsave('9b_Boxplot_redisual_test_set.pdf',width = 9.42, height = 6.61)


AUCresults
# === 找到最佳模型 ===
best_model_name <- test_auc_ci$Model[which.max(test_auc_ci$AUC)]
cat(sprintf("最佳模型: %s (AUC = %.3f)\n", best_model_name, max(test_auc_ci$AUC)))


## 2025-07-09更新，支持对最优模型CATBoost和LightGBM进行SHAP解释
#best_model_name = "CATBoost"
#best_model_name = "LightGBM"

# 获取最佳模型对象并处理预测函数
if (best_model_name == "CATBoost") {
  final_model <- modelContainer[["catboost"]]
  
  # CatBoost专用预测函数
  predict_catboost <- function(model, newdata) {
    pool <- catboost.load_pool(newdata)
    probs <- catboost.predict(model, pool, prediction_type = "Probability")
    pred_labels <- ifelse(probs > 0.5, "DN", "Control")
    return(list(probs = probs, labels = factor(pred_labels, levels = c("Control", "DN"))))
  }
  
} else if (best_model_name == "LightGBM") {
  final_model <- modelContainer[["lightgbm"]]
  
  # LightGBM专用预测函数
  predict_lightgbm <- function(model, newdata) {
    probs <- predict(model, as.matrix(newdata))
    pred_labels <- ifelse(probs > 0.5, "DN", "Control")
    return(list(probs = probs, labels = factor(pred_labels, levels = c("Control", "DN"))))
  }
  
} else {
  impl <- model_settings$Implementation[which(model_settings$AlgorithmName == best_model_name)]
  final_model <- train(formula, train_data,
                       method = impl,
                       trControl = cv_control)
  
  # caret模型专用预测函数
  predict_caret <- function(model, newdata) {
    probs <- predict(model, newdata = newdata, type = "prob")[,"DN"]
    pred_labels <- predict(model, newdata = newdata)
    return(list(probs = probs, labels = pred_labels))
  }
}


# 第八道菜：最优模型混淆矩阵
# === 最佳模型的混淆矩阵可视化 ===
# 1. 训练集混淆矩阵
cat("\n===== 绘制训练集混淆矩阵 =====\n")
if (best_model_name == "CATBoost") {
  pred_result <- predict_catboost(final_model, train_data[, -which(names(train_data) == response_var)])
} else if (best_model_name == "LightGBM") {
  pred_result <- predict_lightgbm(final_model, train_data[, -which(names(train_data) == response_var)])
} else {
  pred_result <- predict_caret(final_model, train_data)
}

train_pred <- pred_result$labels
train_probs <- pred_result$probs
train_truth <- train_data$Group

# 确保因子水平一致
train_truth <- factor(ifelse(train_truth == "Control", "Control", "DN"),
                      levels = c("Control", "DN"))

# 创建混淆矩阵
train_cm <- confusionMatrix(train_pred, train_truth, positive = "DN")
# 计算准确率
accuracy <- round(train_cm$overall["Accuracy"] * 100, 1)

train_cm_data <- as.data.frame(train_cm$table)
train_cm_data$AccuracyLabel <- ifelse(train_cm_data$Reference == train_cm_data$Prediction, "Correct", "Incorrect")

# 改进后的训练集混淆矩阵
ggplot(train_cm_data, aes(x = Prediction, y = Reference, fill = AccuracyLabel)) +
  geom_tile(color = "white", alpha = 0.9, width = 0.95, height = 0.95) +
  geom_text(aes(label = Freq, color = AccuracyLabel), 
            size = 8, fontface = "bold", show.legend = FALSE) +
  scale_fill_manual(values = c(Correct = "#4da6ff", Incorrect = "#ff7b7b")) +
  scale_color_manual(values = c(Correct = "white", Incorrect = "black")) +
  labs(title = "Training Set Confusion Matrix",
       subtitle = sprintf("Accuracy: %.1f%%", accuracy),
       x = "Predicted Label",
       y = "True Label",
       fill = "") +
  theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 18, face = "bold"),
    axis.title.x = element_text(size = 15, face = "bold"),
    axis.title.y = element_text(size = 15, face = "bold"),
    axis.text = element_text(size = 10, face = "bold"),
    legend.position = "none",
    panel.grid = element_blank(),
    panel.border = element_rect(color = "#e0e0e0", fill = NA, size = 1.5),
    aspect.ratio = 1
  ) 

# 保存训练集混淆矩阵
ggsave("10a_Best_Model_Confusion_Matrix_Training_Set.pdf",
       width = 10, height = 8, device = "pdf")

# 2. 测试集混淆矩阵
cat("\n===== 绘制测试集混淆矩阵 =====\n")
if (best_model_name == "CATBoost") {
  pred_result <- predict_catboost(final_model, test_data)
} else if (best_model_name == "LightGBM") {
  pred_result <- predict_lightgbm(final_model, test_data)
} else {
  pred_result <- predict_caret(final_model, test_data)
}

test_pred <- pred_result$labels
test_probs <- pred_result$probs
test_truth <- factor(ifelse(test_group == "1", "DN", "Control"),
                     levels = c("Control", "DN"))

# 创建混淆矩阵
test_cm <- confusionMatrix(test_pred, test_truth, positive = "DN")

# 计算准确率
accuracy_test <- round(test_cm$overall["Accuracy"] * 100, 1)

test_cm_data <- as.data.frame(test_cm$table)
test_cm_data$AccuracyLabel <- ifelse(test_cm_data$Reference == test_cm_data$Prediction, "Correct", "Incorrect")

# 改进后的训练集混淆矩阵
ggplot(test_cm_data, aes(x = Prediction, y = Reference, fill = AccuracyLabel)) +
  geom_tile(color = "white", alpha = 0.9, width = 0.95, height = 0.95) +
  geom_text(aes(label = Freq, color = AccuracyLabel), 
            size = 8, fontface = "bold", show.legend = FALSE) +
  scale_fill_manual(values = c(Correct = "#4da6ff", Incorrect = "#ff7b7b")) +
  scale_color_manual(values = c(Correct = "white", Incorrect = "black")) +
  labs(title = "Test Set Confusion Matrix",
       subtitle = sprintf("Accuracy: %.1f%%", accuracy_test),
       x = "Predicted Label",
       y = "True Label",
       fill = "") +
  theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 18, face = "bold"),
    axis.title.x = element_text(size = 15, face = "bold"),
    axis.title.y = element_text(size = 15, face = "bold"),
    axis.text = element_text(size = 10, face = "bold"),
    legend.position = "none",
    panel.grid = element_blank(),
    panel.border = element_rect(color = "#e0e0e0", fill = NA, size = 1.5),
    aspect.ratio = 1
  ) 

# 保存训练集混淆矩阵
ggsave("10b_Best_Model_Confusion_Matrix_Test_Set.pdf",
       width = 10, height = 8, device = "pdf")
dev.off()


# === 临床影响曲线和校准曲线模块===

# 第九道菜：临床影响曲线
# 加载所需包
library(ggplot2)
library(reshape2)

# 临床影响曲线函数
calculate_clinical_impact <- function(probs_list, truth) {
  thresholds <- seq(0, 1, by = 0.01)
  impact_data <- data.frame(Threshold = thresholds)
  
  for (model_name in names(probs_list)) {
    pred_probs <- probs_list[[model_name]]
    if (all(is.na(pred_probs))) next
    
    # 计算每个阈值下的预测阳性数
    pred_positives <- sapply(thresholds, function(t) sum(pred_probs >= t, na.rm = TRUE))
    
    # 计算每个阈值下的实际阳性数
    true_positives <- sapply(thresholds, function(t) {
      idx <- which(pred_probs >= t)
      sum(truth[idx] == 1, na.rm = TRUE)
    })
    
    impact_data[[model_name]] <- pred_positives  # 简化列名
    impact_data[[paste0(model_name, "_Observed")]] <- true_positives
  }
  
  return(impact_data)
}

# 校准曲线计算函数
calculate_calibration_curve <- function(probs_list, truth, n_bins = 10) {
  calibration_data <- data.frame()
  
  for (model_name in names(probs_list)) {
    pred_probs <- probs_list[[model_name]]
    if (all(is.na(pred_probs))) next
    
    # 移除NA值
    valid_idx <- !is.na(pred_probs)
    pred_probs <- pred_probs[valid_idx]
    truth_sub <- truth[valid_idx]
    
    # 使用等样本量分箱（更准确）
    # quantiles <- quantile(pred_probs, probs = seq(0, 1, length.out = n_bins + 1))
    quantiles <- unique(quantile(pred_probs, probs = seq(0, 1, length.out = n_bins + 1)))
    bins <- cut(pred_probs, breaks = quantiles, include.lowest = TRUE)
    
    # 计算每个箱的实际事件发生率
    obs_rate <- tapply(truth_sub, bins, mean)
    pred_mean <- tapply(pred_probs, bins, mean)
    bin_count <- tapply(pred_probs, bins, length)
    
    calibration_data <- rbind(calibration_data, 
                              data.frame(
                                Model = model_name,
                                Predicted = pred_mean,
                                Observed = obs_rate,
                                Count = bin_count
                              ))
  }
  
  return(calibration_data)
}



# 创建最佳模型的概率列表
best_train_probs_list <- list()
best_train_probs_list[[best_model_name]] <- train_probs_list[[best_model_name]]
best_test_probs_list <- list()
best_test_probs_list[[best_model_name]] <- test_probs_list[[best_model_name]]


# === 在调用部分使用相同的参数 ===
train_truth <- ifelse(train_data$Group == "DN", 1, 0)
test_truth <- as.numeric(test_group)

# 使用新函数计算临床影响曲线
best_train_impact_data <- calculate_clinical_impact(best_train_probs_list, train_truth)
best_test_impact_data <- calculate_clinical_impact(best_test_probs_list, test_truth)

head(best_train_impact_data)
head(best_test_impact_data)

# 使用新函数计算校准曲线
train_calibration_data <- calculate_calibration_curve(train_probs_list, train_truth, n_bins = 10)
test_calibration_data <- calculate_calibration_curve(test_probs_list, test_truth, n_bins = 10)
best_train_calibration_data <- calculate_calibration_curve(best_train_probs_list, train_truth, n_bins = 10)
best_test_calibration_data <- calculate_calibration_curve(best_test_probs_list, test_truth, n_bins = 10)

head(train_calibration_data)
head(test_calibration_data)

# 1. 临床影响曲线可视化函数
plot_clinical_impact <- function(impact_data, title) {
  # 转换为长格式
  long_data <- melt(impact_data, id.vars = "Threshold", variable.name = "Variable", value.name = "Count")
  
  # 分离类型和模型
  long_data$Type <- ifelse(grepl("_Observed", long_data$Variable), "Observed", "Predicted")
  long_data$Model <- gsub("_Observed", "", long_data$Variable)
  
  # 提取模型名称（唯一）
  model_name <- unique(long_data$Model)
  
  # 创建临床影响曲线图
  p <- ggplot(long_data, aes(x = Threshold, y = Count, color = Type, linetype = Type)) +
    geom_line(size = 1.2) +
    scale_color_manual(values = c("Predicted" = "#E41A1C", "Observed" = "#377EB8")) +
    scale_linetype_manual(values = c("Predicted" = "solid", "Observed" = "dashed")) +
    labs(title = paste(title, "-", model_name),
         x = "High Risk Threshold", 
         y = "Number of Cases",
         color = "",
         linetype = "") +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
      axis.title = element_text(face = "bold", size = 12),
      legend.position = "top",
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "gray", fill = NA, size = 1)
    ) +
    scale_x_continuous(breaks = seq(0, 1, 0.2))
  
  return(p)
}

# 2. 校准曲线可视化函数
plot_calibration_curve <- function(calibration_data, title) {
  # 计算样本大小对应点的大小
  calibration_data$PointSize <- sqrt(calibration_data$Count) * 0.8
  
  # 创建校准曲线图
  p <- ggplot(calibration_data, aes(x = Predicted, y = Observed)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40", size = 1) +
    geom_point(aes(size = PointSize, color = Model), alpha = 0.8) +
    geom_line(aes(color = Model), size = 1.0, alpha = 0.7) +
    #geom_text(aes(label = Count), size = 4, vjust = -1, color = "black") +
    scale_color_manual(values = color_mapping) +
    scale_size(range = c(3, 8), guide = "none") + # 点大小不显示在图例
    labs(title = paste("Calibration Curve:", title),
         x = "Mean Predicted Probability", 
         y = "Observed Event Rate",
         color = "Model") +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
      axis.title = element_text(face = "bold", size = 12),
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "gray", fill = NA, size = 1)
    ) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    annotate("text", x = 0.8, y = 0.2, label = "Perfect calibration", 
             color = "gray40", size = 5)
  
  return(p)
}

# 3. 创建并保存所有图表
# 训练集临床影响曲线
train_impact_plot <- plot_clinical_impact(
  best_train_impact_data, "Training Set"
)
train_impact_plot
ggsave("11a_Clinical_Impact_Training_Set_Best_Model.pdf", 
       train_impact_plot, width = 9, height = 7)

# 测试集临床影响曲线
test_impact_plot <- plot_clinical_impact(
  best_test_impact_data, "Validation Set"
)
test_impact_plot 
ggsave("11b_Clinical_Impact_Validation_Set_Best_Model.pdf", 
       test_impact_plot, width = 9, height = 7)


# 第十道菜： 校准曲线
# 训练集校准曲线
train_calibration_plot <- plot_calibration_curve(
  train_calibration_data, "Training Set"
)

train_calibration_plot 
ggsave("12a_Calibration_Curve_Training_Set.pdf", 
       train_calibration_plot, width = 8, height = 7)

# 测试集校准曲线
test_calibration_plot <- plot_calibration_curve(
  test_calibration_data, "Validation Set"
)
test_calibration_plot 
ggsave("12b_Calibration_Curve_Validation_Set.pdf", 
       test_calibration_plot, width = 8, height = 7)

## 最佳训练集和测试集曲线
best_train_calibration_plot <- plot_calibration_curve(
  best_train_calibration_data, "Training Set"
)

best_train_calibration_plot 
ggsave("12c_Calibration_Curve_Training_Set_best.pdf", 
       best_train_calibration_plot, width = 8, height = 7)


best_test_calibration_plot <- plot_calibration_curve(
  best_test_calibration_data, "Validation Set"
)
best_test_calibration_plot 
ggsave("12d_Calibration_Curve_Validation_Set_best_model.pdf", 
       best_test_calibration_plot, width = 8, height = 7)
dev.off()


# 最后一汤：SHAP
# SHAP可视化
#创建自定义预测函数（输出数值概率）
# 创建统一的预测函数
# === SHAP可视化部分 ===
# 创建统一的预测函数
pred_wrapper <- function(model, model_type, newdata) {
  if (model_type == "caret") {
    return(predict(model, newdata = newdata, type = "prob")[,"DN"])
  } else if (model_type == "catboost") {
    pool <- catboost.load_pool(newdata)
    return(catboost.predict(model, pool, prediction_type = "Probability"))
  } else if (model_type == "lightgbm") {
    return(predict(model, as.matrix(newdata)))
  }
}

# 识别最佳模型类型
best_model_type <- case_when(
  best_model_name %in% model_settings$AlgorithmName ~ "caret",
  best_model_name == "CATBoost" ~ "catboost",
  best_model_name == "LightGBM" ~ "lightgbm"
)

# 计算SHAP值
shap_vis <- NULL

if (best_model_type == "caret") {
  # caret模型的SHAP计算
  shap_values <- kernelshap(
    final_model, 
    X = train_data[, -which(names(train_data) == response_var)],
    pred_fun = function(m, x) pred_wrapper(m, "caret", x)
  )
  shap_vis <- shapviz(shap_values, train_data[, -which(names(train_data) == response_var)])
  
# } else if (best_model_type == "catboost") {
#   # CatBoost专用SHAP计算
#   pool_train <- catboost.load_pool(train_data[, -which(names(train_data) == response_var)])
#   shap_values_matrix <- catboost.get_feature_importance(
#     final_model, 
#     pool_train, 
#     type = "ShapValues"
#   )
#   # 提取SHAP值和基线
#   shap_values <- shap_values_matrix[, -ncol(shap_values_matrix)]
#   baseline <- shap_values_matrix[1, ncol(shap_values_matrix)]
#   colnames(shap_values) <- colnames(train_data[, -which(names(train_data) == response_var)])
#   
#   shap_vis <- shapviz(
#     shap_values, 
#     X = train_data[, -which(names(train_data) == response_var)],
#     baseline = baseline)
#   cat("CATboost SHAP计算完成\n")
  
} else if (best_model_type == "lightgbm") {
  # LightGBM备选SHAP计算方法
  if (!requireNamespace("fastshap", quietly = TRUE)) {
    install.packages("fastshap")
  }
  library(fastshap)
  
  # 定义LightGBM的预测函数
  pred_fun_lightgbm <- function(object, newdata) {
    predict(object, as.matrix(newdata))
  }
  
  # 计算SHAP值
  shap_values <- fastshap::explain(
    final_model, 
    X = as.data.frame(train_data[, -which(names(train_data) == response_var)]),
    pred_wrapper = pred_fun_lightgbm,
    nsim = 10  # 减少模拟次数以提高速度
  )
  
  # 创建shapviz对象
  shap_vis <- shapviz(shap_values, X = train_data[, -which(names(train_data) == response_var)])
  cat("lightGBM SHAP计算完成\n")
}


# 特征重要性排序
feature_importance <- colMeans(abs(shap_vis$S))
sorted_features <- names(sort(feature_importance, decreasing = TRUE))

# 可视化设置
visualization_theme <- theme_minimal() + 
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(size = 12))



# Generate model interpretation plots - SHAP visualizations (English)

# 1. 特征重要性柱状图（全局解释）
#    展示每个特征对模型输出的平均绝对影响大小
#    数值越大表示该特征对模型预测的影响越大
#    显示具体数值便于量化比较
pdf("13_SHAP_Feature_Importance_Barplot.pdf", width=8, height=6)
sv_importance(shap_vis, kind="bar", show_numbers=TRUE) +
  visualization_theme +
  labs(title = "Feature Importance (Mean Absolute SHAP)",
       subtitle = "Average impact magnitude of each feature on model predictions",
       x = "Mean |SHAP value|", y = "Feature",
       caption = "Bar height indicates feature importance, with value showing mean absolute SHAP")
dev.off()

# 2. 蜂群图（特征效应分布）
#    展示每个特征的SHAP值在整个数据集上的分布
#    颜色表示特征值的高低（红色高，蓝色低）
#    显示特征值与预测结果的关系趋势
#    点越分散表示非线性关系越强
pdf("14_SHAP_BeeSwarm_Plot.pdf", width=9, height=7)
sv_importance(shap_vis, kind="bee", show_numbers=TRUE) +
  visualization_theme +
  labs(title = "SHAP Value Distribution (Bee Swarm)",
       subtitle = "Each point represents one sample, color indicates feature value",
       x = "SHAP value (impact on model output)",
       y = "Feature",
       caption = "Red: high feature values | Blue: low feature values | Horizontal spread: direction of effect")
dev.off()

# 3. 特征依赖图（多个特征）
#    展示特征值与其SHAP值的非线性关系
#    每个子图显示一个特征与模型输出的依赖关系
#    趋势线帮助理解特征与预测结果的关系模式
pdf("15_SHAP_Feature_Dependence.pdf", width=10, height=8)
sv_dependence(shap_vis, sorted_features[1:6]) +  # Top 6 important features
  visualization_theme
dev.off()

# 4. 瀑布图（单个样本解释）
dir.create('16_Sample_Waterfall', showWarnings = FALSE)
for (i in sample(1:nrow(train_data), 10, replace = TRUE)) {
  pdf(paste0("16_Sample_Waterfall/sample_", i, "_waterfall.pdf"), width=9, height=6)
  p <- sv_waterfall(shap_vis, row_id = i) +  
    labs(title = paste("Prediction Breakdown for Sample", i),
         subtitle = "Cumulative contribution of features to final prediction")
  print(p)
  dev.off()
}

# 5. 单样本力图
dir.create('17_Force_Plot', showWarnings = FALSE)
for (i in sample(1:nrow(train_data), 10, replace = TRUE)) {
  pdf(paste0("17_Force_Plot/sample_", i, "_force.pdf"), width=9, height=6)
  p <- sv_force(shap_vis, row_id = i) +  
    labs(title = paste("Feature Contributions for Sample", i),
         subtitle = "Visual forces pushing prediction from base value to output")
  print(p)
  dev.off()
}

