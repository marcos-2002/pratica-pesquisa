library(dplyr)
library(ggplot2)
library(tidyr)
if (!require(png)) install.packages("png")
library(png)



url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
dados = read.csv(url, header = FALSE)

head(dados)


# Nomear as colunas
colnames(dados) = c("Tipo", "Alcohol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium", 
                     "Total_Phenols", "Flavanoids", "Nonflavanoid_Phenols", 
                     "Proanthocyanins", "Color_Intensity", "Hue", 
                     "OD280_OD315_of_Diluted_Wines", "Proline")

# 1a. Calcular a média e desvio padrão para todos os atributos
summary_stats <- data.frame(
  Media = sapply(dados[,-1], mean),
  Desvio_Padrao = sapply(dados[,-1], sd)
)
print("Média e Desvio Padrão para cada atributo:")
print(summary_stats)

# 1b. Calcular a média e desvio padrão para todos os atributos agrupado pelo tipo de vinho
grouped_stats <- dados %>%
  group_by(Tipo) %>%
  summarise(across(Alcohol:Proline, list(Media = mean, Desvio_Padrao = sd), .names = "{.fn}_{.col}"))
print("Média e Desvio Padrão para cada atributo por Tipo de Vinho:")
print(grouped_stats)

# 1c. Gráficos de distribuição de densidade para cada atributo pelo tipo de vinho
print("Gráficos de Distribuição de Densidade:")
for (atributo in colnames(dados)[-1]) {
  print(
    ggplot(dados, aes_string(x = atributo, color = "factor(Tipo)")) +
      geom_density() +
      ggtitle(paste("Distribuição de Densidade -", atributo)) +
      theme_minimal()
  )
}

# 1d. Gráficos de box-plot para cada atributo pelo tipo de vinho
print("Gráficos de Box-Plot:")
for (atributo in colnames(dados)[-1]) {
  print(
    ggplot(dados, aes_string(x = "factor(Tipo)", y = atributo)) +
      geom_boxplot() +
      ggtitle(paste("Box-Plot -", atributo)) +
      xlab("Tipo de Vinho") +
      theme_minimal()
  )
}

# 1d.2
print("Gráficos de Box-Plot:")
for (atributo in colnames(dados)[-1]) {
  print(
    ggplot(dados, aes_string(x = "factor(Tipo)", y = atributo, fill = "factor(Tipo)")) +
      geom_boxplot() +
      ggtitle(paste("Box-Plot -", atributo)) +
      xlab("Tipo de Vinho") +
      theme_minimal() +
      scale_fill_brewer(palette = "Set3")  # Escolha a paleta de cores desejada
  )
}

# 1e. Gráfico de dispersão entre os atributos
print("Gráficos de Dispersão:")
pairs(dados[,-1], col = dados$Tipo)


# 1e.2
# Instala o pacote GGally, se necessário
if (!require(GGally)) install.packages("GGally")
library(GGally)

# Define a primeira coluna como fator para colorir pelo tipo
dados$Tipo <- factor(dados$Tipo)

# Cria o gráfico de pares com ggpairs
ggpairs(dados, columns = 2:ncol(dados), aes(color = Tipo, alpha = 0.6)) +
  theme_minimal()

# 1e.3
print("Gráficos de Dispersão entre todos os pares de atributos:")

atributos <- colnames(dados)[-1]  # Exclui a coluna Tipo

for (i in 1:(length(atributos) - 1)) {
  for (j in (i + 1):length(atributos)) {
    print(
      ggplot(dados, aes_string(x = atributos[i], y = atributos[j], color = "factor(Tipo)")) +
        geom_point(alpha = 0.7) +
        labs(x = atributos[i], y = atributos[j], color = "Tipo de Vinho") +
        ggtitle(paste("Dispersão entre", atributos[i], "e", atributos[j])) +
        theme_minimal()
    )
  }
}


# 3 Construa um modelo agrupamento k-means para os dados de vinho usando o DAL Toolbox
source("https://raw.githubusercontent.com/cefet-rj-dal/daltoolbox/main/jupyter.R")

#loading DAL
load_library("daltoolbox")

#load dataset
# data(dados)

# setup clustering
model = cluster_kmeans(k=3)


# build model
model = fit(model, dados[,2:14])
clu = cluster(model, dados[,2:14])
table(clu)


# evaluate model using external metric
eval = evaluate(model, clu, dados$Tipo)
eval


# 4 Construa um modelo predição usando redes neurais para os dados de vinho usando o DAL Toolbox
# vamos usar o knn

#extracting the levels for the dataset
dados$Tipo= as.factor(dados$Tipo)

slevels <- levels(dados$Tipo)
slevels

# preparing dataset for random sampling
set.seed(1)
sr <- sample_random()
sr <- train_test(sr, dados)
dados_train <- sr$train
dados_test <- sr$test

tbl <- rbind(table(dados[,"Tipo"]), 
             table(dados_train[,"Tipo"]), 
             table(dados_test[,"Tipo"]))
rownames(tbl) <- c("dataset", "training", "test")
head(tbl)

model <- cla_knn("Tipo", slevels, k=1)
model <- fit(model, dados_train)
train_prediction <- predict(model, dados_train)

dados_train_predictand <- adjust_class_label(dados_train[,"Tipo"])
train_eval <- evaluate(model, dados_train_predictand, train_prediction)
print(train_eval$metrics)


# Test  
test_prediction <- predict(model, dados_test)

dados_test_predictand <- adjust_class_label(dados_test[,"Tipo"])
test_eval <- evaluate(model, dados_test_predictand, test_prediction)
print(test_eval$metrics)


# 5 Execute os padrões frequentes a partir dos dados de vinho
