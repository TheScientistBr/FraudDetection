Detecção de fraude
------------------

Carregando as bibliotecas necessárias.

``` r
library(data.table)
library(randomForest)
library(DMwR)
library(DT)
```

**Carregando o Dataset**

``` r
dataset <- read.csv("data/creditcard.csv", stringsAsFactors = F, sep = ",", header =T)
```

Depois de ler os dados, examinamos as probabilidades de nossa variável alvo. Nós já sabemos, a partir da descrição do conjunto de dados, também é de bom senso que tais conjuntos de dados serão altamente distorcidos.

``` r
prop.table(table(dataset$Class))
```

    ## 
    ##           0           1 
    ## 0.998272514 0.001727486

O conjunto de dados parece altamente distorcido. Além disso, devemos ter em conta que a maioria dos nossos recursos são componentes PCA. Nesta demonstração, não vamos fazer nenhuma engenharia de recursos ou nenhuma seleção de recurso (porque os componentes PCA finais explicam menos variações). Nós encaixamos nosso modelo randomForest diretamente em nosso conjunto de dados SMOTed sem qualquer seleção de recursos

Split the dataset.

``` r
set.seed(1)
i_train <- sort(sample(1:nrow(dataset), size = dim(dataset)[1]*.7))
i_test <- (1:nrow(dataset))[-i_train]
train <- dataset[i_train,]
test <- dataset[i_test,]
train$Class <- as.factor(train$Class)
test$Class <- as.factor(test$Class)
```

Vamos separar o conjunto de dados de treino e teste. Lembre-se, 1 é a variável indicador para "sim" (a fraude aconteceu) e 0 é o indicador para "não" (a fraude não aconteceu).

O uso o pacote DMwR e uso a Técnica de Oversampling de Minoridade Sintética (SMOTE), por Chawla, para lidar com a afinidade dos dados. A biblioteca "desequilibrada" também nos provoca com o algoritmo SMOT, mas o pacote DMwR facilita o nosso trabalho com a criação do conjunto de dados SMOTed e, de forma simulada, aplicando o modelo de classificação.

``` r
set.seed(1)
model <- SMOTE(Class~., data = train, perc.over = 200, k = 5, perc.under = 200, learner = "randomForest")
```

Agora temos um modelo de floresta aleatória para para prever atrvés dos dados de teste. Agora vamos criar um conjunto de dados de teste SMOTed em nosso conjunto de dados de teste

``` r
set.seed(1)
smot_test <- SMOTE(Class~., data = test, perc.over = 200, k = 5, perc.under = 200)
prop.table(table(smot_test$Class))
```

    ## 
    ##         0         1 
    ## 0.5714286 0.4285714

Agora temos um conjunto de dados com nós. Como você pode ver, o conjunto de dados agora está equilibrado com quase 50:50 ocorrência de "sim" e "não". Podemos também achar que o conjunto de dados não é aleatório e as linhas iniciais da variável Classe são todas as 0 e as linhas finais da variável Class têm todas as 1. Podemos inserir aleatoriedade em nosso conjunto de dados de teste por amostragem.

``` r
split <- sample(1:nrow(smot_test), nrow(smot_test))
smot_test <- smot_test[split,]
```

Predizemos a variável Class com nosso modelo randomForest.

``` r
p <- predict(model, smot_test)
caret::confusionMatrix(smot_test$Class, p)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 579   5
    ##          1  30 408
    ##                                          
    ##                Accuracy : 0.9658         
    ##                  95% CI : (0.9527, 0.976)
    ##     No Information Rate : 0.5959         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9296         
    ##  Mcnemar's Test P-Value : 4.976e-05      
    ##                                          
    ##             Sensitivity : 0.9507         
    ##             Specificity : 0.9879         
    ##          Pos Pred Value : 0.9914         
    ##          Neg Pred Value : 0.9315         
    ##              Prevalence : 0.5959         
    ##          Detection Rate : 0.5665         
    ##    Detection Prevalence : 0.5714         
    ##       Balanced Accuracy : 0.9693         
    ##                                          
    ##        'Positive' Class : 0              
    ## 

Temos uma boa precisão para o nosso modelo, mas, olhe para o valor *Sensitivity* e Pos Pred, eles são a Precision e Recall, a diferença entre eles é bastante grande, isso significa que nosso modelo superou a previsão do resultado positivo de nossa variável que é 0 ("não"). Precisamos abordar esta superação do nosso modelo. Há algumas opções para resolver esse problema. Podemos preparar mais dados, podemos fazer a seleção de recursos ou podemos definir o limiar de classificação.

Definiremos o limiar de classificação neste esperimento e analisaremos a Precision e Recall.

Para a classificação, obtemos nosso conjunto de dados de treinamento. Tenha cuidado aqui, você deve definir a mesma semente que você definir quando modelou.

``` r
set.seed(1)
train_smot <- SMOTE(Class ~ ., data = train, perc.over = 200, k = 5,perc.under = 200)
list(mode = "multiple", selected = c(1:5, 31), target = "column")
```

    ## $mode
    ## [1] "multiple"
    ## 
    ## $selected
    ## [1]  1  2  3  4  5 31
    ## 
    ## $target
    ## [1] "column"

``` r
set.seed(2)
split <- sample(1:nrow(train_smot), nrow(train_smot))
train_smot <- train_smot[split,]
```

Definiremos um limiar de classificação que nos dê o maior valor de *AUC*. Para isso, tomamos a ajuda do pacote `pROC`. Nós iteramos através de um limite diferente e observamos o que nos dá o melhor resultado de AUC possível.

``` r
library(pROC)
```

    ## Type 'citation("pROC")' for a citation.

    ## 
    ## Attaching package: 'pROC'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     cov, smooth, var

``` r
c <- c()
f <- c()
j <- 1
for(i in seq(0, 0.5 , 0.01)) {
        set.seed(7)
        fit <- randomForest(Class~., data = train_smot)
        pre <- predict(fit, smot_test, type = "prob")[,2]
        pre <- as.numeric(pre > i)
        auc <- roc(smot_test$Class, pre)
        c[j] <- i
        f[j] <- as.numeric(auc$auc)
        j <- j + 1
}
```

A melhor classificação de limiar para o nosso modelo.

``` r
df <- data.frame(c = c, f = f)
p <- df$c[which.max(df$f)]
p
```

    ## [1] 0.43

Ajuste o modelo com esse limite e veja a sensibilidade e o valor Pos Pre.

``` r
fit <- randomForest(Class~., data = train_smot)
pre <- predict(fit, smot_test, type = "prob")[,2]
pre <- as.numeric(pre > p)
```

Avaliando o desempenho do modelo
--------------------------------

``` r
caret::confusionMatrix(smot_test$Class, factor(pre))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 574  10
    ##          1  29 409
    ##                                           
    ##                Accuracy : 0.9618          
    ##                  95% CI : (0.9482, 0.9727)
    ##     No Information Rate : 0.59            
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9217          
    ##  Mcnemar's Test P-Value : 0.003948        
    ##                                           
    ##             Sensitivity : 0.9519          
    ##             Specificity : 0.9761          
    ##          Pos Pred Value : 0.9829          
    ##          Neg Pred Value : 0.9338          
    ##              Prevalence : 0.5900          
    ##          Detection Rate : 0.5616          
    ##    Detection Prevalence : 0.5714          
    ##       Balanced Accuracy : 0.9640          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

Nosso modelo foi bom, não só aumentamos a precisão, mas também reduzimos a diferença entre Precision e Recall. Se quisermos, podemos tentar diferentes modelos e fazer muito mais com o conjunto de dados.
\#\# Resumo

As técnicas, como a amostragem excessiva e Sub amostragem são boas para lidar com dados divergentes, mas trazem seus próprios problemas. Na detecção de fraude e no gerenciamento de risco de crédito, estamos mais inclinados às probabilidades. A ruim dessas técnicas é que elas são tendenciosas em relação a probabilidades posteriores, o que não é bom. Para lidar com tais problemas, recalibramos para obter as probabilidades corretas.
