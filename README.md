
Sign Language Detection using Landmarks - Python
-
- Nesse projeto, o objetivo principal é criar um modelo que identifique as letras do alfabeto em libras, sem contar as letras que necessitam de um certo movimento para ser identificadas.




## Estrutura dos Arquivos

No repositório vão ter vários arquivos para ser possivel o treinamento do modelo detector, segue um resumo de funcionalidades de cada um:

HandTrackingModule.py

    . Arquivo responsável, por armazenar todos as funções importantes para identificar os landmarks das mãos, hyperpametrizar as bibliotecas utilizadas.
    . Arquivo base para o projeto inteiro.

CollectImages.py

    . Arquivo responsável, por criar as imagens que serão utilizadas em seu treinamento.
    . Terá que seguir o passo a passo mostrado no display da webcam.
    . 150 fotos com a mão direita e 150 fotos com a mão esquerda. (Todos os paramêtros, podem ser facilmente alterados)

CreateDataSet.py

    . Arquivo responsável, por buscar todas as imagens que foram coletadas e armazenar em um arquivo 'pickle', as landamarks dos 20 pontos identificados pela biblioteca mediapipe.

main.ipynb

    . Arquivo Jupyter, responsável por receber os dados coletados, tratar e treinar um modelo para identifação do mesmo.
    . Normalização dos dados: ✅
    . Balanceamento dos dados: ✅
    . Hyperparametrização: Utilizando o método RandomizedSearchCV 
    . Modelo de Treinamento: RandomForestClassifier
    . Modelo de Validação: Cross Validate 

    Após todas as validações é salvo o modelo via pickle, com o nome de 'model.p'

## Bibliotecas Utilizadas

- cv2
- HandTrackingModule
- Pickle
- Os



## Demonstração

Modelo em funcionamento:


## Aprendizados

Foi muito aprender mais sobre ciencias de dados e realizar esse projeto.
Tive muitas dificuldades no começo para entender como cada biblioteca funcionava e confesso que ainda tenho muitas duvidas, principalmente na parte de desempenho de detecção dos pontos de referência das landmarks da biblioteca mediapipe.

Em resumo, foi de grande aprendizado realizar esse projeto, mesmo que com algumas falhas, consegui criar um modelo no qual identifica todas as letras do alfabeto de libras,que não contenham movimento.

Criei também um outro repositorio, no qual utiliza a mesma estrutura, porém ele acaba tendo como dados de entrada e inferencia as distancias dos pontos principais das mãos, segue link para repositorio:

