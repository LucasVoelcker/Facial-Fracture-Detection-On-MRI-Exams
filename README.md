Este repositório contém os códigos desenvolvidos para o projeto de detecção de fraturas faciais em exames de ressonância magnética.

- Para rodar a aplicação, é necessário fazer o download das pastas /app, /weights-resnet, /weights-yolo e /yolov7-dir, e então rodar o script inference-with-interface.py.
Pode ser necessário ajustar os caminhos nos scripts inference-with-interface.py e PROCESS-PATIENT.py de acordo com o local das pastas baixadas no computador.
- A pasta /dev-codes contém códigos de diferentes etapas de pré-processamento, como a anonimização e o data augmentation do dataset, além dos códigos que rodam o treinamento da Yolo-v7 e da Resnet-18.

Link para os datasets utilizados para o treinamento da Resnet-18 e da Yolo-v7 (dataset já anonimizado e com data augmentation):
  https://drive.google.com/drive/folders/1Fry77rmQSRHxqKDspPDiiwmpM7041VMh?usp=sharing 
