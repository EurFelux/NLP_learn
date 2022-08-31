# chatbot-transformer
学校专业实习，一个星期做的项目。

使用transformer架构，加载了预训练模型albert作为encoder，decoder部分与论文中大致相同。

语料库使用小黄鸡(40w+ pairs)。

效果不算很好，但能说人话就算成功。

顺手写了个简单的前后端，可以部署到web上（虽然是开发环境）。

前端是简单的html+css+js，没有使用框架。

后端是一个简单的flask应用。

## 效果：
![chat.png](https://github.com/EurFelux/chatbot-transformer/blob/main/resources/chat.png)


## 感谢
- https://github.com/bentrevett/pytorch-seq2seq
- https://github.com/codemayq/chinese_chatbot_corpus
- https://github.com/candlewill/Dialog_Corpus
- hugging face model: clue/albert_chinese_tiny
