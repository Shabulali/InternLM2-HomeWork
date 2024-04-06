# 书生浦语大模型实战营二期 · 课程笔记_02

## cli_demo.py 代码思考：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/model/Shanghai_AI_Laboratory/internlm-chat-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("User  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break
    response, history = model.chat(tokenizer, input_text, history=messages)
    messages.append((input_text, response))
    print(f"robot >>> {response}")
```



### 从指定预训练模型路径加载`tokenizer`

```python
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
```

`AutoTokenizer.from_pretrained` 的作用是从预训练的模型名称或路径中（`model_name_or_path`）加载对应的`tokenizer`。通过使用 `from_pretrained` 方法，可以方便地加载与预训练模型对应的tokenizer，而无需手动定义和配置tokenizer的参数，简化了加载和使用预训练模型的步骤。并将其映射到**CUDA**设备的第一个**GPU**上（即cuda:0），`trust_remote_code=True` 表示信任远程代码，即允许从远程获取的代码运行，加载模型时较为常用。



### Tokenizer的作用是？

`tokenizer`用于将输入文本转换为模型可以接受的**`输入格式`**，主要包括以下几个方面：

1. **将文本转换为tokens：** Tokenizer将输入文本分割成一个个小的单元，通常是单词、子词或字符，这些小单元称为tokens，模型接受的输入是tokens的序列。

2. **将tokens转换为token IDs：** Tokenizer将tokens映射为模型词汇表中的对应token ID。每个token ID对应词汇表中一个单词或字符的索引。

3. **生成attention masks：** Tokenizer生成的attention masks用于指示模型在处理输入时应该关注哪些部分，哪些部分是填充部分，哪些部分是真实的输入内容。

4. **处理特殊tokens：** Tokenizer还负责处理特殊的tokens，如起始token、结束token填充token等，以便地构建模型输入序列。

   

> 总结：`Tokenizer`的作用是将原始文本转换为模型可以接受的**token IDs序列**，并进行一些**必要的预处理**，促使模型能够有效地处理输入数据。





### 从指定预训练模型路径加载`Model`

```python
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
```

从预训练的模型路径或模型名称加载一个用于生成文本的**自回归语言**模型，并且在加载模型时：

1. `trust_remote_code=True`表示信任远程提供的代码，允许从远程地址下载模型参数和配置。
2. `torch_dtype=torch.bfloat16`表示设置模型使用的张量数据类型为`bfloat16`，这是一种16位浮点数格式，可以在硬件级别上加速模型的计算。
3. `device_map='cuda:0'`表示指定模型在特定的CUDA设备上运行。





### 何为**自回归语言**模型？

​		自回归语言模型是一种生成式的语言模型，它基于上下文生成下一个词或字符，从而实现文本的连续生成，有以下特点：

1. **逐词生成**：自回归语言模型逐词地生成文本序列，通过前面已生成的部分来预测下一个词。每次**生成一个词**后，会将其作为**下一步的输入**，以此类推生成整个文本序列。（教程创作300字故事案例。内容是依次生成输出，并非一次输出可以得到验证）

2. **考虑上下文信息**：在生成每个词时，自回归语言模型会考虑前面生成的词语，从而**保持语义和连贯性**。这种考虑上下文信息的方式有助于生成**更加自然的文本**。

3. **序列依赖性**：自回归语言模型会捕捉文本中的**序列依赖性**，即文本中词语之间的顺序关系，从而能够生成符合**语法和语义规则**的文本。

4. **生成多样性**：基于逐词生成和上下文信息的特点，自回归语言模型能够生成具有一定**多样性的文本**，而非简单地复制训练数据中的文本。

   

```reStructuredText
比如，当你写“今天天气很晴朗，我决定……”这句话时，下一个词可能是“出去散步”，“去公园玩”，或者其他动作。这种根据前面内容来预测下一个词的过程，就像自回归语言模型一样，它会考虑前面的上下文信息来生成接下来的文本，从而让整个文本更加通顺和连贯。所以，自回归语言模型就像是一个文本创作的小助手，可以帮助生成连贯的文本内容，而不仅仅是简单地复制粘贴已有的文本。
```



>  **总结**：自回归语言模型通过序列化地生成文本序列，能够在一定程度上模拟自然语言的生成过程，适用于**文本生成、机器翻译、对话生成**等任务。





### 还有哪些预训练模型？

除自回归语言模型之外，Hugging Face的Transformers库还支持多种其他类型的预训练模型，例如:

- 分类模型（AutoModelForSequenceClassification）

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("model_name")
```



- 命名实体识别模型（AutoModelForTokenClassification）

```python
from transformers import AutoModelForTokenClassification
  
model = AutoModelForTokenClassification.from_pretrained("model_name")
```



- 问答模型（AutoModelForQuestionAnswering）

```python
from transformers import AutoModelForQuestionAnswering
  
model = AutoModelForQuestionAnswering.from_pretrained("model_name")
```



### 特点和应用场景：

| 模型类型         | 特点                                                   | 应用场景                                           |
| ---------------- | ------------------------------------------------------ | -------------------------------------------------- |
| 分类模型         | 用于将输入文本分为不同的类别或标签                     | 文本分类 、情感分析、垃圾邮件过滤                  |
| 命名实体识别模型 | 用于提取文本中的特定实体信息                           | 信息抽取、语音识别<br />如人名、地名、组织机构名等 |
| 问答模型         | 具有推理和理解能力<br />用于回答用户提出的自然语言问题 | 问答系统、阅读理解                                 |



在多模态环境下：

- 分类模型可以结合文本、图像、视频等多种信息来源，进行跨模态的特征融合和学习，提高分类任务的准确性和鲁棒性。
- 命名实体识别模型可以结合图像、视频等数据源，以提高对特定实体的识别和分类准确性。
- 问答模型可以结合文本、图像、视频等多种信息来源，实现更全面的问题理解和回答能力。

如灵笔中的视觉问答，系统可以同时接受处理图像和文本形式的问题，经过模型处理后，给出综合性答案。





### torch_dtype=torch.bfloat16 作用是？

`torch_dtype=torch.bfloat16` 的作用是设置模型在加载的过程中使用的张量数据类型为`bfloat16`，即16位浮点数格式。`bfloat16` 是一种低精度浮点数格式，可以在硬件级别上加速计算，同时能够保持相对较高的计算精度，提供计算性能的提升，在一定程度上缓解浮点数计算中的数值稳定性问题。



### PyTorch支持的数据类型

常用的包括：

| 数据类型       | 优势                           | 劣势                                                  |
| -------------- | ------------------------------ | ----------------------------------------------------- |
| torch.bfloat16 | 内存占用较低 、计算速度较快    | 精度相对较低                                          |
| torch.float16  | 内存占用较小、计算速度较快     | 精度较低，可能造成计算错误 - 支持有限，部分操作不支持 |
| torch.float32  | 通用数据类型，适用于大多数场景 | 内存占用较大、计算速度较慢                            |
| torch.float64  | 提供最高精度                   | 内存占用巨大、计算速度较慢                            |





### model.eval()的作用是？

调用`model.eval()`的作用是将模型设置为评估模式。评估模式与训练模式相对应，其主要作用有以下几个方面：

1. **关闭 Dropout 和 Batch Normalization**：
   - 在评估模式下，`PyTorch`会关闭模型中的 `Dropout` 和 `Batch Normalization` 层，这意味着模型在推理阶段不再进行随机丢弃和归一化，而是使用固定的权重进行推断。
2. **禁用梯度计算**：
   - 在评估模式下，`PyTorch`会自动禁用梯度计算，这意味着模型的参数不会发生变化，不会进行反向传播计算，主要是为了避免在推理阶段浪费计算资源。
3. **影响部分层的行为**：
   - 特定的模型结构在评估模式下可能会有一些额外的行为，例如在 Transformers 模型中，评估模式下会关闭 token_type_ids，使模型更易于使用。



在教程代码中，调用`model.eval()`的作用是将加载的Causal Language Model（**自回归语言**）模型设置为评估模式，以便进行后续的**上下文推理**操作。通过将模型设置为评估模式，可以确保在模型**推断阶段**消除不必要的**计算**和**参数更新**，从而提高推理效率并**确保结果的一致性**。





### 什么时候设置为训练模式，又如何设置呢？

```python
#模型设置为训练模式
model.train()
```

​		训练模式是深度学习模型进行训练时的工作模式，其主要目的是通过在训练数据上进行反向传播计算、参数更新，从而使模型不断优化和学习，提高对新数据的泛化能力。模型训练主要流程如下：

- **数据加载**：需要将训练数据按照指定的数据集格式加载进模型进行训练，通常包括了数据预处理、数据增强等操作。

- **模型设置**：将模型设置为训练模式，即调用`model.train()`来启用模型中的 `Dropout` 和 `Batch Normalization` 层，以及允许计算的梯度。

- **前向传播计算**：将训练数据输入至模型中，进行前向传播计算。模型会根据当前的参数以及输入的数据计算出预测值。

- **计算损失函数**：将模型的预测结果与真实标签进行比较，计算损失函数值。损失函数度量了模型的预测值与真实值之间的差异程度。

- **反向传播计算梯度**：通过反向传播算法，计算损失函数对模型参数的梯度。这一步是深度学习中的关键步骤，**通过链式法则将误差沿网络向后传播，获得各层参数的梯度**。

- **参数更新**：根据梯度下降等优化算法，对模型的参数进行更新，使损失函数值逐渐降低，模型性能得到优化。

- **循环迭代**：不断重复上述步骤，将多个批次的数据输入模型进行训练，直到达到预设的训练轮数或者其他停止训练的条件。

- **模型保存**：在训练过程中，可以定期保存模型的权重参数，以便在训练意外中断或者完成后进行模型的恢复或使用。

- **模型评估**：在训练过程中可以定期对模型进行评估，通过验证集或测试集评估模型的性能，分析模型的训练效果。

  

PS: 这块等后面学到模型XTuner微调，再来细究。





### system_prompt的作用是？

定义系统提示信息`system_prompt`是为聊天机器人提供系统介绍信息，包括聊天器人的名称、开发者、设计念等内容。这样用户在和聊天机器人进行对话时，帮助用户更好地理解对话机器人的背景和基本信息，增加友好感。





### 如果模型的回复为空，让其回答固定内容，提高用户体验

```python
while True:
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break

    length = 0
    response_generated = False
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            response_generated = True
            print(response[length:], flush=True, end="")
            length = len(response)

    if not response_generated:
        print("InternLM: I'm sorry, but I couldn't generate a response to that. Please feel free to ask something else or type 'exit' to leave.", flush=True)
```





### length变量作用是？ 

​		变量 `length` 的作用是用来记录已经输出的回复文本的长度。当模型生成回复时，肯定只希望在输出时，只输出新增的部分，不是重复输出之前已经输出过的文本。因此，`length` 变量用来记录上一次输出的文本长度，以便在下一次输出时只输出新增的部分。




