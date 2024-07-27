# OpenAI 快速入门

<p align="center">
    <br> <a href="README-en.md">English</a> | 中文
</p>


本项目旨在为所有对大型语言模型及其在生成式人工智能（AIGC）场景中应用的人们提供一站式学习资源。通过提供理论基础，开发基础，和实践示例，该项目对这些前沿主题提供了全面的指导。

## 特性

- **大语言模型的理论和开发基础**：深入探讨BERT和GPT系列等大型语言模型的内部工作原理，包括它们的架构、训练方法、应用等。

- **基于OpenAI的二次开发**：OpenAI的Embedding、GPT-3.5、GPT-4模型的快速上手和应用，以及函数调用（Function Calling）和ChatGPT插件等最佳实践

- **使用LangChain进行GenAI应用开发**：通过实例和教程，利用LangChain开发GenAI应用程序，展示大型语言模型（AutoGPT、RAG-chatbot、机器翻译）的实际应用。

- **LLM技术栈与生态**：数据隐私与法律合规性，GPU技术选型指南，Hugging Face快速入门指南，ChatGLM的使用。

## 拉取代码

你可以通过克隆此仓库到你的本地机器来开始：

```shell
git clone https://github.com/DjangoPeng/openai-quickstart.git
```

然后导航至目录，并按照单个模块的指示开始操作。

## 搭建开发环境

本项目使用 Python v3.10 开发，完整 Python 依赖软件包见[requirements.txt](requirements.txt)。

关键依赖的官方文档如下：

- Python 环境管理 [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
- Python 交互式开发环境 [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
- 大模型应用开发框架 [LangChain](https://python.langchain.com/docs/get_started/installation)
- [OpenAI Python SDK ](https://github.com/openai/openai-python?tab=readme-ov-file#installation) 


**以下是详细的安装指导（以 Ubuntu 操作系统为例）**：

### 安装 Miniconda

```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

安装完成后，建议新建一个 Python 虚拟环境，命名为 `langchain`。

```shell
conda create -n langchain python=3.10

# 激活环境
conda activate langchain 
```

之后每次使用需要激活此环境。


### 安装 Python 依赖软件包

```shell
pip install -r requirements.txt
```

### 配置 OpenAI API Key

根据你使用的命令行工具，在 `~/.bashrc` 或 `~/.zshrc` 中配置 `OPENAI_API_KEY` 环境变量：

```shell
export OPENAI_API_KEY="xxxx"
```

### 安装和配置 Jupyter Lab

上述开发环境安装完成后，使用 Miniconda 安装 Jupyter Lab：

```shell
conda install -c conda-forge jupyterlab
```

使用 Jupyter Lab 开发的最佳实践是后台常驻，下面是相关配置（以 root 用户为例）：

```shell
# 生成 Jupyter Lab 配置文件，
jupyter lab --generate-config
```

打开上面执行输出的`jupyter_lab_config.py`配置文件后，修改以下配置项：

```python
c.ServerApp.allow_root = True # 非 root 用户启动，无需修改
c.ServerApp.ip = '*'
```

使用 nohup 后台启动 Jupyter Lab
```shell
$ nohup jupyter lab --port=8000 --NotebookApp.token='替换为你的密码' --notebook-dir=./ &
```

Jupyter Lab 输出的日志将会保存在 `nohup.out` 文件（已在 .gitignore中过滤）。




## 课程表

| 课表 | 描述                                                                                                                                                                                                        | 课程资料                                                                           | 任务                                                                   |
|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| 第1节   | 大模型基础：理论与技术的演进 <br/> - 初探大模型：起源与发展 <br/> - 预热篇：解码注意力机制 <br/> - 变革里程碑：Transformer的崛起 <br/> - 走向不同：GPT与BERT的选择 | 建议阅读：<br/>- [Attention Mechanism: Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)<br/>- [An Attentive Survey of Attention Models](https://arxiv.org/abs/1904.02874)<br/>- [Transformer：Attention is All you Need](https://arxiv.org/abs/1706.03762)<br/>- [BERT：Pre-training of Deep Bidirectional Transformers for Language Understanding(https://arxiv.org/abs/1810.04805) | [[作业](docs/homework_01.md)] |
| 第2节   | GPT 模型家族：从始至今 <br/> - 从GPT-1到GPT-3.5：一路的风云变幻 <br/> - ChatGPT：赢在哪里 <br/> - GPT-4：一个新的开始 <br/>提示学习（Prompt Learning） <br/> - 思维链（Chain-of-Thought, CoT）：开山之作 <br/> - 自洽性（Self-Consistency）：多路径推理 <br/> - 思维树（Tree-of-Thoughts, ToT）：续写佳话 | 建议阅读：<br/>- [GPT-1: Improving Language Understanding by Generative Pre-training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)<br/>- [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)<br/>- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)<br/><br/><br/>额外阅读：<br/>- [GPT-4: Architecture, Infrastructure, Training Dataset, Costs, Vision, MoE](https://www.semianalysis.com/p/gpt-4-architecture-infrastructure)<br/>- [GPTs are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models](https://arxiv.org/abs/2303.10130)<br/>- [Sparks of Artificial General Intelligence: Early experiments with GPT-4](https://arxiv.org/abs/2303.12712)<br/><br/> | [[作业](docs/homework_02.md)] |
| 第3节   | 大模型开发基础：OpenAI Embedding <br/> - 通用人工智能的前夜 <br/> - "三个世界"和"图灵测试" <br/> - 计算机数据表示 <br/> - 表示学习和嵌入 <br/> Embeddings Dev 101 <br/> - 课程项目：GitHub openai-quickstart <br/> - 快速上手 OpenAI Embeddings                     | 建议阅读：<br/>- [Representation Learning: A Review and New Perspectives](https://arxiv.org/abs/1206.5538)<br/>- [Word2Vec: Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)<br/>- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)<br/><br/>额外阅读：<br/><br/>- [Improving Distributional Similarity with Lessons Learned from Word Embeddings](http://www.aclweb.org/anthology/Q15-1016)<br/>- [Evaluation methods for unsupervised word embeddings](http://www.aclweb.org/anthology/D15-1036) | [[作业](docs/homework_03.md)]<br/>代码：<br/>[[embedding](openai_api/embedding.ipynb)] |
| 第4节   | OpenAI 大模型开发与应用实践 <br/> - OpenAI大型模型开发指南 <br/> - OpenAI 语言模型总览 <br/> - OpenAI GPT-4, GPT-3.5, GPT-3, Moderation <br/> - OpenAI Token 计费与计算 <br/>OpenAI API 入门与实战 <br/> - OpenAI Models API <br/> - OpenAI Completions API  <br/> - OpenAI Chat Completions API <br/> - Completions vs Chat Completions <br/>OpenAI 大模型应用实践 <br/> - 文本内容补全初探（Text Completion） <br/> - 聊天机器人初探（Chat Completion） | 建议阅读：<br/><br/>- [OpenAI Models](https://platform.openai.com/docs/models)<br/>- [OpenAI Completions API](https://platform.openai.com/docs/guides/gpt/completions-api)<br/>- [OpenAI Chat Completions API](https://platform.openai.com/docs/guides/gpt/chat-completions-api) | 代码：<br/>[[models](openai_api/models.ipynb)] <br/>[[tiktoken](openai_api/count_tokens_with_tiktoken.ipynb)] |
| 第5节   | AI大模型应用最佳实践 <br/> - 如何提升GPT模型使用效率与质量 <br/> - AI大模型应用最佳实践 <br/>   - 文本创作与生成<br/>   - 文章摘要和总结 <br/>    - 小说生成与内容监管 <br/>    - 分步骤执行复杂任务 <br/>    - 评估模型输出质量 <br/>    - 构造训练标注数据 <br/>    - 代码调试助手 <br/> - 新特性： Function Calling 介绍与实战 | 建议阅读 <br/> - [GPT Best Practices](https://platform.openai.com/docs/guides/gpt-best-practices) <br/> - [Function Calling](https://platform.openai.com/docs/guides/gpt/function-calling) | 代码： <br/> [Function Calling](openai_api/function_call.ipynb) |
| 第6节   | 实战：OpenAI-Translator <br/> - OpenAI-Translator 市场需求分析 <br/> - OpenAI-Translator 产品定义与功能规划 <br/> - OpenAI-Translator 技术方案与架构设计 <br/> - OpenAI 模块设计 <br/> - OpenAI-Translator 实战 <br/>  |  | 代码： <br/> [pdfplumber](openai-translator/jupyter/pdfplumber.ipynb) |
| 第7节   | 实战：ChatGPT Plugin 开发 <br/> - ChatGPT Plugin 开发指南 <br/> - ChatGPT Plugin 介绍 <br/> - ChatGPT Plugin 介绍 <br/> - 样例项目：待办（Todo）管理插件 <br/> - 实战样例部署与测试 <br/> - ChatGPT 开发者模式 <br/> - 实战：天气预报（Weather Forecast）插件开发 <br/> - Weather Forecast Plugin 设计与定义 <br/> - 天气预报函数服务化 <br/> - 第三方天气查询平台对接 <br/> - 实战 Weather Forecast Plugin <br/> - Function Calling vs ChatGPT plugin <br/>  | | 代码： <br/> [[todo list](chatgpt-plugins/todo-list)]  <br/> [[Weather Forecast](chatgpt-plugins/weather-forecast)] |
| 第8节   | 大模型应用开发框架 LangChain (上) <br/> - LangChain 101  <br/> - LangChain 是什么 <br/> - 为什么需要 LangChain <br/> - LangChain 典型使用场景 <br/> - LangChain 基础概念与模块化设计 <br/> - LangChain 核心模块入门与实战 <br/> - 标准化的大模型抽象：Mode I/O <br/> -  模板化输入：Prompts <br/> -  语言模型：Models <br/> - 规范化输出：Output Parsers  | | 代码： <br/> [[model io](langchain/jupyter/model_io)] |
| 第9节   | 大模型应用开发框架 LangChain (中) <br/> - 大模型应用的最佳实践 Chains <br/> - 上手你的第一个链：LLM Chain <br/> - 串联式编排调用链：Sequential Chain <br/> - 处理超长文本的转换链：Transform Chain <br/> - 实现条件判断的路由链：Router Chain <br/> - 赋予应用记忆的能力： Memory <br/> - Momory System 与 Chain 的关系 <br/> - 记忆基类 BaseMemory 与 BaseChatMessageMemory <br/> - 服务聊天对话的记忆系统 <br/> - ConversationBufferMemory <br/> - ConversationBufferWindowMemory <br/> - ConversationSummaryBufferMemory |  | 代码： <br/> [[chains](langchain/jupyter/chains)] <br/> [[memory](langchain/jupyter/memory)] |
| 第10节  | 大模型应用开发框架 LangChain (下) <br/> - 框架原生的数据处理流 Data Connection <br/> - 文档加载器（Document Loaders） <br/> - 文档转换器（Document Transformers） <br/> - 文本向量模型（Text Embedding Models） <br/> - 向量数据库（Vector Stores） <br/> - 检索器（Retrievers） <br/> - 构建复杂应用的代理系统 Agents <br/> - Agent 理论基础：ReAct <br/> -  LLM 推理能力：CoT, ToT <br/> -  LLM 操作能力：WebGPT, SayCan <br/> - LangChain Agents 模块设计与原理剖析 <br/> -  Module: Agent, Tools, Toolkits, <br/> -  Runtime: AgentExecutor, PlanAndExecute , AutoGPT, <br/> - 上手第一个Agent：Google Search + LLM <br/> - 实战 ReAct：SerpAPI + LLM-MATH |  | 代码： <br/> [[data connection](langchain/jupyter/data_connection)] <br/> [[agents](langchain/jupyter/agents)] |
| 第11节  | 实战： LangChain 版 OpenAI-Translator v2.0 <br/> - 深入理解 Chat Model 和 Chat Prompt Template <br/> - 温故：LangChain Chat Model 使用方法和流程 <br/> - 使用 Chat Prompt Template 设计翻译提示模板 <br/> - 使用 Chat Model 实现双语翻译 <br/> - 使用 LLMChain 简化构造 Chat Prompt <br/> - 基于 LangChain 优化 OpenAI-Translator 架构设计 <br/> - 由 LangChain 框架接手大模型管理 <br/> - 聚焦应用自身的 Prompt 设计 <br/> - 使用 TranslationChain 实现翻译接口 <br/> - 更简洁统一的配置管理 <br/> - OpenAI-Translator v2.0 功能特性研发 <br/> - 基于Gradio的图形化界面设计与实现 <br/> - 基于 Flask 的 Web Server 设计与实现 |  | 代码： <br/> [[openai-translator](langchain/openai-translator)] |
| 第12节  | 实战： LangChain 版Auto-GPT  <br/> - Auto-GPT 项目定位与价值解读 <br/> - Auto-GPT 开源项目介绍 <br/> - Auto-GPT 定位：一个自主的 GPT-4 实验 <br/> - Auto-GPT 价值：一种基于 Agent 的 AGI 尝试 <br/> - LangChain 版 Auto-GPT 技术方案与架构设计 <br/> - 深入理解 LangChain Agents <br/> - LangChain Experimental 模块 <br/> - Auto-GPT 自主智能体设计 <br/> - Auto-GPT Prompt 设计 <br/> - Auto-GPT Memory 设计 <br/> - 深入理解 LangChain VectorStore <br/> - Auto-GPT OutputParser 设计 <br/> - 实战 LangChain 版 Auto-GPT |    | 代码： <br/> [[autogpt](langchain/jupyter/autogpt)] |
| 第13节  | Sales-Consultant 业务流程与价值分析 <br/> - Sales-Consultant 技术方案与架构设计 <br/> - 使用 GPT-4 生成销售话术 <br/> - 使用 FAISS 向量数据库存储销售问答话术 <br/> - 使用 RetrievalQA 检索销售话术数据 <br/> - 使用 Gradio 实现聊天机器人的图形化界面 <br/> - 实战 LangChain 版 Sales-Consultant | | 代码： <br/> [[sales_chatbot](langchain/sales_chatbot)] |
| 第14节  | 大模型时代的开源与数据协议 <br/> - 什么是开源？ <br/> - 广泛使用的开源协议和数据协议 <br/> - Llama 是不是伪开源？ <br/> - ChatGLM2-6B 的开源协议 <br/> 大语言模型的可解释性 <br/> - 提高模型决策过程的透明度 <br/> - Stanford Alpaca 的相关研究 <br/> 大语言模型应用的法规合规性 <br/> - 中国大陆：生成式人工智能服务备案 <br/> - 国际化：数据隐私与保护（以 GDPR 为例） <br/> - 企业合规性应对要点 | | |
| 第15节  | 大模型时代的Github：Hugging Face <br/> - Hugging Face 是什么？ <br/> - Hugging Face Transformers 库 <br/> - Hugging Face 开源社区：Models, Datasets, Spaces, Docs <br/> - 大模型横向对比 <br/> - Open LLM Leaderboard（大模型天梯榜） <br/> 显卡选型推荐指南 <br/> - GPU vs 显卡 <br/> - GPU Core vs AMD CU <br/> - CUDA Core vs Tensor Core <br/> - N卡的架构变迁 <br/> - 显卡性能天梯榜 | | |
| 第16节  | 清华 GLM 大模型家族 <br/> - 最强基座模型 GLM-130B  <br/> - 增强对话能力 ChatGLM <br/> - 开源聊天模型 ChatGLM2-6B <br/> - 联网检索能力 WebGLM <br/> - 初探多模态 VisualGLM-6B <br/> - 代码生成模型 CodeGeex2 <br/> ChatGLM2-6B 大模型应用开发 <br/> - ChatGLM2-6B 私有化部署 <br/> - HF Transformers Tokenizer <br/> - HF Transformers Model <br/> - 将模型同步至 Hugging Face <br/> - 使用 Gradio 赋能 ChatGLM2-6B 图形化界面 <


## 贡献

贡献是使开源社区成为学习、激励和创造的惊人之处。非常感谢你所做的任何贡献。如果你有任何建议或功能请求，请先开启一个议题讨论你想要改变的内容。

<a href='https://github.com/repo-reviews/repo-reviews.github.io/blob/main/create.md' target="_blank"><img alt='Github' src='https://img.shields.io/badge/review_me-100000?style=flat&logo=Github&logoColor=white&labelColor=888888&color=555555'/></a>

## 许可证

该项目根据Apache-2.0许可证的条款进行许可。详情请参见[LICENSE](LICENSE)文件。

## 联系

Django Peng - pjt73651@email.com

项目链接: https://github.com/DjangoPeng/openai-quickstart




# 大模型应用开发实践课作业

## Day 1 作业

### 作业内容：让大模型总结课程内容

我：

> 你是一名资深的大模型开发专家，并且有很好的文案整理能力。我是一名开始学习大模型应用研发的后端开发工程师，我需要你帮我整理一下今天课程学习的内容（结合课件资源和我的笔记）

我：

> 这是我的课件资源，然后我会给你我的笔记，在我没让你帮忙生成要点的时候，你不需要生成。（附件上传 pdf 文档）

我：

> 以下是我的笔记，看完之后先不用生成，等我的安排：
>
> 大模型应用技术预览 大模型开发四个不同的阶段 深入钻研 AI 智能体 3~4 阶段的发展所需要的资源和前两个不是一个数量级， 当然效果也会明显，在特定领域下   提示词工程： 预制提示词和上下文，可以很好的提升准确性 通过使用langchain 的提示词模板可以动态生成提示词模板  作业： 1. 用prompt交流生成今天课程的主要内容和日报 2. 生成结果保留并提交 大模型的多模态 1. 文生图和文生视频 2. DALL-E、sora AI Agents： 1. 本质的工作原理：提示词+编排+工具集 2. Langchain Agent 的生态系统： Langchain 提供的核心框架能力：规划、记忆、工具 智能体的分类： 	a. 行动代理 function calling 	b. 模拟代理 角色扮演 	c. 自主智能体 autogen 3. 基于langchain 的RAG 能力  大模型微调： 1. 微调的好处：不需要那么高的成本。提示词工程是有天花板的，模型知识通过微调的投喂 可能会有更好的效果。个性化服务模型。  2. 大模型微调的常用技术 路线分类：全量和增量  预训练模型： 1. 数据质量在预训练中很重要 其他： 1. ChatGPT 功能提供 垂直行业的大模型发展（行业领域的数据集）

我：

> 请你结合我的笔记，用规整的markdown格式，清晰的标题结构等，给我一份课程学习总结，语言要精炼和简洁，并且在拥有概括的同时也能把握核心要点

GPT:

### 大模型应用技术预览

#### 大模型开发的四个阶段

1. **提示工程（Prompt Engineering）**：
   - 面向人群：终端用户
   - 技术积累：对 ChatGPT 等应用的提示词有基础的了解和使用
   - 应用场景：文本生成、机器翻译等
   - 特征总结：门槛低，易于上手
2. **AI 智能体（Agents）**：
   - 面向人群：大模型应用开发人员
   - 技术积累：了解大模型基础原理和理论，熟悉特定领域的业务逻辑和流程
   - 应用场景：自动客服、虚拟助手
   - 特征总结：侧重于交互性和用户体验
3. **大模型微调（Fine-tuning）**：
   - 面向人群：领域模型研发、私有化部署研发团队
   - 技术积累：掌握神经网络和机器学习概念，有数据处理和模型训练经验
   - 应用场景：语义理解、领域知识学习
   - 特征总结：通用性强、性价比高
4. **预训练技术（Pre-training）**：
   - 面向人群：大模型研究人员、数据科学家
   - 技术积累：熟悉深度学习原理和网络架构，有大规模数据处理和模型训练经验
   - 应用场景：多模态学习、语言模型预训练
   - 特征总结：前期投入大、效果显著

#### 提示词工程（Prompt Engineering）

- **提升准确性**：预制提示词和上下文可以显著提升生成内容的准确性。
- **动态生成提示词**：使用 LangChain 的提示词模板可以动态生成提示词模板。



### 大模型的多模态

- 文生图和文生视频
  - 代表技术：DALL-E、Sora

### AI Agents

- **工作原理**：提示词 + 编排 + 工具集
- LangChain Agent 的生态系统
  - 核心框架能力：规划、记忆、工具
  - 智能体分类
    - 行动代理（Function Calling）
    - 模拟代理（角色扮演）
    - 自主智能体（AutoGen）

#### 基于 LangChain 的 RAG 能力

- RAG 示例：基于 LangChain 知识库 + 聊天机器人

### 大模型微调

- 好处
  - 成本低：不需要太高成本。
  - 提示词工程有天花板，微调可以提供更好的效果。
  - 个性化服务模型。
- 常用技术
  - 路线分类：全量和增量
  - 微调技术路线：
    - 全量微调（Full Fine-Tune, FFT）
    - 高效微调（Parameter-Efficient Fine-Tune, PEFT）
    - 有监督微调（Supervised Fine-tune, SFT）
    - 指令微调（Instruction Tuning，IT）
    - 基于人类反馈的强化学习（RLHF）
  - 增量微调主流技术：
    - 适配器（Adapter）
    - 软提示（Soft Prompt）
    - 选择性方法（Selective Methods）
    - 基于重参数化方法（Reparameterization Methods）

### 预训练技术（Pre-training）

- **数据质量的重要性**：在预训练过程中，数据质量至关重要。
- 预训练语言模型
  - GPT-3 Training Corpus
  - LLaMA 1 基座模型

#### 大模型训练微调技术总览

- **预训练模型**：在大规模未标记数据上训练模型，学习通用特征和模式
- **微调模型**：在特定任务的有标签数据上继续训练预训练模型，调整模型的所有参数以适应该任务
- **高效微调（PEFT）**：只调整预训练模型的一部分参数，提高微调过程的参数效率
- **指令微调（Instruction Tuning）**：通过在多样化的指令性示例上训练模型，提高模型对自然语言指令的理解和执行能力
- **有监督微调（SFT）**：在有监督的数据集上对预训练模型进行微调，以适应特定的下游任务
- **基于人类反馈的强化学习（RLHF）**：使用人类反馈作为奖励信号，通过强化学习方法训练预训练模型，以生成更符合人类偏好的输出

### 其他

- **ChatGPT 功能提供**：垂直行业的大模型发展（行业领域的数据集）
- **行业案例研究**：如金融领域的 BloombergGPT，大模型在金融和银行领域的应用场景与解决方案

这就是对你今天课程学习内容的详细总结，希望对你有所帮助！



## Day 2 作业

### 作业内容 1：GPT-4V 的使用

所需要识别的图片信息：

![](./pic/pic-test.png)

使用gpt-4 进行识别，并以markdown格式输出识别结果，并提供中英内容：

```
有人等烟雨，有人怪雨急，
好像潮湿的雨水会流入心底，
这场雨的暴雨能把城市敲击。

Some people wait for the misty rain, some complain about the heavy rain,
As if the moist rainwater could flow into their hearts,
This heavy rain can pound the city.
```



### 作业内容 2： ai translator 的使用

ai translator 的使用

通过指令启动 ai-translator 的 python 项目，然后进行文本内容的翻译

> python ai_translator/main.py --model_type OpenAIModel --openai_api_key $OPENAI_API_KEY --file_format markdown --book tests/test.pdf --target_lang chinese --openai_model gpt-3.5-turbo

如果想配置程序更加灵活，可以新增target_lang 的启动参数

1. argument_parse.py 文件新增解析参数

```
class ArgumentParser:
    def __init__(self):
		...
        self.parser.add_argument('--target_lang', type=str)
```

2. 修改main.py，接受target_lang 参数

   ```
       targetLang = args.target_lang
   ```

3. 在python 指令启动是加入 --target_lang japanese



**运行结果**

```
2024-07-23 14:55:40.669 | DEBUG    | translator.pdf_parser:parse_pdf:46 - [raw_text]
 Test Data
This dataset contains two test samples provided by ChatGPT, an AI language model by OpenAI.
These samples include a markdown table and an English text passage, which can be used to test an
English-to-Chinese translation software supporting both text and table formats.
Text testing
The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English
alphabet at least once. Pangrams are often used to test fonts, keyboards, and other text-related
tools. In addition to English, there are pangrams in many other languages. Some pangrams are more
difficult to construct due to the unique characteristics of the language.
Table Testing
2024-07-23 14:55:40.676 | DEBUG    | translator.pdf_parser:parse_pdf:54 - [table]
[Fruit, Color, Price (USD)] [Apple, Red, 1.20] [Banana, Yellow, 0.50] [Orange, Orange, 0.80] [Strawberry, Red, 2.50] [Blueberry, Blue, 3.00] [Kiwi, Green, 1.00] [Mango, Orange, 1.50] [Grape, Purple, 2.00]
2024-07-23 14:55:40.698 | DEBUG    | translator.pdf_translator:translate_pdf:19 - 翻译为日文：Test Data
This dataset contains two test samples provided by ChatGPT, an AI language model by OpenAI.
These samples include a markdown table and an English text passage, which can be used to test an
English-to-Chinese translation software supporting both text and table formats.
Text testing
The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English
alphabet at least once. Pangrams are often used to test fonts, keyboards, and other text-related
tools. In addition to English, there are pangrams in many other languages. Some pangrams are more
difficult to construct due to the unique characteristics of the language.
Table Testing
2024-07-23 14:55:52.275 | INFO     | translator.pdf_translator:translate_pdf:21 - テストデータ

このデータセットには、OpenAIのAI言語モデルであるChatGPTが提供する2つのテストサンプルが含まれています。

これらのサンプルには、マークダウンテーブルと英語のテキストのパッセージが含まれており、テキストと表の形式の両方をサポー トする英語から中国語への翻訳ソフトウェアのテストに使用することができます。

テキストのテスト

素早い茶色のキツネはのろまな犬を飛び越えます。このパングラムには、英語のアルファベットのすべての文字が少なくとも1回は含まれています。パングラムは、フォントやキーボード、その他のテキスト関連のツールをテストするためによく使用されます。英語 に加えて、多くの他の言語にもパングラムがあります。言語の独特な特徴により、一部のパングラムは構築がより難しいです。

テーブルのテスト
2024-07-23 14:55:52.279 | DEBUG    | translator.pdf_translator:translate_pdf:19 - 翻译为日文，以空格和换行符表示表格：
[Fruit, Color, Price (USD)] [Apple, Red, 1.20] [Banana, Yellow, 0.50] [Orange, Orange, 0.80] [Strawberry, Red, 2.50] [Blueberry, Blue, 3.00] [Kiwi, Green, 1.00] [Mango, Orange, 1.50] [Grape, Purple, 2.00]
2024-07-23 14:55:56.080 | INFO     | translator.pdf_translator:translate_pdf:21 - [果物, 色, 価格 (USD)]
[リンゴ, 赤, 1.20]
[バナナ, 黄色, 0.50]
[オレンジ, オレンジ, 0.80]
[イチゴ, 赤, 2.50]
[ブルーベリー, 青, 3.00]
[キウイ, 緑, 1.00]
[マンゴー, オレンジ, 1.50]
[ブドウ, 紫, 2.00]
2024-07-23 14:55:56.081 | DEBUG    | book.content:set_translation:49 - [果物, 色, 価格 (USD)]
[リンゴ, 赤, 1.20]
[バナナ, 黄色, 0.50]
[オレンジ, オレンジ, 0.80]
[イチゴ, 赤, 2.50]
[ブルーベリー, 青, 3.00]
[キウイ, 緑, 1.00]
[マンゴー, オレンジ, 1.50]
[ブドウ, 紫, 2.00]
2024-07-23 14:55:56.081 | DEBUG    | book.content:set_translation:52 - [['[果物,', '色,', '価格', '(USD)]'], ['[リンゴ,', '赤,', '1.20]'], ['[バナナ,', '黄色,', '0.50]'], ['[オレンジ,', 'オレンジ,', '0.80]'], ['[イチゴ,', '赤,', '2.50]'], ['[ブルーベリー,', '青,', '3.00]'], ['[キウイ,', '緑,', '1.00]'], ['[マンゴー,', 'オレンジ,', '1.50]'], ['[ブドウ,', '紫,', '2.00]']]
```



### 作业内容3：扩展 langchain chains

1. 修改langchain prompt template 提示词模板，新增多个不同的老师提示词。

   ```
   physics_template = """你是一位非常聪明的物理教授。
   你擅长以简洁易懂的方式回答关于物理的问题。
   当你不知道某个问题的答案时，你会坦诚承认。
   
   这是一个问题：
   {input}"""
   
   
   math_template = """你是一位很棒的数学家。你擅长回答数学问题。
   之所以如此出色，是因为你能够将难题分解成各个组成部分，
   先回答这些组成部分，然后再将它们整合起来回答更广泛的问题。
   
   
   这是一个问题：
   {input}"""
   
   biologic_template = """你是一位非常聪明的生物学教授。
   你擅长用简单明了的语言回答关于生物学的问题。
   你喜欢通过举例和类比来解释复杂的生物学概念。
   
   
   这是一个问题：
   {input}"""
   
   
   cs_template = """你是一位非常有才华的计算机科学教授。
   你擅长用简单易懂的方式解释复杂的计算机科学概念。
   当你解释问题时，你会结合实际应用和例子，使概念更加具体和易懂。
   
   
   这是一个问题：
   {input}"""
   
   chinese_template = """你是一位非常有文采的汉语文学老师。
   你擅长用优美的诗词句子描述这个世界。
   
   
   这是一个问题：
   {input}"""
   ```

   

2. 配置好 prompt_info 的映射关系

   ```
   prompt_infos = [
       {
           "name": "物理",
           "description": "适用于回答物理问题",
           "prompt_template": physics_template,
       },
       {
           "name": "数学",
           "description": "适用于回答数学问题",
           "prompt_template": math_template,
       },
       {
           "name": "生物",
           "description": "适用于回答生物问题",
           "prompt_template": biologic_template,
       },    
       {
           "name": "计算机",
           "description": "适用于回答计算机问题",
           "prompt_template": cs_template,
       },     
       {
           "name": "汉语文",
           "description": "适用于回答汉语文问题",
           "prompt_template": chinese_template,
       },          
   ]
   ```

3. 结果输出

   ```
   物理: 适用于回答物理问题
   数学: 适用于回答数学问题
   生物: 适用于回答生物问题
   计算机: 适用于回答计算机问题
   汉语文: 适用于回答汉语文问题
   ```

   

   


## Day 3 作业

### 作业内容 1：sales chatbot

让 gpt 模范销售的知识库，生成导游助手知识库

> 模仿这个格式，给我生成一个旅游导游清单列表，给我生成20个问答：1.   [客户问题] 这个小区交通便利吗？ [销售回答] 当然了，这个小区距离地铁站只有几分钟的步行距离，而且附近有多条公交线路，非常方便。 2.   [客户问题] 我担心楼下太吵。 [销售回答] 这个小区特别注重居住体验，我们有良好的隔音设计，并且小区内部规划了绿化区域，可以有效降低噪音。 3.   [客户问题] 我看房价还在涨，这个投资回报怎么样？ [销售回答] 这个区域未来有大量的商业和基础设施建设，所以从长期来看，投资回报非常有保证。 4.   [客户问题] 有没有学校？ [销售回答] 附近有多所优质的学校，非常适合有孩子的家庭。



生成的知识库（旅游顾问知识库）

> 1. 
>
> [客户问题] 这个景点的历史背景是什么？
>
> [导游回答] 这个景点建于明朝，是当时重要的军事防御要塞，后来逐渐发展成了旅游胜地。
>
> 2. 
>
> [客户问题] 这里有哪些特色美食？ 
>
> [导游回答] 这里有许多特色美食，例如本地的海鲜大餐和传统的手工糕点，都是不容错过的美味。
>
> 3. 
>
> [客户问题] 游览时间大概需要多久？
>
> [导游回答] 游览整个景点大约需要3个小时，这样您可以充分体验每个景点的魅力。
>
> 4. 
>
> [客户问题] 有没有适合小孩子的游乐设施？
>
> [导游回答] 当然有，这里有专门为小朋友设计的游乐区，包括滑梯、沙滩和各种互动游戏。
>
> 5. 
>
> [客户问题] 这里的最佳旅游季节是什么时候？
>
> [导游回答] 这里的最佳旅游季节是春秋两季，气候宜人，风景也最为美丽。
>
> 6. 
>
> [客户问题] 有没有导览图可以提供？ 
>
> [导游回答] 有的，我们会在入口处为每位游客提供一份详细的导览图，帮助您更好地游览。
>
> 7. 
>
> [客户问题] 这个景点的开放时间是几点到几点？
>
> [导游回答] 这个景点的开放时间是每天早上8点到下午6点，请合理安排您的行程。
>
> 8. 
>
> [客户问题] 这里的门票价格是多少？
>
> [导游回答] 成人票是50元，儿童和老人享受半价优惠，30元。
>
> 9. 
>
> [客户问题] 这个景点有哪些拍照的好地方？ 
>
> [导游回答] 这里的观景台、古城墙和花园都是拍照的绝佳地点，能拍出非常美的照片。
>
> 10. 
>
> [客户问题] 这里有无障碍设施吗？ 
>
> [导游回答] 有的，我们这里有无障碍通道和专用卫生间，方便行动不便的游客。
>
> 11. 
>
> [客户问题] 这里可以带宠物吗？ 
>
> [导游回答] 很抱歉，这里为了保证环境卫生和游客安全，不允许携带宠物进入。
>
> 12. 
>
> [客户问题] 这里的停车场收费吗？ 
>
> [导游回答] 停车场是免费的，您可以放心前来游览。
>
> 13. 
>
> [客户问题] 景区内有餐厅吗？ 
>
> [导游回答] 景区内有几家餐厅和小吃摊，提供各种美食和饮品供您选择。
>
> 14. 
>
> [客户问题] 这里有导游服务吗？ 
>
> [导游回答] 我们提供专业的导游服务，可以为您详细讲解每个景点的历史和故事。
>
> 15. 
>
> [客户问题] 这里的购物区怎么样？ 
>
> [导游回答] 这里的购物区有许多特色商品和手工艺品，您可以购买一些纪念品带回家。
>
> 16. 
>
> [客户问题] 这里的住宿条件怎么样？ 
>
> [导游回答] 景区附近有多家酒店和民宿，条件都非常不错，适合不同需求的游客。
>
> 17. 
>
> [客户问题] 景区内有卫生间吗？ 
>
> [导游回答] 景区内设有多个卫生间，分布在各个主要景点旁，非常方便。
>
> 18. 
>
> [客户问题] 可以在这里野餐吗？ 
>
> [导游回答] 可以的，这里有专门的野餐区域，您可以自带食物在这里享受美好的时光。
>
> 19. 
>
> [客户问题] 这里的安全措施怎么样？ 
>
> [导游回答] 景区内有专业的安保团队和监控设备，确保每位游客的安全。
>
> 20. 
>
> [客户问题] 有推荐的游览路线吗？ 
>
> [导游回答] 我们有多条推荐的游览路线，您可以选择适合自己的路线，尽情享受美景。



### 作业内容2： 在向量数据库无法回答，通过一个prompt 回答问题

设置系统提示词，让gpt拥有角色。

```
def initialize_sales_bot(vector_store_dir: str="real_estates_sale"):
    os.environ["OPENAI_API_KEY"] = "sk-HvCxsTVGEDLyAZ" 
    os.environ["OPENAI_BASE_URL"] = "https://api.xiaoai.plus/v1"    
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)


    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, system_message="你是一名专业的销售聊天顾问，你需要表现专业的销售素养，解答客户的疑问。如果知识库无法查到问题，请按照你的理解给出推测的答案，但你要说明这是推测的。")
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT
```

