# 中文课程资料混合检索系统

本项目是《信息检索原理》课程作业 demo，主题为“基于混合检索与查询增强的课程资料检索系统设计”。系统完全本地运行，读取 `data/sample_docs/` 目录中的 `.txt` 和 `.md` 文本文件，完成文本切片、中文分词、索引构建、排序检索、查询增强和结果解释。

项目不依赖大模型、数据库或在线服务，适合用于 5-10 分钟课堂展示或录屏演示。

## 功能特点

- 读取本地中文课程资料文件。
- 按段落切分文本片段 chunk。
- 使用 `jieba` 进行中文分词，并过滤简单停用词。
- 支持 BM25 关键词检索。
- 支持 TF-IDF 向量空间检索。
- 支持 BM25 与 TF-IDF 加权融合的混合检索。
- 提供课程术语词典和简单错别字纠错建议。
- 展示查询分词、命中关键词高亮和检索分数解释。

## 项目结构

```text
ChineseCourseSearch/
├── app.py
├── search_engine.py
├── text_utils.py
├── query_enhance.py
├── requirements.txt
├── README.md
└── data/
    └── sample_docs/
        ├── inverted_index.md
        ├── bm25.md
        ├── vector_space_model.md
        └── query_expansion.md
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方式

```bash
streamlit run app.py
```

运行后在浏览器中打开 Streamlit 提供的本地地址即可使用。

## 示例查询

- 倒排索引有什么作用
- BM25 如何计算相关性
- 向量空间模型和 TF-IDF 有什么关系
- 查询扩展有什么用
- 倒排锁引

## 和信息检索课程的关系

本项目围绕信息检索系统的基础流程展开，体现了以下课程知识点：

- 文档预处理
- 中文分词
- 文本切片
- BM25 排序
- TF-IDF 向量空间模型
- 混合检索
- 查询纠错
- 结果解释

通过这个 demo，可以直观看到从原始课程资料到检索结果排序的完整流程，也可以比较关键词检索、向量空间检索和混合检索在不同查询下的表现差异。
