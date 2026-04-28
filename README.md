# 中文课程资料混合检索系统

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

