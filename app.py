from pathlib import Path

import pandas as pd
import streamlit as st

from query_enhance import suggest_query
from search_engine import CourseSearchEngine
from text_utils import highlight_terms


DOCS_FOLDER = Path(__file__).parent / "data" / "sample_docs"


@st.cache_resource
def load_engine(docs_folder):
    engine = CourseSearchEngine(docs_folder)
    engine.build()
    return engine


st.set_page_config(
    page_title="中文课程资料混合检索系统",
    page_icon="🔎",
    layout="wide",
)

st.title("中文课程资料混合检索系统")
st.caption("BM25 关键词检索 + TF-IDF 向量空间检索 + 查询增强")

engine = load_engine(str(DOCS_FOLDER))

with st.sidebar:
    st.header("检索设置")
    st.metric("已加载文档", len(engine.documents))
    st.metric("文本片段 chunks", len(engine.chunks))

    mode = st.radio("检索模式", ["BM25", "TF-IDF", "Hybrid"], index=2)
    alpha = 0.5
    if mode == "Hybrid":
        alpha = st.slider("BM25 权重", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    top_k = st.slider("Top-K 结果数量", min_value=1, max_value=10, value=5, step=1)

query = st.text_input(
    "请输入查询",
    placeholder="例如：BM25 如何计算相关性",
)

query_terms = engine.get_query_terms(query) if query.strip() else []

with st.sidebar:
    st.subheader("当前查询分词")
    if query_terms:
        st.write(" / ".join(query_terms))
    else:
        st.info("输入查询后显示分词结果")

if not engine.chunks:
    st.warning("未在 data/sample_docs/ 目录下找到 .txt 或 .md 文档，请先添加课程资料文件。")
elif not query.strip():
    st.info("请输入一个中文课程资料查询，例如：倒排索引有什么作用")
else:
    suggestion = suggest_query(query)
    if suggestion:
        st.info(f"你是否想搜索：{suggestion}")

    if mode == "BM25":
        results = engine.search_bm25(query, top_k=top_k)
    elif mode == "TF-IDF":
        results = engine.search_tfidf(query, top_k=top_k)
    else:
        results = engine.search_hybrid(query, top_k=top_k, alpha=alpha)

    if not results or all(result["hybrid_score"] == 0 for result in results):
        st.warning("未找到高相关结果。你可以尝试更换关键词，或查看查询增强提示。")

    st.subheader("检索结果")
    for result in results:
        title = (
            f"Top {result['rank']} | source: {result['source_file']} | "
            f"hybrid_score={result['hybrid_score']:.3f}"
        )
        with st.expander(title, expanded=result["rank"] == 1):
            highlighted_text = highlight_terms(result["text"], result["matched_terms"])
            st.markdown(highlighted_text, unsafe_allow_html=True)

            score_df = pd.DataFrame(
                [
                    {
                        "BM25 原始分数": round(result["bm25_score"], 4),
                        "BM25 归一化分数": round(result["bm25_norm"], 4),
                        "TF-IDF 分数": round(result["tfidf_score"], 4),
                        "混合分数": round(result["hybrid_score"], 4),
                    }
                ]
            )
            st.dataframe(score_df, hide_index=True, use_container_width=True)

            matched_terms = result["matched_terms"]
            if matched_terms:
                st.write("命中关键词：", " / ".join(matched_terms))
            else:
                st.write("命中关键词：无")

st.divider()
st.subheader("系统说明")
st.markdown(
    """
- BM25 更重视关键词精确匹配，并结合词频、逆文档频率和文档长度归一化进行相关性排序。
- TF-IDF 使用向量空间模型计算查询与文本片段之间的余弦相似度。
- 混合检索将 BM25 与 TF-IDF 分数归一化后加权融合，兼顾精确匹配和整体相关性。
- 查询增强用于展示分词结果、提供简单错别字建议，并在结果中高亮命中关键词。
"""
)
