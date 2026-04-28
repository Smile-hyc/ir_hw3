from difflib import get_close_matches


COURSE_TERMS = [
    "倒排索引",
    "倒排表",
    "词项",
    "文档列表",
    "布尔检索",
    "BM25",
    "TF-IDF",
    "向量空间模型",
    "余弦相似度",
    "相关性排序",
    "查询扩展",
    "伪相关反馈",
    "编辑距离",
    "分词",
    "停用词",
    "召回率",
    "准确率",
]


def suggest_query(query):
    """Return the closest course term as a correction suggestion, or None."""
    normalized_query = query.strip()
    if not normalized_query:
        return None

    if normalized_query in COURSE_TERMS:
        return None

    lower_query = normalized_query.lower()
    candidates = []
    for term in COURSE_TERMS:
        if term.lower() in lower_query:
            continue
        matches = get_close_matches(normalized_query, [term], n=1, cutoff=0.45)
        if matches:
            candidates.append(matches[0])

    if candidates:
        return candidates[0]

    # For longer natural-language queries, compare each short segment with terms.
    for term in COURSE_TERMS:
        if term.lower() in lower_query:
            continue
        matches = get_close_matches(term, [normalized_query], n=1, cutoff=0.45)
        if matches:
            return term

    return None
