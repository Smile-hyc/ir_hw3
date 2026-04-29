import re
from difflib import SequenceMatcher, get_close_matches


COURSE_TERMS = [
    "倒排索引",
    "倒排表",
    "倒排文件",
    "词项",
    "文档列表",
    "布尔检索",
    "BM25",
    "TF-IDF",
    "向量空间模型",
    "余弦相似度",
    "相关性排序",
    "查询扩展",
    "查询改写",
    "伪相关反馈",
    "编辑距离",
    "中文分词",
    "停用词",
    "召回率",
    "准确率",
    "词频",
    "逆文档频率",
    "文档长度归一化",
]


EXPANSION_MAP = {
    "倒排索引": ["倒排表", "倒排文件", "词项", "文档列表", "索引结构"],
    "BM25": ["词频", "逆文档频率", "文档长度归一化", "相关性排序"],
    "TF-IDF": ["词频", "逆文档频率", "权重计算", "向量空间模型"],
    "向量空间模型": ["TF-IDF", "余弦相似度", "文本向量化", "相似度计算"],
    "查询扩展": ["查询改写", "相关词扩展", "伪相关反馈", "用户查询意图"],
    "布尔检索": ["AND", "OR", "NOT", "逻辑运算符", "精确匹配"],
    "中文分词": ["词项", "停用词", "文本预处理", "分词结果"],
    "相关性排序": ["BM25", "TF-IDF", "混合检索", "排序分数"],
}


def _contains_term(query: str, term: str) -> bool:
    return term.lower() in query.lower()


def _correction_cutoff(term: str) -> float:
    if len(term) <= 3:
        return 0.75
    if len(term) >= 5:
        return 0.6
    return 0.65


def _query_fragments(query: str) -> list[str]:
    """Build short query fragments so embedded Chinese typos can be compared."""
    compact_query = re.sub(r"\s+", "", query.strip())
    fragments = [query.strip()]

    for part in re.split(r"[\s,.;:!?，。；：！？、（）()《》<>]+", query):
        part = part.strip()
        if len(part) >= 2:
            fragments.append(part)

    for term in COURSE_TERMS:
        term_len = len(term)
        min_len = term_len if term_len <= 4 else term_len - 1
        if term_len <= 2 or len(compact_query) < min_len:
            continue

        max_len = min(len(compact_query), term_len + 1)
        for window_len in range(min_len, max_len + 1):
            for start in range(0, len(compact_query) - window_len + 1):
                fragments.append(compact_query[start : start + window_len])

    seen = set()
    unique_fragments = []
    for fragment in fragments:
        normalized = fragment.lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_fragments.append(fragment)
    return unique_fragments


def suggest_correction(query: str) -> str | None:
    """Return a course-term correction suggestion without changing the query."""
    normalized_query = query.strip()
    if not normalized_query:
        return None

    # If the query already contains any valid course term, avoid over-correction.
    if any(_contains_term(normalized_query, term) for term in COURSE_TERMS):
        return None

    best_term = None
    best_score = 0.0

    for fragment in _query_fragments(normalized_query):
        fragment_lower = fragment.lower()
        for term in COURSE_TERMS:
            term_lower = term.lower()
            matches = get_close_matches(
                fragment_lower,
                [term_lower],
                n=1,
                cutoff=_correction_cutoff(term),
            )
            if not matches:
                continue

            score = SequenceMatcher(None, fragment_lower, term_lower).ratio()
            if score > best_score:
                best_score = score
                best_term = term

    return best_term


def get_expansion_terms(query: str) -> list[str]:
    """Return up to five related course terms for query expansion."""
    normalized_query = query.strip()
    if not normalized_query:
        return []

    matched_keys = [
        key for key in EXPANSION_MAP if _contains_term(normalized_query, key)
    ]

    if not matched_keys:
        correction = suggest_correction(normalized_query)
        if correction in EXPANSION_MAP:
            matched_keys = [correction]

    expansion_terms = []
    seen = set()
    query_lower = normalized_query.lower()

    for key in matched_keys:
        for term in EXPANSION_MAP[key]:
            term_lower = term.lower()
            if term_lower in query_lower or term_lower in seen:
                continue
            seen.add(term_lower)
            expansion_terms.append(term)
            if len(expansion_terms) >= 5:
                return expansion_terms

    return expansion_terms


def build_expanded_query(query: str, expansion_terms: list[str]) -> str:
    """Append expansion terms to the original query with spaces."""
    base_query = query.strip()
    query_lower = base_query.lower()
    terms = []
    seen = set()

    for term in expansion_terms:
        clean_term = term.strip()
        term_lower = clean_term.lower()
        if not clean_term or term_lower in query_lower or term_lower in seen:
            continue
        seen.add(term_lower)
        terms.append(clean_term)

    if not base_query:
        return " ".join(terms)
    if not terms:
        return base_query
    return " ".join([base_query, *terms])


def suggest_query(query: str) -> str | None:
    """Backward-compatible alias for the old app import."""
    return suggest_correction(query)
