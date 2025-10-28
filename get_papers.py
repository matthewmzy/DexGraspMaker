import os
from Bio import Entrez
# from google import genai # 你原来的代码
from tqdm import tqdm

# os.environ["GEMINI_API_KEY"] = "AIzaSyCqNHQtP0q9WjzCelswsaJb_By5T-Pfst8"

# client = genai.Client()
# prompt = """
# 你是一个资深的生物医学研究助理，精通 PubMed 文献检索和分析。我现在在做一个关于“成纤维细胞在脊髓损伤中的作用机理”的研究综述。现在我检索到了一篇论文，名为"{title}"，摘要如下：
# {abstract}，请你分析这篇论文的研究是否对我的综述有帮助。你的回复只能是一个单词：“yes”、“no”或者“not sure”，不能有任何额外的字符。
# """

# --- 必须设置：告诉 NCBI 你是谁 ---
Entrez.email = "matthewmzy123@gmail.com"  # 你的邮箱
Entrez.tool = "research"                 # 可以随便起个名字

# --- 辅助函数：格式化作者列表 ---
def format_authors(author_list):
    """将 Entrez 返回的作者列表字典格式化为 BibTeX 字符串"""
    if not author_list:
        return "N/A"
    
    formatted_authors = []
    for author in author_list:
        # 优先使用 LastName 和 ForeName
        last_name = author.get('LastName', '')
        fore_name = author.get('ForeName', '')
        
        if last_name and fore_name:
            formatted_authors.append(f"{last_name}, {fore_name}")
        elif last_name:
            # 备选：如果只有姓氏
            formatted_authors.append(last_name)
        elif author.get('CollectiveName'):
            # 备选：处理团体作者
            formatted_authors.append(author.get('CollectiveName'))
            
    return " and ".join(formatted_authors)

# --- 辅助函数：从ID列表中查找 DOI ---
def find_doi(article_id_list):
    """从 ArticleIdList 中查找 DOI"""
    if not article_id_list:
        return "N/A"
    
    for item in article_id_list:
        # item 是一个带 .attributes 属性的字符串对象
        if item.attributes.get('IdType') == 'doi':
            return str(item) # 返回 DOI 字符串
    return "N/A"

# --- 搜索关键词 ---
search_term = "(fibroblast[Title/Abstract]) AND (spinal cord[Title/Abstract]) AND (injury[Title/Abstract])"

try:
    print("正在搜索 PubMed...")
    # Esearch：搜索并返回论文的 ID 列表 (PMID)
    handle_search = Entrez.esearch(db="pubmed",
                                   term=search_term,
                                   retmax=450)  # 你设置的数量

    record_search = Entrez.read(handle_search)
    handle_search.close()

    id_list = record_search["IdList"] # 拿到 PMIDs 列表
    
    if not id_list:
        print("没有找到相关论文。")
        exit()

    print(f"找到了 {len(id_list)} 篇论文，正在获取详细信息...")

    # --- 获取详细信息 ---
    # Efetch：根据 ID 列表获取论文的详细记录 (XML 格式)
    handle_fetch = Entrez.efetch(db="pubmed", 
                                 id=id_list,       # 使用上面获取的 ID 列表
                                 rettype="xml")    # 获取 XML 格式

    records_fetch = Entrez.read(handle_fetch)
    handle_fetch.close()

    # helpful_papers = []
    content = "" # 用来累积所有内容的字符串

    # --- 解析并打印信息 ---
    print("正在解析论文并生成 BibTeX 引用...")
    for i, paper in tqdm(enumerate(records_fetch['PubmedArticle'])):
        # 'MedlineCitation' 和 'Article' 是 XML 记录中的标准层级
        medline_citation = paper.get('MedlineCitation', {})
        article = medline_citation.get('Article', {})
        
        # --- 基础信息提取（你原来的代码） ---
        pmid = medline_citation.get('PMID', 'N/A')
        title = article.get('ArticleTitle', 'N/A')
        
        # JournalIssue 字典
        journal_issue = article.get('Journal', {}).get('JournalIssue', {})
        pubtime = journal_issue.get('PubDate', {}) # 这是一个字典
        
        # 摘要（Abstract）
        abstract = "N/A"
        if 'Abstract' in article:
            abstract_parts = article.get('Abstract', {}).get('AbstractText', [])
            abstract = "\n".join(abstract_parts)

        # --- 【新增】为 BibTeX 提取更多字段 ---
        
        # 作者 (使用辅助函数)
        authors_list = article.get('AuthorList', [])
        authors = format_authors(authors_list)
        
        # 期刊名
        journal = article.get('Journal', {}).get('Title', 'N/A')
        
        # 年份 (PubDate 字典可能只有 Year，也可能只有 MedlineDate)
        if 'Year' in pubtime:
            year = pubtime.get('Year', 'N/A')
        elif 'MedlineDate' in pubtime:
            # MedlineDate 格式通常是 "2023 Jan" 或 "2023-2024"，取前4位
            year = pubtime.get('MedlineDate', 'N/A')[:4]
        else:
            year = 'N/A'
            
        # 卷 (Volume)
        volume = journal_issue.get('Volume', 'N/A')
        
        # 期 (Issue)
        issue = journal_issue.get('Issue', 'N/A')
        
        # 页码
        pages = article.get('Pagination', {}).get('MedlinePgn', 'N/A')
        
        # DOI (使用辅助函数)
        article_id_list = paper.get('PubmedData', {}).get('ArticleIdList', [])
        doi = find_doi(article_id_list)

        # --- 【新增】格式化 BibTeX 条目 ---
        # 我们使用 PMID 作为 BibTeX 的唯一键 (key)
        # 注意：BibTeX 字段值最好用花括号 {} 包裹，以保留大小写
        bibtex_entry = f"""
@article{{{pmid},
  author  = {{{authors}}},
  title   = {{{title}}},
  journal = {{{journal}}},
  year    = {{{year}}},
  volume  = {{{volume}}},
  number  = {{{issue}}},
  pages   = {{{pages}}},
  doi     = {{{doi}}},
  pmid    = {{{pmid}}}
}}"""

        # --- 累积 Markdown 内容 ---
        content += f"\n# --- 论文 {i+1} ---\n"
        content += f"#### {title}\n"
        content += f"*PMID*: {pmid}\n"
        # repr() 会把字典转成 '{\'Year\': \'2023\', ...}' 格式
        content += f"*发表时间*: {pubtime.__repr__()}\n" 
        content += f"*摘要*:\n {abstract}\n"
        
        # --- 【新增】将 BibTeX 条目添加到 Markdown ---
        content += "\n*BibTeX 引用*:\n"
        content += f"```bibtex\n{bibtex_entry.strip()}\n```\n"

        # (你原来的 Gemini API 部分，保持注释)
        # ...

    # --- 一次性写入文件 ---
    print(f"正在将所有 {len(id_list)} 篇论文写入 papers.md ...")
    with open("papers.md", "w", encoding="utf-8") as f:
        f.write(content)

    print("\n处理完成！")
    # print(f"\n在这些论文中，有 {len(helpful_papers)} 篇可能对你的综述有帮助")

except Entrez.HTTPError as http_err:
    print(f"发生 HTTP 错误: {http_err}")
    print("这可能是因为请求过于频繁。请稍后再试。")
except Exception as e:
    print(f"发生错误: {e}")
    print("请检查你的邮箱地址是否正确，以及网络连接是否正常。")