from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, AgentExecutor, create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain import hub
from typing import List, Dict, Any
import configparser
import pandas as pd
import re
import os
import torch
from transformers import AutoTokenizer, AutoModel
from bert_score import BERTScorer


# ========= 这里是你已有的各种 import，如 CSVLoader, DataFrameLoader, 以及 bert_score 等 =============
from langchain_community.document_loaders import TextLoader, CSVLoader, DataFrameLoader
from langchain.tools.retriever import create_retriever_tool
# from transformers import AutoTokenizer, AutoModel
# from bert_score import BERTScorer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_model = AutoModel.from_pretrained("bert-base-multilingual-cased")
## Initialize the BERTScorer for multilingual BERT or a Japanese-specific BERT
scorer = BERTScorer(model_type="bert-base-multilingual-cased", lang="ja", device='cuda' if torch.cuda.is_available() else 'cpu')
from load_and_embed import custermized_trend_retriever, custermized_retriever

# ... 等等
# ========= 省略 ===========

from utils import run_with_retries
from metrics import find_most_relevant_keywords, update_clicks

class reflextion_agent:
    def __init__(self, config_file='./config_3_day_obs.ini'):
        self.config = configparser.ConfigParser()
        try:
            self.config.read(config_file)
        except Exception as e:
            raise ValueError("Failed to read the configuration file: " + str(e))

        self.observation_period = int(self.config['SYSTEM']['OBSERVATION_PERIOD'])
        self.csv_file_path = self.config['FILE']['CSV_FILE']
        self.setting_day = pd.to_datetime(self.config['SYSTEM']['SETTING_DAY'])
        self.dataframe = pd.read_csv(str(self.config['FILE']['CSV_FILE']))

        # 载入 df_score, 做一些初始筛选
        product_name = str(self.config['CAMPAIGN']['PRODUCT_NAME'])
        if product_name == 'ソニーテレビ ブラビア':
            self.df_score = pd.read_csv('./preprocessing/data/score_data/rakkokeyword_sony_bravia.csv',
                                        delimiter='\t', quotechar='"', encoding='utf-16')
        elif product_name == 'ソニー損保 医療保険':
            self.df_score = pd.read_csv('./preprocessing/data/score_data/rakkokeyword_sony_insurance.csv',
                                        delimiter='\t', quotechar='"', encoding='utf-16')
        elif product_name == 'ソニーデジタル一眼カメラ α（アルファ）':
            self.df_score = pd.read_csv('./preprocessing/data/score_data/rakkokeyword_sony_camera.csv',
                                        delimiter='\t', quotechar='"', encoding='utf-16')
        elif product_name == 'ソニー銀行 住宅ローン':
            self.df_score = pd.read_csv('./preprocessing/data/score_data/rakkokeyword_sony_bank_morgage.csv',
                                        delimiter='\t', quotechar='"', encoding='utf-16')
        elif product_name == 'ソニー Prediction One':
            self.df_score = pd.read_csv('./dataset/sony_prediction_one.csv', delimiter='\t')
        else:
            raise ValueError("Failed to read the PRODUCT_NAME: " + product_name)
        # 仅保留前130行示例
        self.df_score = self.df_score.iloc[:130]

        # 设置 SERPAPI / TAVILY 的 KEY。这里替换成 TAVILY
        os.environ['TAVILY_API_KEY'] = self.config['KEY']['SERPAPI_API_KEY']  # 你可以写成 self.config['KEY']['TAVILY_API_KEY']
        self.product_name = product_name

        # 初始化 TAVILY 搜索工具
        self.tavi_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=True
        )
        # 初始化 LLM
        # 注: 这里用 ChatOpenAI / gpt-4o，如果你想用 AzureChatOpenAI 也可以。
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # 你也可以把 agent_tool 的初始化放这里，
        # 但本示例中演示 generate_agent 和 reflect_agent 分开写。
    def build_tools_for_generation(self):
        """
        构建多个Retriever工具 + TAVILY 工具，让大模型在生成时可以访问：
          - KW_retriever: 可能是CSV里加载一些初始关键字库
          - exampler_retriever: 可能是TextLoader里加载“关键字示例与一般规则”
          - TAVILY: 搜索工具
        返回三个 Tool 对象
        """
        # 1) KW_retriever_tool
        #   假设你在 './current_KW.csv' 里存了一些当前的关键词信息
        if str(self.config['CAMPAIGN']['PRODUCT_NAME']) == 'ソニーテレビ ブラビア':
            kw_str = "initial_KW_sony_bravia.csv"
        elif str(self.config['CAMPAIGN']['PRODUCT_NAME']) == 'ソニー損保 医療保険':
            kw_str = "initial_KW_sony_insurance.csv"
        elif str(self.config['CAMPAIGN']['PRODUCT_NAME']) == 'ソニーデジタル一眼カメラ α（アルファ）':
            kw_str = "initial_KW_sony_camera.csv"
        elif str(self.config['CAMPAIGN']['PRODUCT_NAME']) == 'ソニー銀行 住宅ローン':
            kw_str = "initial_KW_sony_bank_morgage.csv"
        elif str(self.config['CAMPAIGN']['PRODUCT_NAME']) == 'ソニー Prediction One':
            kw_str = "initial_KW_sony_prediction_one.csv"
        KW_loader = CSVLoader(f'preprocessing_data/kw_data/{kw_str}')
        df = pd.read_csv(f'preprocessing_data/kw_data/{kw_str}')
        #   需要embedding key:
        kw_retriever = custermized_trend_retriever(
            KW_loader,
            str(self.config['KEY']['OPENAI_EMBEDDING_API_KEY']),
            str(self.config['KEY']['OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT'])
        )
        KW_retriever_tool = create_retriever_tool(
            kw_retriever,
            tool_name=str(self.config['TOOL']['GOOD_KW_RETRIEVAL_NAME']),
            description=str(self.config['TOOL']['GOOD_KW_RETRIEVAL_DISCRPTION'])
        )

        # 2) exampler_retriever_tool
        #   假设你在 ./some_example_rules.txt 中存放了“通用关键字规则和例子”
        exampler_loader = TextLoader(str(self.config['FILE']['EXAMPLER_FILE']))
        exampler_retriever = custermized_trend_retriever(
            exampler_loader,
            str(self.config['KEY']['OPENAI_EMBEDDING_API_KEY']),
            str(self.config['KEY']['OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT']))
        exampler_retriever_tool = create_retriever_tool(
            exampler_retriever,
            tool_name=str(self.config['TOOL']['RULE_RETRIEVAL_NAME']),
            description=str(self.config['TOOL']['RULE_RETRIEVAL_DISCRPTION'])
        )

        # 3) TavilySearchResults 工具(已在 __init__ 中初始化)
        #   但是langchain需要一个Tool对象。可以手动包装一下:
        #   但是在最新的 langchain_community 里 TavilySearchResults 通常已经是满足 Tool 协议的对象
        #   如果不是，就需要自定义一个Tool子类来封装 self.tavi_tool.run(query)
        #   假设 self.tavi_tool 已经是Tool
        tavi_tool = self.tavi_tool  # 直接拿过来

        return [KW_retriever_tool, exampler_retriever_tool, tavi_tool]

    def generate_agent(self, rejected_kw_list, good_kw_list, step) -> dict:
        """
        使用生成 Agent (多工具: KW_retriever_tool, exampler_retriever_tool, tavi_tool)
        与 LLM 进行多轮对话 / ReAct，最后产出新关键词字典
        """
        # 构造System和Human提示
        system_prompt = SystemMessage(content=f"""
あなたは {self.product_name} の広告キーワードを作成するエキスパートです。
以下のツールを使って必要な情報を取得できます:
1) KW_Retriever: 現在のCSVからキーワードを読み取る
2) Example_Retriever: 一般的なキーワード設定例やルールのドキュメント
3) TavilySearchResults: ウェブ上で検索
""")

        # 超参
        n_kw_per_cat = int(self.config['KEYWORD']['NUM_KEYWORDS_PER_CATEGORY'])
        n_new_cat = int(self.config['KEYWORD']['NUM_NEW_CATEGORIES'])
        n_kw_per_newcat = int(self.config['KEYWORD']['NUM_KEYWORDS_PER_NEW_CATEGORY'])

        # 把rejected和good变成字符串
        rejected_str = ", ".join(rejected_kw_list)
        good_str = ", ".join(good_kw_list)

        # 构造HumanMessage
        human_prompt = f"""
以下の条件で新しいキーワードを生成してください:
1. 現在のキーワードを分析し、各カテゴリにつき {n_kw_per_cat} 個の新キーワード候補を提案する
2. 新しいカテゴリを {n_new_cat} 個提案し、それぞれ {n_kw_per_newcat} 個のキーワードを作る
3. 以下のキーワードは生成しないこと: {rejected_str}
4. 以下のキーワードは既に有望なので、重複しない範囲で再利用してもよい: {good_str}
5. 出力は Python の dict 形式 {{"カテゴリ名": ["キーワードA","キーワードB", ...], ...}} とする
"""

        human_message = HumanMessage(content=human_prompt)

        # 构建多个 Tool
        tools = self.build_tools_for_generation()
        # 根据 REACT_DOCSTORE 或 ZERO_SHOT_REACT_DESCRIPTION 初始化 agent
        agent_chain = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            return_intermediate_steps=True
        )

        # 调用 run_with_retries
        final_answer, scratchpad = run_with_retries(agent_chain, [system_prompt, human_message], max_attempts=3)

        # 解析 final_answer (字符串 -> dict)
        try:
            new_keywords_dict = eval(final_answer)
        except:
            new_keywords_dict = {}

        return new_keywords_dict

    def reflect_agent(self, new_keywords_dict: dict) -> dict:
        """
        对 new_keywords_dict 做Reflection(评估打分/改进建议等)
        和之前示例一样
        """
        # 把 new_keywords_dict 转成文本
        keywords_text = ""
        for cat, kws in new_keywords_dict.items():
            keywords_text += f"【{cat}】\n- " + "\n- ".join(kws) + "\n"

        system_prompt = SystemMessage(content="あなたはキーワード評価のエキスパートです。")
        human_content = f"""
以下のキーワードセットについて、3つの評価基準で採点し、フィードバックと改良提案を述べてください。
```
{keywords_text}
```
Evaluation:
1) 商品適合度 (1-10)
2) 検索意図カバー度 (1-10)
3) 冗長性 (1-10) （スコアが低いほど冗長）

Output Format:
Evaluation Score:
  商品適合度: X/10
  検索意図カバー度: Y/10
  冗長性: Z/10
Feedback:
  ...
Improvement Suggestions:
  ...
Further Improvement Needed: [Yes/No]
"""
        human_prompt = HumanMessage(content=human_content)

        reflection_response = self.llm([system_prompt, human_prompt])
        content = reflection_response.content

        reflection_result = {
            "evaluation_scores": {},
            "feedback": "",
            "improvement_suggestions": "",
            "improvement_needed": False
        }

        # 解析示例(可自行替换 parse_reflection_result)
        lines = content.split("\n")
        section = None
        for line in lines:
            if "Evaluation Score" in line:
                section = "scores"
                continue
            elif "Feedback" in line:
                section = "feedback"
                continue
            elif "Improvement Suggestions" in line:
                section = "suggestions"
                continue
            elif "Further Improvement Needed" in line:
                section = "needed"
                continue

            if section == "scores" and ":" in line:
                # 例: "商品適合度: 8/10"
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val_str = re.sub(r"/\d+", "", parts[1]).strip()
                    try:
                        val = int(val_str)
                    except:
                        val = 0
                    reflection_result["evaluation_scores"][key] = val
            elif section == "feedback":
                reflection_result["feedback"] += line
            elif section == "suggestions":
                reflection_result["improvement_suggestions"] += line
            elif section == "needed":
                if "Yes" in line:
                    reflection_result["improvement_needed"] = True

        return reflection_result

    def run(self):
        """
        综合入口：
        1) 检查日期 / CSV / DataFrame
        2) 如果在可执行的日期，启动多轮循环： generate_agent -> reflect_agent -> decide break or continue
        3) 写入结果 CSV
        """

        # 0. 如果 self.setting_day 不满足某些条件就直接返回 (如原代码)
        if self.setting_day.day not in [4, 8, 12, 16, 20, 24, 28]:
            print("Not in the scheduled day, no generation performed.")
            return {}

        # 初始化 rejected_kw_list / good_kw_list
        if not os.path.exists('./preprocessing/data/string_list.txt'):
            rejected_kw_list = []
        else:
            with open('./preprocessing/data/string_list.txt', 'r') as file:
                rejected_kw_list = [line.strip() for line in file]
        good_kw_list = []
        df = self.dataframe
        df = df [["Keyword", "Match type", "Category", "Clicks"]]
        df = df[df['Match type'] == 'Phrase match']
        # remove "" from the 'Keyword' column
        df['Keyword'] = df['Keyword'].str.replace('"', '')
        # remove the colomn of 'Match type'
        df = df.drop(columns=['Match type'])
        # save it to a new csv file
        df.to_csv('./current_KW.csv', index=False)

        mean_score = 0
        mean_jacard_score = 0
        mean_cosine_score = 0
        mean_bert_score = 0
        mean_search_volume = 0
        mean_competitor_score = 0
        mean_cpc = 0

        iteration_count = 0
        max_rounds = int(self.config['EXE']['GENERATION_ROUND'])

        all_generated = []
        step = 0
        while True:
            print(f"\n===== Start generate round {iteration_count + 1} =====")

            # 1) 生成新的关键词
            new_kw_dict = self.generate_agent(rejected_kw_list, good_kw_list, step)
            # 记录一下
            all_generated.append(new_kw_dict)

            # 2) 做 reflection
            reflection_result = self.reflect_agent(new_kw_dict)
            print("Reflection result:", reflection_result)

            # 3) 如果不需要改进，就 break
            if not reflection_result["improvement_needed"]:
                print("No further improvement needed, break.")
                # 在这里可以把最终关键词写入 CSV
                self.save_new_keywords(new_kw_dict)
            else:
                print("Evaluator suggests improvement, continue next round...")
                # 也可以在这里，把低分关键词加入 rejected，或者把好关键词加入 good_kw_list
                # 示例：如果全部都低的话，就 rejected，反之则 good_kw_list
                # 这里只演示一个简化逻辑
                for cat, kws in new_kw_dict.items():
                    # 假装全部一并放好
                    good_kw_list.extend(kws)

                self.save_new_keywords(new_kw_dict)
                iteration_count += 1

            # 如果到了最大轮次，还没 break，也可以写 final CSV
            if iteration_count == max_rounds:
                print("Reached max rounds, finish.")
            return all_generated

    def save_new_keywords(self, new_kw_dict: Dict[str, List[str]]):
        """
        把新生成的关键词写入 whole_kw.csv 或做更新 DataFrame 的操作
        这里参考你原先 run() 里写 CSV 的逻辑即可
        """
        # 先把 new_kw_dict 转成 DF
        new_keywords_df = pd.DataFrame(
            [(cat, kw) for cat, kw_list in new_kw_dict.items() for kw in kw_list],
            columns=['Category', 'Keyword']
        )
        # 先判断旧 DF
        if not os.path.exists('/home/ubuntu/reflexion/New_LLM_Agent_4_Ad_Keyword_and_Text/preprocessing/data/whole_kw.csv'):
            # 如果不存在，就直接写入
            new_keywords_df.to_csv('/home/ubuntu/reflexion/New_LLM_Agent_4_Ad_Keyword_and_Text/preprocessing/data/whole_kw.csv', index=False)
            print("New keywords written to CSV.")
        else:
            df = pd.read_csv('/home/ubuntu/reflexion/New_LLM_Agent_4_Ad_Keyword_and_Text/preprocessing/data/whole_kw.csv')

        # 判断是否新老Category
        existing_cats = df['Category'].unique()
        new_keywords_df['Category Status'] = new_keywords_df['Category'].apply(
            lambda c: 'deeper' if c in existing_cats else 'wider'
        )
        # 合并
        combined_df = pd.concat([df, new_keywords_df], ignore_index=True)
        combined_df['Clicks'] = combined_df['Clicks'].fillna(0)

        # 用 metrics 相关逻辑把 df_score 进行合并, 例如:
        all_new_kws = new_keywords_df["Keyword"].tolist()
        results = find_most_relevant_keywords(all_new_kws, self.df_score, 'キーワード', '推定流入数')
        updated_df = update_clicks(combined_df, results, 'Estimated Traffic')

        updated_df.to_csv('/home/ubuntu/reflexion/New_LLM_Agent_4_Ad_Keyword_and_Text/preprocessing/data/whole_kw.csv', index=False)
        print("New keywords appended to CSV.")
