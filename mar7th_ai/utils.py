import json
import os
import random
import re
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Union

from EdgeGPT import Chatbot as bingChatbot
from loguru import logger
from nonebot.adapters.onebot.v11 import MessageEvent, MessageSegment
from nonebot.matcher import Matcher
from revChatGPT.V3 import Chatbot as openaiChatbot

from .config import config
from .txtToImg import txt_to_img


class Utils:
    def __init__(self) -> None:
        """初始化"""
        self.reply_private: bool = config.ai_reply_private
        self.bot_nickname: str = config.bot_nickname
        self.poke__reply: Tuple = (
            "lsp你再戳？",
            "连个可爱美少女都要戳的肥宅真恶心啊。",
            "你再戳！",
            "？再戳试试？",
            "别戳了别戳了再戳就坏了555",
            "我爪巴爪巴，球球别再戳了",
            "你戳你🐎呢？！",
            f"请不要戳{self.bot_nickname} >_<",
            "放手啦，不给戳QAQ",
            f"喂(#`O′) 戳{self.bot_nickname}干嘛！",
            "戳坏了，赔钱！",
            "戳坏了",
            "嗯……不可以……啦……不要乱戳",
            "那...那里...那里不能戳...绝对...",
            "(。´・ω・)ん?",
            "有事恁叫我，别天天一个劲戳戳戳！",
            "欸很烦欸！你戳🔨呢",
            "再戳一下试试？",
            "正在关闭对您的所有服务...关闭成功",
            "啊呜，太舒服刚刚竟然睡着了。什么事？",
            "正在定位您的真实地址...定位成功。轰炸机已起飞",
        )
        self.hello_reply: Tuple = (
            "你好！",
            "哦豁？！",
            "你好！Ov<",
            f"库库库，呼唤{config.bot_nickname}做什么呢",
            "我在呢！",
            "呼呼，叫俺干嘛",
        )
        self.nonsense: Tuple = (
            "你好啊",
            "你好",
            "在吗",
            "在不在",
            "您好",
            "您好啊",
            "你好",
            "在",
        )
        self.superuser = config.superusers
        self.module_path: Path = Path(__file__).parent
        self.keyword_path: Path = self.module_path / "resource/json/data.json"
        self.anime_thesaurus: Dict = json.load(
            open(self.keyword_path, "r", encoding="utf-8")
        )
        self.audio_path: Path = self.module_path / "resource/audio"
        self.audio_list: List[str] = os.listdir(self.audio_path)
        self.proxy = config.bing_or_openai_proxy
        # ==================================== bing工具属性 ====================================================
        # 会话字典，用于存储会话   {"user_id": {"chatbot": bot, "last_time": time, "model": "balanced", isRunning: bool}}
        self.bing_chat_dict: Dict = {}
        bing_cookies_files: List[Path] = [
            file
            for file in config.smart_reply_path.rglob("*.json")
            if file.stem.startswith("cookie")
        ]
        try:
            self.bing_cookies: List = [
                json.load(open(file, "r", encoding="utf-8"))
                for file in bing_cookies_files
            ]
            logger.success(f"bing_cookies读取, 初始化成功, 共{len(self.bing_cookies)}个cookies")
        except Exception as e:
            logger.error(f"读取bing cookies失败 error信息: {repr(e)}")
            self.bing_cookies: List = []
        # ==================================== openai工具属性 ====================================================
        # 会话字典，用于存储会话   {"user_id": {"chatbot": bot, "last_time": time, "sessions_number": 0}}
        self.openai_chat_dict: dict = {}
        self.openai_api_key: List = config.openai_api_key  # type: ignore
        self.openai_max_tokens: int = config.openai_max_tokens
        self.max_sessions_number: int = config.openai_max_conversation

        if self.proxy:
            logger.info(f"已设置代理, 值为:{self.proxy}")
        else:
            logger.warning("未检测到代理，国内用户可能无法使用bing或openai功能")
            # ==================================== 检查并下载文件 ====================================
        self.check_and_download_file()

    def check_and_download_file(self):
        # 检查目标文件是否存在
        target_file_path = "marth7th_ai/trained/mar7th_G_40000.pth"
        if not os.path.exists(target_file_path):
            # 如果文件不存在，下载文件
            download_url = "https://huggingface.co/kaze-mio/so-vits-star-rail/raw/main/mar7th/mar7th_G_40000.pth"
            try:
                response = requests.get(download_url, stream=True)
                response.raise_for_status()
                with open(target_file_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                print("mar7th_G_40000.pth 文件已下载并保存在 marth7th_ai/trained 目录下。")
            except Exception as e:
                print(f"下载 mar7th_G_40000.pth 文件时发生错误：{str(e)}，请根据readme手动下载")

    # ================================================================================================
    def check_and_download_pretrained_files(self):
        # 预训练文件列表
        pretrained_files = [
            "checkpoint_best_legacy_500.pt",
            "hubert-soft-0d54a1f4.pt",
            "rmvpe.pt",
            "fcpe.pt",
            "DPHuBERT-sp0.75.pth"
        ]

        pretrain_dir = "marth7th_ai/pretrain"

        for file_name in pretrained_files:
            target_file_path = os.path.join(pretrain_dir, file_name)

            # 检查文件是否存在
            if not os.path.exists(target_file_path):
                # 根据文件名选择下载链接
                download_url = None
                if file_name == "checkpoint_best_legacy_500.pt":
                    download_url = "https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr"
                elif file_name == "rmvpe.pt" or file_name == "fcpe.pt":
                    download_url = f"https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/{file_name}"
                elif file_name == "hubert-soft-0d54a1f4.pt":
                    download_url = "https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt"
                elif file_name == "DPHuBERT-sp0.75.pth":
                    download_url = "https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPHuBERT-sp0.75.pth"

                # 下载文件
                if download_url:
                    try:
                        response = requests.get(download_url, stream=True)
                        response.raise_for_status()
                        with open(target_file_path, "wb") as file:
                            for chunk in response.iter_content(chunk_size=8192):
                                file.write(chunk)
                        print(f"{file_name} 文件已下载并保存在 {pretrain_dir} 目录下。")
                    except Exception as e:
                        print(f"下载 {file_name} 文件时发生错误：{str(e)}请根据readme手动下载")
                else:
                    print(f"无法找到 {file_name} 文件的下载链接。请根据readme手动下载")
    async def newbing_new_chat(self, event: MessageEvent, matcher: Matcher) -> None:
        """重置会话"""
        current_time: int = event.time
        user_id: str = str(event.user_id)
        if user_id in self.bing_chat_dict:
            last_time: int = self.bing_chat_dict[user_id]["last_time"]
            if (current_time - last_time < config.newbing_cd_time) and (
                event.get_user_id() not in config.superusers
            ):  # 如果当前时间减去上一次时间小于CD时间, 直接返回 # type: ignore
                await matcher.finish(
                    MessageSegment.reply(event.message_id)
                    + MessageSegment.text(
                        f"非报错情况下每个会话需要{config.newbing_cd_time}秒才能新建哦, 当前还需要{config.newbing_cd_time - (current_time - last_time)}秒"
                    )
                )
        bot: bingChatbot = await bingChatbot.create(
            cookies=random.choice(self.bing_cookies), proxy=self.proxy
        )  # 随机选择一个cookies创建一个Chatbot
        self.bing_chat_dict[user_id] = {
            "chatbot": bot,
            "last_time": current_time,
            "model": config.newbing_style,
            "sessions_number": 0,
            "isRunning": False,
        }

    @staticmethod
    async def bing_string_handle(input_string: str) -> str:
        """处理一下bing返回的字符串"""
        return re.sub(r'\[\^(\d+)\^]',  r'[\1]', input_string)

    # ================================================================================================

    # ================================================================================================
    async def openai_new_chat(self, event: MessageEvent, matcher: Matcher) -> None:
        """重置会话"""
        current_time: int = event.time  # 获取当前时间
        user_id: str = str(event.user_id)
        if user_id in self.openai_chat_dict:
            last_time: int = self.openai_chat_dict[user_id]["last_time"]
            if (current_time - last_time < config.openai_cd_time) and (
                event.get_user_id() not in config.superusers
            ):  # 如果当前时间减去上一次时间小于CD时间, 直接返回 # type: ignore
                await matcher.finish(
                    MessageSegment.reply(event.message_id)
                    + MessageSegment.text(
                        f"非报错情况下每个会话需要{config.openai_cd_time}秒才能新建哦, 当前还需要{config.openai_cd_time - (current_time - last_time)}秒"
                    )
                )
        bot = openaiChatbot(
            api_key=random.choice(self.openai_api_key),
            max_tokens=self.openai_max_tokens,
            proxy=self.proxy,
        )  # 随机选择一个api_key创建一个Chatbot
        self.openai_chat_dict[user_id] = {
            "chatbot": bot,
            "last_time": current_time,
            "sessions_number": 0,
            "isRunning": False,
        }

    # ================================================================================================

    # ================================================================================================
    async def rand_hello(self) -> str:
        """随机问候语"""
        return random.choice(self.hello_reply)

    async def rand_poke(self) -> str:
        """随机戳一戳"""
        return random.choice(self.poke__reply)

    async def get_chat_result(self, text: str, nickname: str) -> Union[str, None]:
        """从字典中返回结果"""
        if len(text) < 7:
            keys = self.anime_thesaurus.keys()
            for key in keys:
                if key in text:
                    return random.choice(self.anime_thesaurus[key]).replace(
                        "你", nickname
                    )

    async def add_word(self, word1: str, word2: str) -> Union[str, None]:
        """添加词条"""
        lis = []
        for key in self.anime_thesaurus:
            if key == word1:
                lis = self.anime_thesaurus[key]
                for word in lis:
                    if word == word2:
                        return "寄"
        if lis == []:
            axis: Dict[str, List[str]] = {word1: [word2]}
        else:
            lis.append(word2)
            axis = {word1: lis}
        self.anime_thesaurus.update(axis)
        with open(self.keyword_path, "w", encoding="utf-8") as f:
            json.dump(self.anime_thesaurus, f, ensure_ascii=False, indent=4)

    async def check_word(self, target: str) -> str:
        """查询关键词下词条"""
        for item in self.anime_thesaurus:
            if target == item:
                mes: str = f"下面是关键词 {target} 的全部响应\n\n"
                # 获取关键词
                lis = self.anime_thesaurus[item]
                for n, word in enumerate(lis, start=1):
                    mes = mes + str(n) + "、" + word + "\n"
                return mes
        return "寄"

    async def check_all(self) -> str:
        """查询全部关键词"""
        mes = "下面是全部关键词\n\n"
        for c in self.anime_thesaurus:
            mes: str = mes + c + "\n"
        return mes

    async def del_word(self, word1: str, word2: int) -> Union[str, None]:
        """删除关键词下具体回答"""
        axis = {}
        for key in self.anime_thesaurus:
            if key == word1:
                lis: list = self.anime_thesaurus[key]
                word2 = int(word2) - 1
                try:
                    lis.pop(word2)
                    axis = {word1: lis}
                except Exception:
                    return "寄"
        if axis == {}:
            return "寄"
        self.anime_thesaurus.update(axis)
        with open(self.keyword_path, "w", encoding="utf8") as f:
            json.dump(self.anime_thesaurus, f, ensure_ascii=False, indent=4)

    # ================================================================================================

    @staticmethod
    async def text_to_img(text: str) -> bytes:
        """将文字转换为图片"""
        return await txt_to_img.txt_to_img(text)


# 创建一个工具实例
utils = Utils()
