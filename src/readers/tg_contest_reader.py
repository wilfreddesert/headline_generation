from typing import Dict
import pandas as pd
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter

from readers.summarization_reader import SummarizationReader


def parse_tg_jsonl(path):
    with open(path, "r", encoding="utf-8") as r:
        data = pd.read_json(path, lines=True)
        for i in range(len(data)):
            text = (
                data.iloc[i]["text"]
                .lower()
                .replace("\xa0", " ")
                .replace("\n", " ")
                .strip()
            )
            title = data.iloc[i]["title"].lower()

            if not text or not title or text.count(" ") < 3 or title.count(" ") < 3:
                continue
            yield text, title
        print(text, title)


@DatasetReader.register("tg")
class TGReader(SummarizationReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        source_max_tokens: int = 400,
        target_max_tokens: int = 100,
        separate_namespaces: bool = False,
        target_namespace: str = "target_tokens",
        save_copy_fields: bool = False,
        save_pgn_fields: bool = False,
    ) -> None:
        if not tokenizer:
            tokenizer = WordTokenizer(word_splitter=SimpleWordSplitter())
        super().__init__(
            tokenizer=tokenizer,
            source_token_indexers=source_token_indexers,
            target_token_indexers=target_token_indexers,
            source_max_tokens=source_max_tokens,
            target_max_tokens=target_max_tokens,
            separate_namespaces=separate_namespaces,
            target_namespace=target_namespace,
            save_copy_fields=save_copy_fields,
            save_pgn_fields=save_pgn_fields,
        )

    def parse_set(self, path):
        return parse_tg_jsonl(path)
