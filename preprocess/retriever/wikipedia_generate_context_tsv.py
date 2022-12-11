#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import argparse
from tqdm import tqdm
import sqlite3
import re
import spacy
from typing import List, Union, Optional, Mapping, Any
import numpy as np
from dataclasses import dataclass
from copy import deepcopy
from utils import make_wiki_id

'''After WikiExtractor and https://github.com/facebookresearch/DrQA/tree/main/scripts/retriever pre-processing is done on the XML dump, 
the final pre-processing is done in this script to generate the WIKI_6_3, WIKI_8_4, and WIKI_100w corpuses (without tables and lists) '''

class Text:
    """Class representing a text to be reranked.
    A text is unspecified with respect to it length; in principle, it
    could be a full-length document, a paragraph-sized passage, or
    even a short phrase.
    Parameters
    ----------
    text : str
        The text to be reranked.
    metadata : Mapping[str, Any]
        Additional metadata and other annotations.
    score : Optional[float]
        The score of the text. For example, the score might be the BM25 score
        from an initial retrieval stage.
    title : Optional[str]
        The text's title.
    """

    def __init__(self,
                 text: str,
                 metadata: Mapping[str, Any] = None,
                 score: Optional[float] = 0,
                 title: Optional[str] = None):
        self.text = text
        if metadata is None:
            metadata = dict()
        self.metadata = metadata
        self.score = score
        self.title = title


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='generate .tsv files for wiki corpuses of passages without tables and lists')
    parser.add_argument('--db_path', type=str, required=True, help='path to .db file containing preprocessed wiki pages from DrQA')
    parser.add_argument('--lang', type=str, required=True, help='')
    parser.add_argument('--output_path_100w', type=str, help='path to write .tsv with 100 word splits')
    args = parser.parse_args()

    sqliteConnection = sqlite3.connect(args.db_path)
    cursor = sqliteConnection.cursor()
    sqlite_select_query = "SELECT id, original_title, text FROM documents"
    cursor.execute(sqlite_select_query)
    pages = cursor.fetchall()
    cursor.close()

    print("PAGES: ", len(pages))

    documents = {}

    if args.lang == "en":
        nlp = spacy.load("en_core_web_lg")
    elif args.lang == "fr":
        nlp = spacy.load("fr_core_news_lg")

    # To avoid duplicate pages
    for row in tqdm(pages):
        text = row[2]
        title = row[1]
        
        if title in documents:
            documents[title] += " " + text
        else:
            documents[title] = text

    f3 = open(args.output_path_100w, "w")
    f3.write("id\ttext\ttitle\n")

    id3 = 1
    for document in tqdm(documents):
        texts = []
        text = documents[document]
        text = text.strip()

        if text.startswith("REDIRECT") or text.startswith("redirect"):
            continue
        if text.endswith(". References."):
            text = text[:-len(" References.")].strip()
            
        text = re.sub('\{\{cite .*?\}\}', ' ', text, flags=re.DOTALL) 
        text = text.replace(r"TABLETOREPLACE", " ")
        text = text.replace(r"'''", " ")
        text = text.replace(r"[[", " ")
        text = text.replace(r"]]", " ")
        text = text.replace(r"{{", " ")
        text = text.replace(r"}}", " ")
        text = text.replace("<br>", " ")
        text = text.replace("&quot;", "\"")
        text = text.replace("&amp;", "&")  
        text = text.replace("& amp;", "&")
        text = text.replace("nbsp;", " ")
        text = text.replace("formatnum:", "")
        
        #text = re.sub('<poem.*?</poem>', ' ', text, flags=re.DOTALL) # might have useful information?
        text = re.sub('<math.*?</math>', '', text, flags=re.DOTALL)
        text = re.sub('<chem.*?</chem>', '', text, flags=re.DOTALL) 
        text = re.sub('<score.*?</score>', '', text, flags=re.DOTALL) 

        # clean residual mess from xml dump that shouldn't have made its way here
        text = re.sub('\| ?item[0-9]?_?style= ?.*? ', ' ', text)
        text = re.sub('\| ?col[0-9]?_?style= ?.*? ', ' ', text)
        text = re.sub('\| ?row[0-9]?_?style= ?.*? ', ' ', text)    
        text = re.sub('\| ?style= ?.*? ', ' ', text)
        text = re.sub('\| ?bodystyle= ?.*? ', ' ', text)
        text = re.sub('\| ?frame_?style= ?.*? ', ' ', text)
        text = re.sub('\| ?data_?style= ?.*? ', ' ', text)
        text = re.sub('\| ?label_?style= ?.*? ', ' ', text)
        text = re.sub('\| ?headerstyle= ?.*? ', ' ', text)
        text = re.sub('\| ?list_?style= ?.*? ', ' ', text)
        text = re.sub('\| ?title_?style= ?.*? ', ' ', text)
        text = re.sub('\| ?ul_?style= ?.*? ', ' ', text)
        text = re.sub('\| ?li_?style= ?.*? ', ' ', text)
        text = re.sub('\| ?border-style= ?.*? ', ' ', text)
        text = re.sub('\|? ?style=\".*?\"', '', text)
        text = re.sub('\|? ?rowspan=\".*?\"', '', text)
        text = re.sub('\|? ?colspan=\".*?\"', '', text)
        text = re.sub('\|? ?scope=\".*?\"', '', text)
        text = re.sub('\|? ?align=\".*?\"', '', text)
        text = re.sub('\|? ?valign=\".*?\"', '', text)
        text = re.sub('\|? ?lang=\".*?\"', '', text)
        text = re.sub('\|? ?bgcolor=\".*?\"', '', text)
        text = re.sub('\|? ?bg=\#[a-z]+', '', text)
        text = re.sub('\|? ?width=\".*?\"', '', text)
        text = re.sub('\|? ?height=[0-9]+', '', text)
        text = re.sub('\|? ?width=[0-9]+', '', text)
        text = re.sub('\|? ?rowspan=[0-9]+', '', text)
        text = re.sub('\|? ?colspan=[0-9]+', '', text)
        text = re.sub(r'[\n\t]', ' ', text)
        text = re.sub('<.*?/>', '', text)
        text = re.sub('\|? ?align=[a-z]+', '', text)
        text = re.sub('\|? ?valign=[a-z]+', '', text)
        text = re.sub('\|? ?scope=[a-z]+', '', text)
        text = re.sub('&lt;ref&gt;.*?&lt;/ref&gt;', ' ', text)
        text = re.sub('&lt;.*?&gt;', ' ', text)
        text = re.sub('File:[A-Za-z0-9 ]+\.[a-z]{3,4}(\|[0-9]+px)?', '', text)
        text = re.sub('Source: \[.*?\]', '', text)
        text = re.sub('Source: \[.*?\]', '', text)
        text = re.sub(r'(\bSection::::\b)', "", text)
        text = text.replace("Country flag|", "country:")
        text = text.replace("flag|", "country:")
        text = text.replace("flagicon|", "country:")
        text = text.replace("flagcountry|", "country:")
        text = text.replace("Flagu|", "country:")
        text = text.replace("display=inline", "")
        text = text.replace("display=it", "")
        text = text.replace("abbr=on", "")
        text = text.replace("disp=table", "")
        text = re.sub(' +', ' ',text)
        texts.append(Text(text, metadata={"docid":make_wiki_id(document, 0)}))

        document = document.replace("\n", " ").replace("\t", " ")
        
        full_text = ""
        for text in texts:
            full_text += text.text + " "
        doc = nlp.make_doc(full_text)
        segments = []
        word_count = 0
        segment_tokens = []
        for token in doc:
            segment_tokens.append(token.text_with_ws)
            if not token.is_space and not token.is_punct:
                word_count+=1
                if word_count == 100:
                    word_count = 0
                    segments.append(''.join([token for token in segment_tokens]))
                    segment_tokens = []
        if word_count != 0:
            for token in doc:
                segment_tokens.append(token.text_with_ws)
                if not token.is_space and not token.is_punct:
                    word_count+=1
                    if word_count == 100:
                        word_count = 0
                        segments.append(''.join([token for token in segment_tokens]))
                        break
        if word_count != 0:
            segments.append(''.join([token for token in segment_tokens]))
        if len(segments) > 0:
            segments[0] = document + " " + segments[0]
        for segment in segments:
            text = segment.replace("\n", " ").replace("\t", " ")
            f3.write(str(id3) + '\t' + text + '\t' + document + '\n')
            id3+=1

    f3.close()