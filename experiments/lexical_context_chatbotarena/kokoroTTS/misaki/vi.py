import re
from string import punctuation
from .en import G2P
from underthesea.pipeline.word_tokenize import tokenize, regex_tokenize
import re
from .num2vi import n2w, n2w_single
from collections import deque
from . import data
import importlib.resources
import json
from dataclasses import dataclass

# import prosodic as p
#  (C1)(w)V(G|C2)+T

# symbol " ' " for undefine symbol and sign for english

"""
C1 = initial consonant onset
w = labiovelar on-glide /w/
V = vowel nucleus
G = off-glide coda (/j/ or /w/)
C2 = final consonant coda
T = tone.
"""
Cus_onsets = {
    "b": "b",
    "t": "t",
    "th": "tʰ",
    "đ": "d",
    "ch": "c",
    "kh": "x",
    "g": "ɣ",
    "l": "l",
    "m": "m",
    "n": "n",
    "ngh": "ŋ",
    "nh": "ɲ",
    "ng": "ŋ",
    "ph": "f",
    "v": "v",
    "x": "s",
    "d": "z",
    "h": "h",
    "p": "p",
    "qu": "kw",
    "gi": "j",
    "tr": "ʈ",
    "k": "k",
    "c": "k",
    "gh": "ɣ",
    "r": "ʐ",
    "s": "ʂ",
    "gi": "j",
}

# Old or mixed alphabet
Cus_onsets.update({"f": "f", "j": "j", "w": "w", "z": "z"})

Cus_nuclei = {
    "a": "a",
    "á": "a",
    "à": "a",
    "ả": "a",
    "ã": "a",
    "ạ": "a",
    "â": "ɤ̆",
    "ấ": "ɤ̆",
    "ầ": "ɤ̆",
    "ẩ": "ɤ̆",
    "ẫ": "ɤ̆",
    "ậ": "ɤ̆",
    "ă": "ă",
    "ắ": "ă",
    "ằ": "ă",
    "ẳ": "ă",
    "ẵ": "ă",
    "ặ": "ă",
    "e": "ɛ",
    "é": "ɛ",
    "è": "ɛ",
    "ẻ": "ɛ",
    "ẽ": "ɛ",
    "ẹ": "ɛ",
    "ê": "e",
    "ế": "e",
    "ề": "e",
    "ể": "e",
    "ễ": "e",
    "ệ": "e",
    "i": "i",
    "í": "i",
    "ì": "i",
    "ỉ": "i",
    "ĩ": "i",
    "ị": "i",
    "o": "ɔ",
    "ó": "ɔ",
    "ò": "ɔ",
    "ỏ": "ɔ",
    "õ": "ɔ",
    "ọ": "ɔ",
    "ô": "o",
    "ố": "o",
    "ồ": "o",
    "ổ": "o",
    "ỗ": "o",
    "ộ": "o",
    "ơ": "ɤ",
    "ớ": "ɤ",
    "ờ": "ɤ",
    "ở": "ɤ",
    "ỡ": "ɤ",
    "ợ": "ɤ",
    "u": "u",
    "ú": "u",
    "ù": "u",
    "ủ": "u",
    "ũ": "u",
    "ụ": "u",
    "ư": "ɯ",
    "ứ": "ɯ",
    "ừ": "ɯ",
    "ử": "ɯ",
    "ữ": "ɯ",
    "ự": "ɯ",
    "y": "i",
    "ý": "i",
    "ỳ": "i",
    "ỷ": "i",
    "ỹ": "i",
    "ỵ": "i",
    "eo": "eo",
    "éo": "eo",
    "èo": "eo",
    "ẻo": "eo",
    "ẽo": "eo",
    "ẹo": "eo",
    "êu": "ɛu",
    "ếu": "ɛu",
    "ều": "ɛu",
    "ểu": "ɛu",
    "ễu": "ɛu",
    "ệu": "ɛu",
    "ia": "iə",
    "ía": "iə",
    "ìa": "iə",
    "ỉa": "iə",
    "ĩa": "iə",
    "ịa": "iə",
    "ia": "iə",
    "iá": "iə",
    "ià": "iə",
    "iả": "iə",
    "iã": "iə",
    "iạ": "iə",
    "iê": "iə",
    "iế": "iə",
    "iề": "iə",
    "iể": "iə",
    "iễ": "iə",
    "iệ": "iə",
    "oo": "ɔ",
    "óo": "ɔ",
    "òo": "ɔ",
    "ỏo": "ɔ",
    "õo": "ɔ",
    "ọo": "ɔ",
    "oo": "ɔ",
    "oó": "ɔ",
    "oò": "ɔ",
    "oỏ": "ɔ",
    "oõ": "ɔ",
    "oọ": "ɔ",
    "ôô": "o",
    "ốô": "o",
    "ồô": "o",
    "ổô": "o",
    "ỗô": "o",
    "ộô": "o",
    "ôô": "o",
    "ôố": "o",
    "ôồ": "o",
    "ôổ": "o",
    "ôỗ": "o",
    "ôộ": "o",
    "ua": "uə",
    "úa": "uə",
    "ùa": "uə",
    "ủa": "uə",
    "ũa": "uə",
    "ụa": "uə",
    "uô": "uə",
    "uố": "uə",
    "uồ": "uə",
    "uổ": "uə",
    "uỗ": "uə",
    "uộ": "uə",
    "ưa": "ɯə",
    "ứa": "ɯə",
    "ừa": "ɯə",
    "ửa": "ɯə",
    "ữa": "ɯə",
    "ựa": "ɯə",
    "ươ": "ɯə",
    "ướ": "ɯə",
    "ườ": "ɯə",
    "ưở": "ɯə",
    "ưỡ": "ɯə",
    "ượ": "ɯə",
    "yê": "iɛ",
    "yế": "iɛ",
    "yề": "iɛ",
    "yể": "iɛ",
    "yễ": "iɛ",
    "yệ": "iɛ",
    "uơ": "uə",
    "uở": "uə",
    "uờ": "uə",
    "uở": "uə",
    "uỡ": "uə",
    "uợ": "uə",
}


Cus_offglides = {
    "ai": "aj",
    "ái": "aj",
    "ài": "aj",
    "ải": "aj",
    "ãi": "aj",
    "ại": "aj",
    "ay": "ăj",
    "áy": "ăj",
    "ày": "ăj",
    "ảy": "ăj",
    "ãy": "ăj",
    "ạy": "ăj",
    "ao": "aw",
    "áo": "aw",
    "ào": "aw",
    "ảo": "aw",
    "ão": "aw",
    "ạo": "aw",
    "au": "ăw",
    "áu": "ăw",
    "àu": "ăw",
    "ảu": "ăw",
    "ãu": "ăw",
    "ạu": "ăw",
    "ây": "ɤ̆j",
    "ấy": "ɤ̆j",
    "ầy": "ɤ̆j",
    "ẩy": "ɤ̆j",
    "ẫy": "ɤ̆j",
    "ậy": "ɤ̆j",
    "âu": "ɤ̆w",
    "ấu": "ɤ̆w",
    "ầu": "ɤ̆w",
    "ẩu": "ɤ̆w",
    "ẫu": "ɤ̆w",
    "ậu": "ɤ̆w",
    "eo": "ew",
    "éo": "ew",
    "èo": "ew",
    "ẻo": "ew",
    "ẽo": "ew",
    "ẹo": "ew",
    "iu": "iw",
    "íu": "iw",
    "ìu": "iw",
    "ỉu": "iw",
    "ĩu": "iw",
    "ịu": "iw",
    "oi": "ɔj",
    "ói": "ɔj",
    "òi": "ɔj",
    "ỏi": "ɔj",
    "õi": "ɔj",
    "ọi": "ɔj",
    "ôi": "oj",
    "ối": "oj",
    "ồi": "oj",
    "ổi": "oj",
    "ỗi": "oj",
    "ội": "oj",
    "ui": "uj",
    "úi": "uj",
    "ùi": "uj",
    "ủi": "uj",
    "ũi": "uj",
    "ụi": "uj",
    # u'uy' : u'uj', u'úy' : u'uj', u'ùy' : u'uj', u'ủy' : u'uj', u'ũy' : u'uj', u'ụy' : u'uj',
    "uy": "ʷi",
    "úy": "uj",
    "ùy": "uj",
    "ủy": "uj",
    "ũy": "uj",
    "ụy": "uj",
    # prevent duplicated phonemes
    "uy": "ʷi",
    "uý": "ʷi",
    "uỳ": "ʷi",
    "uỷ": "ʷi",
    "uỹ": "ʷi",
    "uỵ": "ʷi",
    "ơi": "ɤj",
    "ới": "ɤj",
    "ời": "ɤj",
    "ởi": "ɤj",
    "ỡi": "ɤj",
    "ợi": "ɤj",
    "ưi": "ɯj",
    "ứi": "ɯj",
    "ừi": "ɯj",
    "ửi": "ɯj",
    "ữi": "ɯj",
    "ựi": "ɯj",
    "ưu": "ɯw",
    "ứu": "ɯw",
    "ừu": "ɯw",
    "ửu": "ɯw",
    "ữu": "ɯw",
    "ựu": "ɯw",
    "iêu": "iəw",
    "iếu": "iəw",
    "iều": "iəw",
    "iểu": "iəw",
    "iễu": "iəw",
    "iệu": "iəw",
    "yêu": "iəw",
    "yếu": "iəw",
    "yều": "iəw",
    "yểu": "iəw",
    "yễu": "iəw",
    "yệu": "iəw",
    "uôi": "uəj",
    "uối": "uəj",
    "uồi": "uəj",
    "uổi": "uəj",
    "uỗi": "uəj",
    "uội": "uəj",
    "ươi": "ɯəj",
    "ưới": "ɯəj",
    "ười": "ɯəj",
    "ưởi": "ɯəj",
    "ưỡi": "ɯəj",
    "ượi": "ɯəj",
    "ươu": "ɯəw",
    "ướu": "ɯəw",
    "ườu": "ɯəw",
    "ưởu": "ɯəw",
    "ưỡu": "ɯəw",
    "ượu": "ɯəw",
}
# The rounded vowels here are exactly not rounded: no w before => Try to add ʷ
Cus_onglides = {
    "oa": "ʷa",
    "oá": "ʷa",
    "oà": "ʷa",
    "oả": "ʷa",
    "oã": "ʷa",
    "oạ": "ʷa",
    "óa": "ʷa",
    "òa": "ʷa",
    "ỏa": "ʷa",
    "õa": "ʷa",
    "ọa": "ʷa",
    "oă": "ʷă",
    "oắ": "ʷă",
    "oằ": "ʷă",
    "oẳ": "ʷă",
    "oẵ": "ʷă",
    "oặ": "ʷă",
    "oe": "ʷɛ",
    "oé": "ʷɛ",
    "oè": "ʷɛ",
    "oẻ": "ʷɛ",
    "oẽ": "ʷɛ",
    "oẹ": "ʷɛ",
    "oe": "ʷɛ",
    "óe": "ʷɛ",
    "òe": "ʷɛ",
    "ỏe": "ʷɛ",
    "õe": "ʷɛ",
    "ọe": "ʷɛ",
    "ua": "ʷa",
    "uá": "ʷa",
    "uà": "ʷa",
    "uả": "ʷa",
    "uã": "ʷa",
    "uạ": "ʷa",
    "uă": "ʷă",
    "uắ": "ʷă",
    "uằ": "ʷă",
    "uẳ": "ʷă",
    "uẵ": "ʷă",
    "uặ": "ʷă",
    "uâ": "ʷɤ̆",
    "uấ": "ʷɤ̆",
    "uầ": "ʷɤ̆",
    "uẩ": "ʷɤ̆",
    "uẫ": "ʷɤ̆",
    "uậ": "ʷɤ̆",
    "ue": "ʷɛ",
    "ué": "ʷɛ",
    "uè": "ʷɛ",
    "uẻ": "ʷɛ",
    "uẽ": "ʷɛ",
    "uẹ": "ʷɛ",
    "uê": "ʷe",
    "uế": "ʷe",
    "uề": "ʷe",
    "uể": "ʷe",
    "uễ": "ʷe",
    "uệ": "ʷe",
    "uơ": "ʷɤ",
    "uớ": "ʷɤ",
    "uờ": "ʷɤ",
    "uở": "ʷɤ",
    "uỡ": "ʷɤ",
    "uợ": "ʷɤ",
    "uy": "ʷi",
    "uý": "ʷi",
    "uỳ": "ʷi",
    "uỷ": "ʷi",
    "uỹ": "ʷi",
    "uỵ": "ʷi",
    "uya": "ʷiə",
    "uyá": "ʷiə",
    "uyà": "ʷiə",
    "uyả": "ʷiə",
    "uyã": "ʷiə",
    "uyạ": "ʷiə",
    "uyê": "ʷiə",
    "uyế": "ʷiə",
    "uyề": "ʷiə",
    "uyể": "ʷiə",
    "uyễ": "ʷiə",
    "uyệ": "ʷiə",
    "uyu": "ʷiu",
    "uyú": "ʷiu",
    "uyù": "ʷiu",
    "uyủ": "ʷiu",
    "uyũ": "ʷiu",
    "uyụ": "ʷiu",
    "uyu": "ʷiu",
    "uýu": "ʷiu",
    "uỳu": "ʷiu",
    "uỷu": "ʷiu",
    "uỹu": "ʷiu",
    "uỵu": "ʷiu",
    "oen": "ʷen",
    "oén": "ʷen",
    "oèn": "ʷen",
    "oẻn": "ʷen",
    "oẽn": "ʷen",
    "oẹn": "ʷen",
    "oet": "ʷet",
    "oét": "ʷet",
    "oèt": "ʷet",
    "oẻt": "ʷet",
    "oẽt": "ʷet",
    "oẹt": "ʷet",
}

Cus_onoffglides = {
    "oe": "ɛj",
    "oé": "ɛj",
    "oè": "ɛj",
    "oẻ": "ɛj",
    "oẽ": "ɛj",
    "oẹ": "ɛj",
    "oai": "aj",
    "oái": "aj",
    "oài": "aj",
    "oải": "aj",
    "oãi": "aj",
    "oại": "aj",
    "oay": "ăj",
    "oáy": "ăj",
    "oày": "ăj",
    "oảy": "ăj",
    "oãy": "ăj",
    "oạy": "ăj",
    "oao": "aw",
    "oáo": "aw",
    "oào": "aw",
    "oảo": "aw",
    "oão": "aw",
    "oạo": "aw",
    "oeo": "ew",
    "oéo": "ew",
    "oèo": "ew",
    "oẻo": "ew",
    "oẽo": "ew",
    "oẹo": "ew",
    "oeo": "ew",
    "óeo": "ew",
    "òeo": "ew",
    "ỏeo": "ew",
    "õeo": "ew",
    "ọeo": "ew",
    "ueo": "ew",
    "uéo": "ew",
    "uèo": "ew",
    "uẻo": "ew",
    "uẽo": "ew",
    "uẹo": "ew",
    "uai": "aj",
    "uái": "aj",
    "uài": "aj",
    "uải": "aj",
    "uãi": "aj",
    "uại": "aj",
    "uay": "ăj",
    "uáy": "ăj",
    "uày": "ăj",
    "uảy": "ăj",
    "uãy": "ăj",
    "uạy": "ăj",
    "uây": "ɤ̆j",
    "uấy": "ɤ̆j",
    "uầy": "ɤ̆j",
    "uẩy": "ɤ̆j",
    "uẫy": "ɤ̆j",
    "uậy": "ɤ̆j",
}

Cus_codas = {
    "p": "p",
    "t": "t",
    "c": "k",
    "m": "m",
    "n": "n",
    "ng": "ŋ",
    "nh": "ɲ",
    "ch": "tʃ",
    "k": "k",
}

Cus_tones_p = {
    "á": 5,
    "à": 2,
    "ả": 4,
    "ã": 3,
    "ạ": 6,
    "ấ": 5,
    "ầ": 2,
    "ẩ": 4,
    "ẫ": 3,
    "ậ": 6,
    "ắ": 5,
    "ằ": 2,
    "ẳ": 4,
    "ẵ": 3,
    "ặ": 6,
    "é": 5,
    "è": 2,
    "ẻ": 4,
    "ẽ": 3,
    "ẹ": 6,
    "ế": 5,
    "ề": 2,
    "ể": 4,
    "ễ": 3,
    "ệ": 6,
    "í": 5,
    "ì": 2,
    "ỉ": 4,
    "ĩ": 3,
    "ị": 6,
    "ó": 5,
    "ò": 2,
    "ỏ": 4,
    "õ": 3,
    "ọ": 6,
    "ố": 5,
    "ồ": 2,
    "ổ": 4,
    "ỗ": 3,
    "ộ": 6,
    "ớ": 5,
    "ờ": 2,
    "ở": 4,
    "ỡ": 3,
    "ợ": 6,
    "ú": 5,
    "ù": 2,
    "ủ": 4,
    "ũ": 3,
    "ụ": 6,
    "ứ": 5,
    "ừ": 2,
    "ử": 4,
    "ữ": 3,
    "ự": 6,
    "ý": 5,
    "ỳ": 2,
    "ỷ": 4,
    "ỹ": 3,
    "ỵ": 6,
}

Cus_gi = {"gi": "zi", "gí": "zi", "gì": "zi", "gì": "zi", "gĩ": "zi", "gị": "zi"}

Cus_qu = {
    "quy": "kwi",
    "qúy": "kwi",
    "qùy": "kwi",
    "qủy": "kwi",
    "qũy": "kwi",
    "qụy": "kwi",
}

# letter pronunciation
EN = {
    "a": "ây",
    "b": "bi",
    "c": "si",
    "d": "đi",
    "e": "i",
    "f": "ép",
    "g": "giy",
    "h": "hếch",
    "i": "ai",
    "j": "giây",
    "k": "cây",
    "l": "eo",
    "m": "em",
    "n": "en",
    "o": "âu",
    "p": "pi",
    "q": "kiu",
    "r": "a",
    "s": "ét",
    "t": "ti",
    "u": "diu",
    "ư": "ư",
    "v": "vi",
    "w": "đắp liu",
    "x": "ít",
    "y": "quai",
    "z": "giét",
}
VI = {
    "a": "a",
    "ă": "á",
    "â": "ớ",
    "b": "bê",
    "c": "cê",
    "d": "dê",
    "đ": "đê",
    "e": "e",
    "ê": "ê",
    "f": "phờ",
    "g": "gờ",
    "h": "hờ",
    "i": "i",
    "j": "giây",
    "k": "ka",
    "l": "lờ",
    "m": "mờ",
    "n": "nờ",
    "o": "o",
    "ô": "ô",
    "ơ": "ơ",
    "p": "pờ",
    "q": "quy",
    "r": "rờ",
    "s": "sờ",
    "t": "tờ",
    "u": "u",
    "ư": "ư",
    "v": "vi",
    "w": "gờ",
    "x": "xờ",
    "y": "i",
    "z": "gia",
}
vi_syms = [
    "ɯəj",
    "ɤ̆j",
    "ʷiə",
    "ɤ̆w",
    "ɯəw",
    "ʷet",
    "iəw",
    "uəj",
    "ʷen",
    "tʰw",
    "ʷɤ̆",
    "ʷiu",
    "kwi",
    "ŋ͡m",
    "k͡p",
    "cw",
    "jw",
    "uə",
    "eə",
    "bw",
    "oj",
    "ʷi",
    "vw",
    "ăw",
    "ʈw",
    "ʂw",
    "aʊ",
    "fw",
    "ɛu",
    "tʰ",
    "tʃ",
    "ɔɪ",
    "xw",
    "ʷɤ",
    "ɤ̆",
    "ŋw",
    "ʊə",
    "zi",
    "ʷă",
    "dw",
    "eɪ",
    "aɪ",
    "ew",
    "iə",
    "ɣw",
    "zw",
    "ɯj",
    "ʷɛ",
    "ɯw",
    "ɤj",
    "ɔ:",
    "əʊ",
    "ʷa",
    "mw",
    "ɑ:",
    "hw",
    "ɔj",
    "uj",
    "lw",
    "ɪə",
    "ăj",
    "u:",
    "aw",
    "ɛj",
    "iw",
    "aj",
    "ɜ:",
    "kw",
    "nw",
    "t∫",
    "ɲw",
    "eo",
    "sw",
    "tw",
    "ʐw",
    "iɛ",
    "ʷe",
    "i:",
    "ɯə",
    "dʒ",
    "ɲ",
    "θ",
    "ʌ",
    "l",
    "w",
    "1",
    "ɪ",
    "ɯ",
    "d",
    "∫",
    "p",
    "ə",
    "u",
    "o",
    "3",
    "ɣ",
    "!",
    "ð",
    "ʧ",
    "6",
    "ʒ",
    "ʐ",
    "z",
    "v",
    "g",
    "ă",
    "æ",
    "ɤ",
    "2",
    "ʤ",
    "i",
    ".",
    "ɒ",
    "b",
    "h",
    "n",
    "ʂ",
    "ɔ",
    "ɛ",
    "k",
    "m",
    "5",
    " ",
    "c",
    "j",
    "x",
    "ʈ",
    ",",
    "4",
    "ʊ",
    "s",
    "ŋ",
    "a",
    "ʃ",
    "?",
    "r",
    ":",
    "η",
    "f",
    ";",
    "e",
    "t",
    "'",
]

with importlib.resources.open_text(data, "vi_symbols.json", encoding="utf-8") as r:
    SYMBOL_MAPPING = json.load(r)
with importlib.resources.open_text(data, "vi_teencode.json", encoding="utf-8") as r:
    TEENCODE_MAPPING = json.load(r)
NUMBER_REGEX = re.compile(regex_tokenize.number)
SYMBOL_REGEX = re.compile(
    "|".join(re.escape(symbol) for symbol in SYMBOL_MAPPING.keys())
)
EN_VI_REGEX = re.compile(
    "^[!-~“”–" + regex_tokenize.VIETNAMESE_CHARACTERS_LOWER + "]+$", re.IGNORECASE
)
VI_ONLY = re.compile(
    "|".join(
        re.escape(c)
        for c in regex_tokenize.VIETNAMESE_CHARACTERS_LOWER
        if not c.isascii()
    ),
    re.IGNORECASE,
)

################################################3


def trans(word, dialect, glottal, pham, cao, palatals):
    # Custom
    onsets, nuclei, codas, onglides, offglides, onoffglides, qu, gi = (
        Cus_onsets,
        Cus_nuclei,
        Cus_codas,
        Cus_onglides,
        Cus_offglides,
        Cus_onoffglides,
        Cus_qu,
        Cus_gi,
    )
    if pham or cao:
        # Custom
        tones_p = Cus_tones_p
        tones = tones_p

    ons = ""
    nuc = ""
    cod = ""
    ton = 0
    oOffset = 0
    cOffset = 0
    l = len(word)

    if l > 0:
        if word[0:3] in onsets:  # if onset is 'ngh'
            ons = onsets[word[0:3]]
            oOffset = 3
        elif word[0:2] in onsets:  # if onset is 'nh', 'gh', 'kʷ' etc
            ons = onsets[word[0:2]]
            oOffset = 2
        elif word[0] in onsets:  # if single onset
            ons = onsets[word[0]]
            oOffset = 1

        if word[l - 2 : l] in codas:  # if two-character coda
            cod = codas[word[l - 2 : l]]
            cOffset = 2
        elif word[l - 1] in codas:  # if one-character coda
            cod = codas[word[l - 1]]
            cOffset = 1

        # if word[0:2] == u'gi' and cod and len(word) == 3:  # if you just have 'gi' and a coda...
        if (
            word[0:2] in gi and cod and len(word) == 3
        ):  # if you just have 'gi' and a coda...
            nucl = "i"
            ons = "z"
        else:
            nucl = word[oOffset : l - cOffset]

        if nucl in nuclei:
            if oOffset == 0:
                if glottal == 1:
                    if word[0] not in onsets:  # if there isn't an onset....
                        ons = "ʔ" + nuclei[nucl]  # add a glottal stop
                    else:  # otherwise...
                        nuc = nuclei[nucl]  # there's your nucleus
                else:
                    nuc = nuclei[nucl]  # there's your nucleus
            else:  # otherwise...
                nuc = nuclei[nucl]  # there's your nucleus

        elif nucl in onglides and ons != "kw":  # if there is an onglide...
            nuc = onglides[nucl]  # modify the nuc accordingly
            if ons:  # if there is an onset...
                ons = ons + "w"  # labialize it, but...
            else:  # if there is no onset...
                ons = "w"  # add a labiovelar onset

        elif nucl in onglides and ons == "kw":
            nuc = onglides[nucl]

        elif nucl in onoffglides:
            cod = onoffglides[nucl][-1]
            nuc = onoffglides[nucl][0:-1]
            if ons != "kw":
                if ons:
                    ons = ons + "w"
                else:
                    ons = "w"
        elif nucl in offglides:
            cod = offglides[nucl][-1]
            nuc = offglides[nucl][:-1]

        elif word in gi:  # if word == 'gi', 'gì',...
            ons = gi[word][0]
            nuc = gi[word][1]

        elif word in qu:  # if word == 'quy', 'qúy',...
            ons = qu[word][:-1]
            nuc = qu[word][-1]

        else:
            # Something is non-Viet
            return (None, None, None, None)

        # Velar Fronting (Northern dialect)
        if dialect == "n":
            if nuc == "a":
                if cod == "k" and cOffset == 2:
                    nuc = "ɛ"
                if cod == "ɲ" and nuc == "a":
                    nuc = "ɛ"

            # Final palatals (Northern dialect)
            if nuc not in ["i", "e", "ɛ"]:
                if cod == "ɲ":
                    cod = "ɲ"  # u'ŋ'
            elif palatals != 1 and nuc in ["i", "e", "ɛ"]:
                if cod == "ɲ":
                    cod = "ɲ"  # u'ŋ'
            if palatals == 1:
                if cod == "k" and nuc in ["i", "e", "ɛ"]:
                    cod = "c"

        # Velar Fronting (Southern and Central dialects)
        else:
            if nuc in ["i", "e"]:
                if cod == "k":
                    cod = "t"
                if cod == "ŋ":
                    cod = "n"

            # There is also this reverse fronting, see Thompson 1965:94 ff.
            elif nuc in ["iə", "ɯə", "uə", "u", "ɯ", "ɤ", "o", "ɔ", "ă", "ɤ̆"]:
                if cod == "t":
                    cod = "k"
                if cod == "n":
                    cod = "ŋ"

        # Monophthongization (Southern dialects: Thompson 1965: 86; Hoàng 1985: 181)
        if dialect == "s":
            if cod in ["m", "p"]:
                if nuc == "iə":
                    nuc = "i"
                if nuc == "uə":
                    nuc = "u"
                if nuc == "ɯə":
                    nuc = "ɯ"

        # Tones
        # Modified 20 Sep 2008 to fix aberrant 33 error
        tonelist = [tones[word[i]] for i in range(0, l) if word[i] in tones]
        if tonelist:
            ton = str(tonelist[len(tonelist) - 1])
        else:
            if not (pham or cao):
                if dialect == "c":
                    ton = str("35")
                else:
                    ton = str("33")
            else:
                ton = str("1")

        # Modifications for closed syllables
        if cOffset != 0:
            # Obstruent-final nang tones are modal voice
            if (
                (dialect == "n" or dialect == "s")
                and ton == "21g"
                and cod in ["p", "t", "k"]
            ):
                # if ton == u'21\u02C0' and cod in ['p', 't', 'k']: # fixed 8 Nov 2016
                ton = "21"

            # Modification for sắc in closed syllables (Northern and Central only)
            if (
                (dialect == "n" and ton == "24") or (dialect == "c" and ton == "13")
            ) and cod in ["p", "t", "k"]:
                ton = "45"

            # Modification for 8-tone system
            if cao == 1:
                if ton == "5" and cod in ["p", "t", "k"]:
                    ton = "5b"
                if ton == "6" and cod in ["p", "t", "k"]:
                    ton = "6b"

            # labialized allophony (added 17.09.08)
            if nuc in ["u", "o", "ɔ"]:
                if cod == "ŋ":
                    cod = "ŋ͡m"
                if cod == "k":
                    cod = "k͡p"

        return (ons, nuc, cod, ton)


def convert(word, dialect, glottal, pham, cao, palatals, delimit):
    """Convert a single orthographic string to IPA."""

    ons = ""
    nuc = ""
    cod = ""
    ton = 0
    seq = ""

    try:
        (ons, nuc, cod, ton) = trans(word, dialect, glottal, pham, cao, palatals)
        if None in (ons, nuc, cod, ton):
            seq = "[" + word + "]"
        else:
            seq = delimit + delimit.join(filter(None, (ons, nuc, cod, ton))) + delimit
    except TypeError:
        pass

    return seq


########################333


def Parsing(listParse, text, delimit):
    undefine_symbol = "'"
    if listParse == "default":
        listParse = vi_syms.copy()
    listParse.sort(reverse=True, key=len)
    output = ""
    skip = 0
    for ic, char in enumerate(text):
        ##print(char,skip)
        check = 0
        if skip > 0:
            skip = skip - 1
            continue
        for l in listParse:
            if len(l) <= len(text[ic:]) and l == text[ic : ic + len(l)]:
                output += delimit + l
                check = 1
                skip = len(l) - 1
                break
        if check == 0:
            # Case symbol not in list
            if str(char) in ["ˈ", "ˌ", "*"]:
                continue
            # print("this is not in symbol :"+ char + ":")
            output += delimit + undefine_symbol
    return output.rstrip() + delimit


# Pseudo implementation of vinorm
def TTSnorm(text, use_linking_words=True):
    def number_fn(match):
        number = "".join([c for c in match.group("number") if c.isdigit()])
        if len(number) == 0:
            return ""
        if len(number) > 1:
            number = number.removeprefix("0")
        if use_linking_words:
            words = n2w(number)
        else:
            words = n2w_single(number)
        return words

    text = NUMBER_REGEX.sub(number_fn, text)
    return text


@dataclass
class ViToken:
    text: str
    phonemes: str = None
    parent: str = None


class VIG2P:
    def __init__(
        self,
        glottal=0,
        pham=0,
        cao=0,
        palatals=0,
        substr_tokenize=True,
        dialect="north",
        tone_type=0,
        num2words_use_linking_words=True,
        enable_en_g2p=True,
        en_g2p_kwargs={},
    ):
        self.glottal = glottal
        self.pham = pham
        self.cao = cao
        self.palatals = palatals
        self.substr_tokenize = substr_tokenize
        self.delimit = ""
        self.tone_type = tone_type
        self.num2words_use_linking_words = num2words_use_linking_words
        if dialect in ["north", "central", "south"]:
            self.dialect = dialect[0]
        else:
            raise NotImplementedError(f"Vietnamese dialect {dialect}")
        if self.tone_type == 0:
            self.pham = 1
        else:
            self.cao = 1
        en_g2p_kwargs["unk"] = "❓"
        self.en_g2p = G2P(**en_g2p_kwargs) if enable_en_g2p else lambda _: ("❓", [])

    def substr2ipa(self, tk, ipa):
        """
        Approximation of foreign name pronunciation
        Return (parent, text, phonemes)
        Example:
            Y:  /i/

            Blôk:
            - k -> /k/
            - ôk -> /ok͡p1/
            - lôk -> /lok͡p1/
            - Blôk -> ❌
            - B -> Bờ -> bɤ2
            => /bɤ2 lok͡p1/

            Êban:
            - n -> /nɤ2/
            - an -> /an1/
            - ban -> /ban1/
            - Êban -> ❌
            - Ê -> /e1/
            => /e1 ban1/

        => /i bɤ2 lok͡p1 e1 ban1/
        """
        if "[" not in ipa:
            return [(None, tk, ipa)]

        if tk.lower().upper() == tk:
            # Handle acronym by letter-by-letter
            mapping = VI if VI_ONLY.search(tk) is not None else EN
            return [
                (
                    tk,
                    char,
                    convert(
                        mapping.get(char, char),
                        self.dialect,
                        self.glottal,
                        self.pham,
                        self.cao,
                        self.palatals,
                        "",
                    ),
                )
                for char in tk.lower()
            ]

        orig_tk = tk
        tk = tk.lower()
        if VI_ONLY.search(tk) is None:
            eng, _ = self.en_g2p(tk)
            if "❓" not in eng:
                return [(None, tk, eng)]

        if not self.substr_tokenize:
            return [(None, tk, ipa)]

        parents = deque()
        parts = deque()
        sub_ipa = deque()
        while tk:
            if len(tk) == 1:
                char = tk
                _ipa = convert(
                    VI.get(char, char),
                    self.dialect,
                    self.glottal,
                    self.pham,
                    self.cao,
                    self.palatals,
                    "",
                )
                parents.appendleft(orig_tk)
                parts.appendleft(char)
                sub_ipa.appendleft(_ipa)
                break

            start, converted_ipa = -1, ""
            for i in range(len(tk) - 1, -1, -1):
                tkc = tk[i:]
                sub_tk = tkc if len(tkc) > 1 else VI.get(tkc, tkc)
                _ipa = convert(
                    sub_tk,
                    self.dialect,
                    self.glottal,
                    self.pham,
                    self.cao,
                    self.palatals,
                    "",
                )
                if "[" not in _ipa:
                    start = i
                    converted_ipa = _ipa

            if start != -1:
                parents.appendleft(orig_tk)
                sub_ipa.appendleft(converted_ipa)
                parts.appendleft(tk[start:])
                tk = tk[:start]
            else:
                break

        return list(zip(parents, parts, sub_ipa))

    def __call__(self, text):
        if self.substr_tokenize:
            text = text.replace("_", " ").replace("-", " ")
        TN = TTSnorm(text, self.num2words_use_linking_words)
        # Words in Vietnamese only have one morphological form, regardless of the compound words
        # so word segmentation is unnecessary
        # E.g. "nhà" trong "nhà xe" and "nhà lầu" are the same /ɲa2/
        TK = tokenize(TN)
        new_TK = []
        for word in TK:
            new_TK.extend(word.split())
        TK = new_TK
        IPA = ""
        vitokens: list[ViToken] = []
        for tk in TK:
            if EN_VI_REGEX.match(tk) is None:
                IPA += "[" + tk + "]" + " "
                vitokens.append(ViToken(tk, "[" + tk + "]"))
                continue
            if tk in [".", ",", ";", ":", "!", "?", ")", "}", "]"]:
                if tk in [")", "}", "]"]:
                    tk = ")"
                IPA = IPA.rstrip() + tk + " "
                vitokens.append(ViToken(tk))
                continue
            if tk in ["(", "{", "["]:
                tk = "("
                IPA += tk
                vitokens.append(ViToken(tk))
                continue
            if tk in ['"', "'", "–", "“", "”"]:
                IPA += tk + " "
                vitokens.append(ViToken(tk))
                continue
            tk = SYMBOL_REGEX.sub(
                lambda match: " " + SYMBOL_MAPPING[match[0]] + " ", tk
            )
            tk = TEENCODE_MAPPING.get(tk.lower(), tk)
            parent_tk_ipas = self.substr2ipa(
                tk,
                convert(
                    tk.lower(),
                    self.dialect,
                    self.glottal,
                    self.pham,
                    self.cao,
                    self.palatals,
                    self.delimit,
                ),
            )
            for parent, tk, ipa in parent_tk_ipas:
                IPA += ipa.strip() + " "
                vitokens.append(ViToken(tk, ipa, parent))
        return IPA.strip(), vitokens


__all__ = [VIG2P, vi_syms]
