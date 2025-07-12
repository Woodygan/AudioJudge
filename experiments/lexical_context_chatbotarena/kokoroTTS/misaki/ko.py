from g2pk2 import G2p
from .hangul.worker import convert
import re

class KOG2P:
    def __init__(self):
        self.g2p = G2p()

    def __call__(self, text):
        text = self.g2p(text)
        ps = ''
        for a, b in re.findall(r'([\uAC00-\uD7A3]+)|([^\uAC00-\uD7A3]+)', text):
            assert (not a) != (not b)
            if a:
                ps += convert(a)
            else:
                ps += b
                if not b.endswith(' ') and not b.endswith(chr(8220)):
                    ps += ' '
        ps = re.sub(r'(.)\*', r'\1\1', ps.strip()).replace('dʑ', 'ʥ').replace('tɕ', 'ʨ')
        return ps
