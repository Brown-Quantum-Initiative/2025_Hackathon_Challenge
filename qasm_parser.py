'''
parse tokens from .qasm file

'''

import re

#   token defs
class Token:
    typ: str     # 'identifier', 'number', 'string', 'operator', 'symbol', 'arrow'
    val: str

#   regex to remove comments // line comments and /* block comments */
RE_LINE_COMMENT = re.compile(r"//.*?$", re.MULTILINE)
RE_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

#   tokenizer regex
TOKEN_RE = re.compile(
    r"""
    (?P<ID>        [A-Za-z_][A-Za-z0-9_]*)
  | (?P<NUMBER>    (?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<STRING>    "([^"\\]|\\.)*")
  | (?P<ARROW>     ->)
  | (?P<OP>        ==|!=|<=|>=|\+=|-=|\*=|/=|&&|\|\||::)
  | (?P<SYMBOL>    [{}\[\]();,.:<>+\-*/%&|^~?=])
    """,
    re.VERBOSE,
)

def strip_comments(src: str) -> str:
    src = RE_BLOCK_COMMENT.sub("", src)
    src = RE_LINE_COMMENT.sub("", src)
    return src

def tokenize(src: str):
    #   get token objects with type, value, line and column
    
    i = 0
    n = len(src)
    while i < n:
        m = TOKEN_RE.match(src, i)
        if m:
            typ = m.lastgroup
            val = m.group(typ)
            yield Token(typ=typ, val=val)
            i = m.end()
        else:
            #   skip whitespaces
            if src[i].isspace():
                i += 1
                continue

