
'''
Tokenize a .qasm
'''

import re
import argparse
import sys
import json

#   remove comments regex
RE_LINE_COMMENT  = re.compile(r"//.*?$", re.MULTILINE)
RE_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

# tokenize regex
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

#   type names
TYPE_MAP = {
    "ID": "identifier",
    "NUMBER": "number",
    "STRING": "string",
    "ARROW": "arrow",
    "OP": "operator",
    "SYMBOL": "symbol",
}

#   types cooper needs
KEYWORDS = {
    #   stuff we don't really need
    "OPENQASM", "qreg", "creg", "gate", "opaque", "barrier",
    "measure", "reset", "if", "include", "U", "CX",
    #   stuff we really need
    "qubit", "bit", "uint", "int", "let", "const", "def"
}

def strip_comments(src):
    src = RE_BLOCK_COMMENT.sub("", src)
    src = RE_LINE_COMMENT.sub("", src)
    return src

def tokenize(src):
    #   return dict tokens: {'typ','val'}
    i = 0
    n = len(src)
    while i < n:
        m = TOKEN_RE.match(src, i)
        if m:
            raw_typ = m.lastgroup
            val = m.group(raw_typ)
            typ = TYPE_MAP.get(raw_typ, raw_typ.lower())

            #   cooper keyword tagging
            if typ == "identifier" and val in KEYWORDS:
                typ = "keyword"

            yield {"typ": typ, "val": val}
            i = m.end()
            continue

        ch = src[i]
        if ch.isspace():
            i += 1
            continue

        snippet = src[i:i+20].replace("\n", "\\n")
        raise SyntaxError("Unexpected character {!r} at {}:{} near '{}'".format(ch, snippet))

def _next_id(tokens, idx):
    # return identifier at idx+1 if present
    if idx + 1 < len(tokens) and tokens[idx + 1]["typ"] == "identifier":
        return tokens[idx + 1]["val"]
    return None

def collect_declared_identifiers(tokens):
    """
    Returns a dict of type -> [names] for common QASM decls.
    Recognizes:
      - qreg ID [ NUMBER ] ;
      - creg ID [ NUMBER ] ;
      - qubit ID [ ... ]? ;
      - bit ID [ ... ]? ;
      - gate ID ... { ... }   (just collects the gate name)
      - opaque ID ... ;
    """
    out = {"qreg": [], "creg": [], "qubit": [], "bit": [], "gate": []}

    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t["typ"] == "keyword":
            kw = t["val"]
            if kw in ("qreg", "creg", "qubit", "bit"):
                name = _next_id(tokens, i)
                if name:
                    out[kw].append(name)
            elif kw in ("gate", "opaque"):
                name = _next_id(tokens, i)
                if name:
                    out["gate"].append(name)
        i += 1

    # de-dup while keeping order
    for k, vals in out.items():
        seen = set()
        dedup = []
        for v in vals:
            if v not in seen:
                seen.add(v)
                dedup.append(v)
        out[k] = dedup
    return out

def main(argv=None):

    '''
    Usage:
    python qasm_parser.py "qft.qasm" --idents-of qubit

    should return something like:
    [
        { "typ": "qreg", "val": "q" },
        { "typ": "qubit", "val": "q0" }
    ]

    '''

    parser = argparse.ArgumentParser(description="Tokenize a .qasm file and produce JSON.")
    parser.add_argument("path", nargs="?", default="-",
                        help="Path to .qasm file, or '-' to read from stdin")
    parser.add_argument("--ndjson", action="store_true",
                        help="Output newline-delimited JSON (one token per line).")
    parser.add_argument("--include", nargs="*", choices=sorted(set(TYPE_MAP.values())),
                        help="Only include these token types (identifier number string).")
    parser.add_argument("--idents-of", nargs="*", choices=["qreg", "creg", "qubit", "bit", "gate"],
                        help="Only emit declared identifiers of these types ( qubit bit qreg).")
    args = parser.parse_args(argv)

    # read input
    if args.path == "-":
        src = sys.stdin.read()
    else:
        with open(args.path, "r", encoding="utf-8") as f:
            src = f.read()

    cleaned = strip_comments(src)
    try:
        tokens = list(tokenize(cleaned))
    except SyntaxError as e:
        print("error: {}".format(e), file=sys.stderr)
        return 1

    #   which identifiers, do that and exit
    if args.idents_of:
        table = collect_declared_identifiers(tokens)
        # Flatten to list of {typ, val} so output shape matches coopers request
        filtered = []
        for kind in args.idents_of:
            for name in table.get(kind, []):
                filtered.append({"typ": kind, "val": name})
        if args.ndjson:
            for t in filtered:
                print(json.dumps(t, ensure_ascii=False))
        else:
            print(json.dumps(filtered, ensure_ascii=False, indent=2))
        return 0

    #   otherwise   filter by token types)
    if args.include:
        tokens = [t for t in tokens if t["typ"] in set(args.include)]

    if args.ndjson:
        for t in tokens:
            print(json.dumps(t, ensure_ascii=False))
    else:
        print(json.dumps(tokens, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
