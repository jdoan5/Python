# Assume doing a HackerRank, breakdown how to solve, questions to ask, pattern to apply... Problem 1 — Parse Field Lines (15 pts, target ~12 min)
# You receive raw text lines from a document-extraction system. Each valid line contains comma-separated `key: value` pairs.
# Write `parse_lines(lines: list[str]) -> dict` returning:
# python
#
# ```python
# {"records": [dict, ...], "errors": [int, ...]}  # errors = 0-based indices of malformed lines
# ```
#
# Rules: a line is malformed if it's empty/whitespace-only, or if any segment lacks exactly one `:` separator with a non-empty key and non-empty value. Keys and values are stripped of surrounding whitespace. Keys are lowercased; values keep original case. Duplicate keys within a line: last one wins.
# python
#
# ```python
# lines = [
#     "rate: 10.5%, currency: USD, maturity: 2026-07-14",
#     "rate: 3.2%,, currency: EUR",
#     "   ",
#     "Currency : gbp ,  RATE :7.0%",
#     "noseparator",
# ]
# # expected:
# # records = [
# #   {"rate": "10.5%", "currency": "USD", "maturity": "2026-07-14"},
# #   {"currency": "gbp", "rate": "7.0%"},
# # ]
# # errors = [1, 2, 4]
# ```
#
# (Line 1 is malformed because the double comma creates an empty segment.)