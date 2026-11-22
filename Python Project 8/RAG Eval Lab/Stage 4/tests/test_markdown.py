from __future__ import annotations

from raglab.markdown import clean_markdown, slugify


def files(**contents: str):
    """An include resolver backed by a dict, keyed by the path after docs_src/."""
    def resolve(ref: str) -> str | None:
        return contents.get(ref.split("docs_src/", 1)[-1])
    return resolve


# ---------------------------------------------------------------- headings

def test_hash_inside_a_code_fence_is_a_comment_not_a_heading():
    r = clean_markdown("# Title\n\n```python\n# create the app\napp = FastAPI()\n```\n\nAfter.\n")
    assert [h.title for h in r.headings] == ["Title"]
    assert "# create the app" in r.text


def test_heading_anchor_is_recorded_and_removed_from_the_text():
    r = clean_markdown("# First Steps { #first-steps }\n\nBody.\n")
    assert r.headings[0].anchor == "first-steps"
    assert r.text.startswith("# First Steps\n")
    assert "{" not in r.text


def test_unanchored_headings_get_mkdocs_slugs_and_duplicates_are_suffixed():
    r = clean_markdown("# Page\n\n## Set Up!\n\nx\n\n## Set Up!\n\ny\n")
    assert [h.anchor for h in r.headings] == ["page", "set-up", "set-up_1"]


def test_every_heading_offset_points_at_its_own_heading_line():
    r = clean_markdown("Intro text.\n\n# A { #a }\n\nbody\n\n### B\n\nmore\n\n## C { #c }\n")
    for h in r.headings:
        assert r.text[h.offset:].startswith("#" * h.level + " " + h.title)


def test_heading_titles_lose_inline_markup_but_keep_identifiers():
    r = clean_markdown("# The `response_model` **parameter**\n")
    assert r.headings[0].title == "The response_model parameter"


def test_slugify_matches_python_markdown_defaults():
    assert slugify("Path Parameters & Numeric Validations") == "path-parameters-numeric-validations"
    assert slugify("  Café  ") == "cafe"


# ---------------------------------------------------------------- includes

def test_star_include_is_inlined_as_a_fenced_block_with_its_language():
    r = clean_markdown("{* ../../docs_src/a/app.py hl[2] *}\n", files(**{"a/app.py": "x = 1\ny = 2\n"}))
    assert r.text == "```python\nx = 1\ny = 2\n```"
    assert r.unresolved_includes == ()


def test_line_range_include_keeps_exactly_the_requested_lines():
    src = "\n".join(f"line{i}" for i in range(1, 6))
    r = clean_markdown("{* ../../docs_src/a.py ln[2:3] hl[3] *}\n", files(**{"a.py": src}))
    assert r.text == "```python\nline2\nline3\n```"


def test_multiple_line_ranges_are_joined_with_an_ellipsis():
    src = "\n".join(f"line{i}" for i in range(1, 6))
    r = clean_markdown("{* ../../docs_src/a.py ln[1:1,4:5] *}\n", files(**{"a.py": src}))
    assert r.text == "```python\nline1\n...\nline4\nline5\n```"


def test_old_bang_include_inside_a_fence_is_inlined_raw():
    md = "```Python\n{!> ../../docs_src/t.py!}\n```\n"
    r = clean_markdown(md, files(**{"t.py": "print('hi')"}))
    assert r.text == "```python\nprint('hi')\n```"


def test_an_unresolvable_include_is_reported_not_fatal():
    r = clean_markdown("Before.\n\n{* ../../fastapi/openapi/docs.py ln[1:5] *}\n\nAfter.\n", files())
    assert r.unresolved_includes == ("../../fastapi/openapi/docs.py",)
    assert r.text == "Before.\n\nAfter."


# ---------------------------------------------------------------- code fences

def test_console_fences_drop_styling_html_but_keep_the_command():
    md = '<div class="termy">\n\n```console\n$ <font color="#4E9A06">fastapi</font> dev &amp; run\n```\n\n</div>\n'
    r = clean_markdown(md)
    assert "$ fastapi dev & run" in r.text
    assert "<font" not in r.text and "termy" not in r.text


def test_html_inside_a_python_example_is_code_and_survives():
    # Real case: the WebSockets page embeds an HTML page in a Python string.
    md = '```python\nhtml = """<h2>Your ID: <span id="ws-id"></span></h2>"""\n```\n'
    assert '<span id="ws-id"></span>' in clean_markdown(md).text


def test_dict_unpacking_in_code_survives():
    # Real case: `{**old_dict}` looks like an include directive to a naive filter.
    md = '```python\nnew = {**old, "k": 1}\n```\n'
    assert '{**old, "k": 1}' in clean_markdown(md).text


def test_fence_info_strings_are_reduced_to_the_language():
    r = clean_markdown('```Python hl_lines="3"\nx\n```\n\n```hl_lines="4"\ny\n```\n')
    assert r.text == "```python\nx\n```\n\n```\ny\n```"


# ---------------------------------------------------------------- prose cleanup

def test_admonitions_and_tab_blocks_become_labels_and_their_closers_vanish():
    md = "/// note | Technical Details\n\nInside.\n\n///\n\n//// tab | Python\n\nTabbed.\n\n////\n"
    r = clean_markdown(md)
    assert r.text == "**Note: Technical Details**\n\nInside.\n\n**Tab: Python**\n\nTabbed."


def test_abbr_is_expanded_and_links_and_images_reduce_to_their_text():
    md = 'An <abbr title="Large Language Model">LLM</abbr> reads [the guide](https://x.dev/g) ![Swagger UI](img.png).\n'
    assert clean_markdown(md).text == "An LLM (Large Language Model) reads the guide Swagger UI."


def test_html_inside_inline_code_is_prose_and_survives():
    assert clean_markdown("Wrap fields in an HTML `<form>` element.\n").text == "Wrap fields in an HTML `<form>` element."


def test_jinja_is_removed_from_prose_but_kept_in_code():
    md = "{% for s in sponsors %}\n<a href=\"{{ s.url }}\">x</a>\n{% endfor %}\n\n```jinja\n{{ url_for('static') }}\n```\n"
    r = clean_markdown(md)
    assert "{%" not in r.text
    assert r.text.count("{{") == 1 and "url_for" in r.text


def test_mkdocstrings_directive_and_its_options_are_removed():
    md = "# `FastAPI` class\n\nImport it.\n\n::: fastapi.FastAPI\n    options:\n        members:\n            - get\n\nTrailing prose.\n"
    r = clean_markdown(md)
    assert ":::" not in r.text and "options" not in r.text and "members" not in r.text
    assert r.text.endswith("Trailing prose.")


def test_html_comments_are_removed_single_and_multi_line():
    md = "A <!-- hidden --> B\n<!-- start\nstill hidden\nend -->C\n"
    assert clean_markdown(md).text == "A B\nC"


def test_front_matter_is_removed():
    assert clean_markdown("---\ninclude_yaml:\n  people: x.yml\n---\n\n# People\n").text == "# People"


def test_blank_line_runs_collapse_and_text_is_trimmed():
    assert clean_markdown("\n\n\nA\n\n\n\nB   \n\n\n").text == "A\n\nB"


# ---------------------------------------------------------------- regressions from the Stage 1 review

def test_a_link_whose_text_is_inline_code_is_reduced_to_that_code():
    # 81 such links in the real corpus kept their URL in the indexed text.
    assert clean_markdown("Use [`Depends`](https://x.dev/deps) here.\n").text == "Use `Depends` here."


def test_link_syntax_inside_a_code_span_is_left_alone():
    assert clean_markdown("Write `[text](url)` for a link.\n").text == "Write `[text](url)` for a link."


def test_a_comment_marker_inside_a_code_span_is_text_not_a_comment():
    r = clean_markdown("Type `<!--` here.\n\n# Next\n\nBody `-->` end.\n")
    assert [h.title for h in r.headings] == ["Next"]
    assert "Type `<!--` here." in r.text


def test_a_backtick_line_with_backticks_in_its_info_string_is_not_a_fence():
    # CommonMark: such a line is inline code. Treating it as a fence would flip
    # fence parity for the rest of the file and turn comments into headings.
    md = "```js``` is inline code.\n\n```\n# not a heading\n```\n\n# Real heading\n\nBody.\n"
    assert [h.title for h in clean_markdown(md).headings] == ["Real heading"]


def test_table_cells_stay_separated():
    md = "<td>name</td><td>str</td><td><strong>REQUIRED</strong></td>\n"
    assert clean_markdown(md).text == "name str REQUIRED"


def test_style_and_script_blocks_are_dropped_entirely():
    md = "# FastAPI\n\n<style>\n.md-content h1 { display: none; }\n</style>\n\nIntro.\n<script>track()</script>\n"
    assert clean_markdown(md).text == "# FastAPI\n\nIntro."


def test_only_terminated_entities_are_decoded():
    assert clean_markdown("Query ?a=1&notify=1&copy=2 &amp; done.\n").text == "Query ?a=1&notify=1&copy=2 & done."


def test_autolinks_keep_their_url():
    assert clean_markdown("Docs at <https://fastapi.tiangolo.com>.\n").text == "Docs at https://fastapi.tiangolo.com."


def test_author_anchors_are_never_renamed_and_generated_ones_never_collide():
    # An author's { #intro } wins even though an earlier heading slugs to "intro".
    r = clean_markdown("## Intro\n\nx\n\n## Other { #intro }\n\ny\n")
    assert [h.anchor for h in r.headings] == ["intro_1", "intro"]
    # Suffixing never produces a duplicate, even against a literal "A_1" heading.
    r = clean_markdown("## A\n\nx\n\n## A\n\ny\n\n## A_1\n\nz\n")
    anchors = [h.anchor for h in r.headings]
    assert len(set(anchors)) == 3 and anchors[:2] == ["a", "a_1"]


def test_a_leading_horizontal_rule_is_not_front_matter():
    md = "---\n\nIntro paragraph.\n\n---\n\n# Title\n\nBody\n"
    assert clean_markdown(md).text.startswith("---\n\nIntro paragraph.")
