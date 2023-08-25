import re
import json
from functools import partial
import os
import pickle
from datetime import datetime
import frontmatter
import re
import pandas as pd
from slugify import slugify


HEADERS_REGEX = re.compile("#{1,5}")
NUMBERED_HEADER_REGEX = re.compile("")
END_ESCAPE = "< END >"
QUESTION_HEADER = "## Question"


def is_useful_line(line):
    """
    Utility function to decide if this is some autogenerated or unhelpful line from filetypes pages, which can be generated in a variety of ways.
    """
    # if line == "\n": # New lines are helpful in the sections to determine paragraph chunks. Might want to make this a generalized filter instead of hand coded heuristics.
    #     return False
    if line == "---\n":
        return False
    if line == "\n---":
        return False
    if "<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT!" in line:
        return False
    return True


def find_question(lines):
    """
    Find the line that contains a question in an Outerbounds how-to guide.
    Return it with its index.
    """

    # TODO: Check that this is a How to guide.

    # check QUESTION_HEADER is in post
    full_str = "".join(lines)
    if not QUESTION_HEADER in full_str:
        return -1e12, "Question not found."

    for i, line in enumerate(lines):
        if QUESTION_HEADER in line:
            return i, lines[i + 1]


def get_contents_until_next_section(lines, start_idx=0):

    """
    Get the contents of a section of a docusaurus markdown file.
    """

    _i = start_idx + 1
    N = len(lines)
    contents = []
    while _i < N and not lines[_i].startswith("#"):
        try:
            if lines[_i + 1].startswith("-"):
                break
        except IndexError:
            pass  # end of file
        contents.append(lines[_i])
        _i += 1
    return " ".join(contents).strip()


def parse_md_file_headers(markdown_file_path, deployment_url, subdir):

    """
    Parse the header sections of a docusaurus markdown file.

    :param markdown_file_path: str - path to a markdown file.
    :param deployment_url: str - the url of the deployment used to generate links to web pages.
    :param subdir: str - the subdirectory of the repo where the markdown files should be walked for.

    Return a dataframe with the following columns:
        - header: str - the header text
        - contents: str - the contents of the section the header opens
        - type: str - the type of header, e.g. H1, H2, H3, H4, H5
        - page_url: str - the url of the page this header is on with the header slug appended to it

    """

    if not (
        deployment_url.startswith("https://") or deployment_url.startswith("http://")
    ):
        deployment_url = "https://" + deployment_url

    def _try_read_line(_i, lines):
        try:
            return lines[_i]
        except IndexError:
            return END_ESCAPE

    page_type = get_page_type(markdown_file_path)

    def _construct_url(header):
        def strip_slug(s):
            return s.replace(">", "").replace("<", "").replace(".", "").replace("=", "")

        oss_release_notes_regex = "\d.\d.\d \((.*)\d{4}\)"
        res = re.match(oss_release_notes_regex, header)
        if res:
            slug = slugify(strip_slug(res.group()))
        else:
            slug = slugify(strip_slug(header))
        if "github" in slug:
            slug = slug.split("-https-github")[0]

        with open(markdown_file_path, "r") as f:
            fm_meta = frontmatter.load(f).metadata
        if page_type == "blog":
            url = os.path.join(deployment_url, subdir, fm_meta["slug"] + "#" + slug)
        elif page_type == "docs":
            try:
                prefix = fm_meta["slug"][1:]
            except KeyError:
                prefix = markdown_file_path.split("docs")[-1].split(".")[0]
            if deployment_url.endswith("/") or prefix.startswith("/"):
                url = deployment_url + prefix + "#" + slug
            else:
                url = os.path.join(deployment_url, prefix + "#" + slug)

        return url

    def _add_one_row(headers_df, header, embeddable_chunk, type, page_url):
        "This function is only intended to be used with parse_md_file_headers below."
        headers_df["header"].append(header)
        headers_df["contents"].append(embeddable_chunk)
        headers_df["type"].append(type)
        headers_df["page_url"].append(page_url)

    def _process_special_heading(x):
        "Artifact of how Outerbounds customizes docusaurus markdown files."
        numbering_regex = re.compile("<NumberHeading number=(.*)</NumberHeading>")
        res = numbering_regex.match(x)
        if res and res.group() == x:
            return x.split(">")[1].split("<")[0].strip()
        return x

    headers_df = {
        "header": [],
        "contents": [],  # This is a list of embedding chunks.
        "type": [],
        "page_url": [],
    }

    lines = list(filter(is_useful_line, get_lines(markdown_file_path)))

    code_block_open = False
    for i, line in enumerate(lines):

        if not code_block_open:

            if line.startswith("#"):
                # This is a header line.
                n_hashes = len(HEADERS_REGEX.match(line).group())
                embeddable_chunk = get_contents_until_next_section(lines, start_idx=i)
                header = _process_special_heading(line.strip().replace("#", "").strip())
                page_url = _construct_url(header)
                _add_one_row(
                    headers_df,
                    header,
                    embeddable_chunk,
                    "H{}".format(n_hashes),
                    page_url,
                )

            elif line.startswith("--"):

                # Another way header lines get specified in .md docs is above --* lines.
                # Walk up lines until we find the header for this section.
                _i = i - 1
                _prev = _try_read_line(_i, lines)
                while "\n" in _prev.strip():
                    res = _try_read_line(_i, lines)
                    if res != END_ESCAPE:
                        _prev = res
                    else:
                        break
                    _i -= 1
                header = _process_special_heading(
                    _prev.strip().replace("#", "").strip()
                )
                embeddable_chunk = get_contents_until_next_section(lines, start_idx=i)
                page_url = _construct_url(header)
                _add_one_row(headers_df, header, embeddable_chunk, "H2", page_url)

            elif line.startswith("```"):
                code_block_open = True
                # TODO: do something wiser with these code blocks.

        else:
            # check if code block is closing
            if line.startswith("```"):
                code_block_open = False

    headers_df = pd.DataFrame(headers_df)
    headers_df["is_howto"] = (
        headers_df.header.apply(lambda s: s.lower()) == "question"
    )
    headers_df["char_count"] = [len(c) for c in headers_df["contents"]]
    headers_df["word_count"] = [len(c.split(" ")) for c in headers_df["contents"]]

    return headers_df


def headers_df_from_file_list(md_file_path_list, deployment_url, subdir):
    """
    Fetches headers from all markdown files in the md_file_path_list directory.
    """

    headers_df = pd.DataFrame()
    for md_file_path in md_file_path_list:
        headers_df_iter = parse_md_file_headers(
            md_file_path, deployment_url=deployment_url, subdir=subdir
        )
        headers_df = pd.concat([headers_df, headers_df_iter], axis=0)

    headers_df.index = range(len(headers_df))
    headers_df = headers_df[headers_df["char_count"] > 0]
    return headers_df


def get_page_type(md_file):
    if "blog" in md_file:
        return "blog"
    elif "docs" in md_file:
        return "docs"
    else:
        raise ValueError(
            "process_docusaurus_page only tested against docs or blog page, not {}".format(
                md_file
            )
        )


def process_docusaurus_page(md_file):

    # extract front matter
    with open(md_file, "r") as f:
        fm_meta = frontmatter.load(f).metadata

    # fetch full document
    with open(md_file, "r") as f:
        unfiltered_contents = f.readlines()

    filtered_contents = list(filter(is_useful_line, unfiltered_contents))
    processed_contents = list(map(lambda s: s.replace("\n", ""), filtered_contents))

    # identify type of the page
    page_type = get_page_type(md_file)

    # extract top-level question for how-tos
    if page_type == "docs":
        cutoff_idx, question = find_question(processed_contents)

    return processed_contents, fm_meta


def save_docusaurus_page_as_data_sample(local_in_path, root_dir=os.path.abspath(".")):
    """
    TODO: address magic formatting/number vibes in local_out_path = ... and 'full text': ... lines
    """
    assert os.path.exists(root_dir)  # TODO: throw error with useful message
    local_out_path = os.path.join(
        root_dir,
        local_in_path.split("/")[-1].replace(
            ".md", f"_{datetime.now().strftime('%Y-%m-%d')}.pkl"
        ),
    )
    lines, fm = process_docusaurus_page(local_in_path)
    result = {"full text": "\n".join(lines), "front matter": fm}
    with open(local_out_path, "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return local_out_path


def load_docusaurus_page_data_sample(local_path):
    with open(local_path, "rb") as handle:
        return pickle.load(handle)


def find_markdown_links(file_path):
    links = []
    link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"  # regex pattern for [text](url)

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    for match in re.finditer(link_pattern, content):
        text, url = match.groups()
        links.append((text, url))

    return links


def get_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    return lines



class Mixin:

    def headers_df_parser(self, filepath, filename, deployment_url, subdir) -> pd.DataFrame:
        """
        Example of a custom parser. 
        This one parses markdown files one section/row per header and returns a dataframe.
        The section markdown is still raw in the contents column.
        """
        return parse_md_file_headers(filepath, deployment_url=deployment_url, subdir=subdir)


    def load_df_from_repo_list(self) -> pd.DataFrame:

        """
        Built to parse the headers of the self.repo_params
        Assumes self.repo_params is set. 
        See: config/repo_params.py and markdown_chunker.py for example.
        """

        from rag_tools.repo.ops import ensure_repo_exists_locally, DocumentationExtractor
        import pandas as pd

        headers_df = pd.DataFrame()

        for params in self.repo_params:

            if not (
                params["repository_path"].startswith("https") or \
                params["repository_path"].startswith("http") or \
                params["repository_path"].startswith("ssh") or \
                params["repository_path"].startswith("git")
            ):
                print("Looking for local repository at %s" % params["repository_path"])
                extractor = DocumentationExtractor(
                    local_repo_path = ensure_repo_exists_locally(
                        local_repo_path = params["repository_path"],
                        ref = params["repository_ref"],
                    )
                )

            else:
                print("Looking for remote repository at %s" % params["repository_path"])
                extractor = DocumentationExtractor(repo_url=params["repository_path"])

            _dfs = extractor.extract(
                base_path=params["base_search_path"],
                ref=params["repository_ref"],
                exclude_paths=params["exclude_paths"],
                exclude_files=params["exclude_files"],
                considered_extensions=[".md"],
                parser=partial(self.headers_df_parser, deployment_url=params['deployment_url'], subdir=params['base_search_path']),
            )
            _df = pd.concat(_dfs, axis=0)
            _df.index = range(len(_df))

            headers_df = pd.concat([headers_df, _df], axis=0, ignore_index=True)

        return headers_df