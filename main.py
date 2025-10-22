import asyncio
import configparser
import datetime
import feedparser
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, NamedTuple, List

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from jinja2 import Template
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_PATH = SCRIPT_DIR / "prompt.txt"
SCHEMA_PATH = SCRIPT_DIR / "schema.json"

try:
    PROMPT_TEXT = PROMPT_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    PROMPT_TEXT = None

try:
    ARTICLE_SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
except FileNotFoundError:
    ARTICLE_SCHEMA = None
except json.JSONDecodeError as exc:
    print(f"Warning: Failed to parse schema.json: {exc}", file=sys.stderr)
    ARTICLE_SCHEMA = None


def _strip_text(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


class ArticleScoring(BaseModel):
    interpretability: int = Field(..., description="Interpretability score (1-10)")
    understanding: int = Field(..., description="Understanding score (1-10)")
    safety: int = Field(..., description="Safety score (1-10)")
    technicality: int = Field(..., description="Technicality score (1-10)")
    surprisal: int = Field(..., description="Surprisal score (1-10)")

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_int(cls, value: Any) -> Any:
        value = _strip_text(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
        return value


class ArticleCategory(BaseModel):
    failure_mode_addressed: str = Field(..., description="Failure mode category")
    primary_focus: str = Field(..., description="Primary focus area")

    @field_validator("failure_mode_addressed", "primary_focus", mode="before")
    @classmethod
    def _strip_category(cls, value: Any) -> Any:
        return _strip_text(value)


class ArticleAnalysis(BaseModel):
    summary: str = Field(..., description="Concise summary text")
    summary_cn: str = Field(..., description="Concise Chinese summary text")
    keywords: str = Field(..., description="Comma-separated keywords")
    scoring: ArticleScoring
    category: ArticleCategory

    @field_validator("summary", "summary_cn", "keywords", mode="before")
    @classmethod
    def _strip_text_fields(cls, value: Any) -> Any:
        return _strip_text(value)

    def keywords_list(self) -> Tuple[str, ...]:
        return tuple(kw.strip() for kw in self.keywords.split(",") if kw.strip())

    def to_summary_html(self) -> str:
        blocks = [
            f"<strong>Summary:</strong> {self.summary}",
            f"<strong>Summary (CN):</strong> {self.summary_cn}",
        ]
        keywords = ", ".join(self.keywords_list())
        if keywords:
            blocks.append(f"<strong>Keywords:</strong> {keywords}")
        scores = self.scoring.model_dump()
        score_str = ", ".join(f"{name.capitalize()}: {value}" for name, value in scores.items())
        blocks.append(f"<strong>Scores:</strong> {score_str}")
        category_dump = self.category.model_dump()
        blocks.append(
            "<strong>Categories:</strong> "
            f"Failure mode - {category_dump['failure_mode_addressed']}; "
            f"Primary focus - {category_dump['primary_focus']}"
        )
        return "<br>".join(blocks)

    def to_json_str(self) -> str:
        return self.model_dump_json(ensure_ascii=False)
#from dateutil.parser import parse

def get_cfg(sec, name, default=None):
    value=config.get(sec, name, fallback=default)
    if value:
        return value.strip('"')

config = configparser.ConfigParser()
config.read('config.ini')
secs = config.sections()
# Maxnumber of entries to in a feed.xml file
max_entries = 1000

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('OPENAI_API_KEY')
OPENROUTER_BASE_URL = os.environ.get('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
OPENROUTER_MODEL = os.environ.get('OPENROUTER_MODEL', 'openai/gpt-5-mini')
OPENROUTER_REASONING = os.environ.get('OPENROUTER_REASONING', 'medium')
OPENROUTER_CONCURRENCY = int(os.environ.get('OPENROUTER_CONCURRENCY', '50'))
U_NAME = os.environ.get('U_NAME')
deployment_url = f'https://{U_NAME}.github.io/RSS-GPT/' if U_NAME else ''
BASE = get_cfg('cfg', 'base') or get_cfg('cfg', 'BASE')
keyword_length = int(get_cfg('cfg', 'keyword_length'))
summary_length = int(get_cfg('cfg', 'summary_length'))
DEFAULT_MODEL = get_cfg('cfg', 'model') or OPENROUTER_MODEL

ASYNC_OPENROUTER_CLIENT: Optional[AsyncOpenAI] = None
if OPENROUTER_API_KEY:
    try:
        ASYNC_OPENROUTER_CLIENT = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
    except Exception as exc:
        print(f"Warning: failed to initialise OpenRouter client: {exc}", file=sys.stderr)

SUMMARIZATION_ENABLED = bool(ASYNC_OPENROUTER_CLIENT and PROMPT_TEXT and ARTICLE_SCHEMA)
if not SUMMARIZATION_ENABLED:
    print("Warning: OpenRouter summarisation disabled; ensure OPENROUTER_API_KEY, prompt.txt, and schema.json are available.", file=sys.stderr)
if not BASE:
    BASE = "docs/"

TOTAL_SUMMARIZATION_COST = 0.0
MAX_SUMMARY_ATTEMPTS = 3
SUMMARY_RETRY_DELAY_SECONDS = 1.0

def fetch_feed(url, log_file):
    feed = None
    response = None
    headers = {}
    try:
        ua = UserAgent()
        headers['User-Agent'] = ua.random.strip()
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            feed = feedparser.parse(response.text)
            return {'feed': feed, 'status': 'success'}
        else:
            with open(log_file, 'a') as f:
                f.write(f"Fetch error: {response.status_code}\n")
            return {'feed': None, 'status': response.status_code}
    except requests.RequestException as e:
        with open(log_file, 'a') as f:
            f.write(f"Fetch error: {e}\n")
        return {'feed': None, 'status': 'failed'}

def generate_untitled(entry):
    try: return entry.title
    except: 
        try: return entry.article[:50]
        except: return entry.link


def clean_html(html_content):
    """
    This function is used to clean the HTML content.
    It will remove all the <script>, <style>, <img>, <a>, <video>, <audio>, <iframe>, <input> tags.
    Returns:
        Cleaned text for summarization
    """
    soup = BeautifulSoup(html_content, "html.parser")

    for script in soup.find_all("script"):
        script.decompose()

    for style in soup.find_all("style"):
        style.decompose()

    for img in soup.find_all("img"):
        img.decompose()

    for a in soup.find_all("a"):
        a.decompose()

    for video in soup.find_all("video"):
        video.decompose()

    for audio in soup.find_all("audio"):
        audio.decompose()
    
    for iframe in soup.find_all("iframe"):
        iframe.decompose()
    
    for input in soup.find_all("input"):
        input.decompose()

    return soup.get_text()

def filter_entry(entry, filter_apply, filter_type, filter_rule):
    """
    This function is used to filter the RSS feed.

    Args:
        entry: RSS feed entry
        filter_apply: title, article or link
        filter_type: include or exclude or regex match or regex not match
        filter_rule: regex rule or keyword rule, depends on the filter_type

    Raises:
        Exception: filter_apply not supported
        Exception: filter_type not supported
    """
    if filter_apply == 'title':
        text = entry.title
    elif filter_apply == 'article':
        text = entry.article
    elif filter_apply == 'link':
        text = entry.link
    elif not filter_apply:
        return True
    else:
        raise Exception('filter_apply not supported')

    if filter_type == 'include':
        return re.search(filter_rule, text)
    elif filter_type == 'exclude':
        return not re.search(filter_rule, text)
    elif filter_type == 'regex match':
        return re.search(filter_rule, text)
    elif filter_type == 'regex not match':
        return not re.search(filter_rule, text)
    elif not filter_type:
        return True
    else:
        raise Exception('filter_type not supported')

def read_entry_from_file(sec):
    """
    This function is used to read the RSS feed entries from the feed.xml file.

    Args:
        sec: section name in config.ini
    """
    out_dir = os.path.join(BASE, get_cfg(sec, 'name'))
    try:
        with open(out_dir + '.xml', 'r') as f:
            rss = f.read()
        feed = feedparser.parse(rss)
        return feed.entries
    except:
        return []

def truncate_entries(entries, max_entries):
    if len(entries) > max_entries:
        entries = entries[:max_entries]
    return entries

def usage_to_dict(usage_obj: Any) -> Dict[str, Any]:
    if usage_obj is None:
        return {}
    if hasattr(usage_obj, "model_dump"):
        return usage_obj.model_dump()
    if isinstance(usage_obj, dict):
        return usage_obj
    return getattr(usage_obj, "__dict__", {}) or {}


def extract_cost(usage_info: Dict[str, Any]) -> Optional[float]:
    for key in ("total_cost", "total_cost_usd", "cost", "estimated_cost"):
        value = usage_info.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


async def async_generate_analysis(article_text: str, model_name: Optional[str] = None) -> Tuple[ArticleAnalysis, Dict[str, Any]]:
    if not ASYNC_OPENROUTER_CLIENT:
        raise RuntimeError("Async OpenRouter client not initialised.")
    if not PROMPT_TEXT:
        raise RuntimeError("Prompt text not available.")
    if not ARTICLE_SCHEMA:
        raise RuntimeError("JSON schema not available for structured output.")

    messages = [
        {"role": "system", "content": PROMPT_TEXT},
        {"role": "user", "content": article_text},
    ]

    request_payload: Dict[str, Any] = {
        "model": model_name or OPENROUTER_MODEL,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": ARTICLE_SCHEMA,
        },
        "extra_body": {
            "usage": {
                "include": True
            }
        }
    }

    if OPENROUTER_REASONING:
        request_payload["reasoning_effort"] = OPENROUTER_REASONING

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_SUMMARY_ATTEMPTS + 1):
        try:
            completion = await ASYNC_OPENROUTER_CLIENT.chat.completions.create(**request_payload)
            choice = completion.choices[0]
            parsed_payload: Optional[Dict[str, Any]] = None

            message_obj = getattr(choice, "message", None)
            if message_obj is not None:
                parsed_payload = getattr(message_obj, "parsed", None)
                if parsed_payload is None:
                    content = getattr(message_obj, "content", "")
                    if isinstance(content, str):
                        content_str = content.strip()
                    elif isinstance(content, list):
                        content_str = "".join(
                            (part.get("text", "") if isinstance(part, dict) else str(part))
                            for part in content
                        ).strip()
                    else:
                        content_str = ""
                    if not content_str:
                        raise ValueError("Empty response received from OpenRouter.")
                    try:
                        parsed_payload = json.loads(content_str)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Failed to parse JSON response: {exc}") from exc

            if parsed_payload is None:
                raise ValueError("Structured response missing parsed content.")

            analysis = ArticleAnalysis.model_validate(parsed_payload)
            usage_info = usage_to_dict(completion.usage)
            return analysis, usage_info

        except asyncio.CancelledError:
            raise
        except (ValidationError, ValueError) as exc:
            last_error = exc
        except Exception as exc:  # API/network errors
            last_error = exc

        if attempt == MAX_SUMMARY_ATTEMPTS:
            raise last_error  # type: ignore[misc]

        delay = SUMMARY_RETRY_DELAY_SECONDS * attempt
        print(
            f"Retrying summarization (attempt {attempt}/{MAX_SUMMARY_ATTEMPTS}) after error: {last_error}",
            file=sys.stderr,
        )
        await asyncio.sleep(delay)

    # Should not reach here
    raise RuntimeError("Failed to generate analysis after retries.")


def store_analysis_snapshot(feed_name: str, entry: Any, analysis: ArticleAnalysis, usage: Dict[str, Any], model_name: str) -> None:
    analysis_dir = Path(BASE) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "feed": feed_name,
        "title": getattr(entry, "title", ""),
        "link": getattr(entry, "link", ""),
        "analysis": analysis.model_dump(),
        "usage": usage,
        "model": model_name,
    }
    file_path = analysis_dir / f"{feed_name}.jsonl"
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class SummarizationJob(NamedTuple):
    feed_name: str
    entry: Any
    token_length: int
    article_input: str
    log_file: str
    model_name: str


class SummarizationResult(NamedTuple):
    job: SummarizationJob
    analysis: Optional[ArticleAnalysis]
    usage: Dict[str, Any]
    error: Optional[Exception]


async def process_summaries(
    jobs: Tuple[SummarizationJob, ...],
    on_result: Optional[Any] = None,
) -> None:
    if not jobs:
        return

    semaphore = asyncio.Semaphore(max(1, OPENROUTER_CONCURRENCY))

    async def run_job(job: SummarizationJob) -> SummarizationResult:
        async with semaphore:
            try:
                analysis, usage_info = await async_generate_analysis(job.article_input, job.model_name)
                return SummarizationResult(job=job, analysis=analysis, usage=usage_info or {}, error=None)
            except Exception as exc:
                return SummarizationResult(job=job, analysis=None, usage={}, error=exc)

    tasks = [asyncio.create_task(run_job(job)) for job in jobs]
    for task in asyncio.as_completed(tasks):
        result = await task
        if on_result is not None:
            try:
                on_result(result)
            except Exception as exc:
                print(f"Error handling summarization result: {exc}", file=sys.stderr)


def run_summaries_with_concurrency(
    jobs: Tuple[SummarizationJob, ...],
    on_result: Optional[Any] = None,
) -> None:
    if not jobs:
        return
    return asyncio.run(process_summaries(jobs, on_result))

def output(sec):
    """ output
    This function is used to output the summary of the RSS feed.

    Args:
        sec: section name in config.ini

    Raises:
        Exception: filter_apply, type, rule must be set together in config.ini
    """
    global TOTAL_SUMMARIZATION_COST
    feed_name = get_cfg(sec, 'name')
    log_file = os.path.join(BASE, feed_name + '.log')
    out_dir = os.path.join(BASE, feed_name)
    # read rss_url as a list separated by comma
    rss_urls = get_cfg(sec, 'url')
    rss_urls = rss_urls.split(',')

    # RSS feed filter apply, filter title, article or link, summarize title, article or link
    filter_apply = get_cfg(sec, 'filter_apply')

    # RSS feed filter type, include or exclude or regex match or regex not match
    filter_type = get_cfg(sec, 'filter_type')

    # Regex rule or keyword rule, depends on the filter_type
    filter_rule = get_cfg(sec, 'filter_rule')

    # filter_apply, type, rule must be set together
    if filter_apply and filter_type and filter_rule:
        pass
    elif not filter_apply and not filter_type and not filter_rule:
        pass
    else:
        raise Exception('filter_apply, type, rule must be set together')

    # Max number of items to summarize
    max_items = get_cfg(sec, 'max_items')
    if not max_items:
        max_items = 0
    else:
        max_items = int(max_items)
    cnt = 0
    existing_entries = read_entry_from_file(sec)
    with open(log_file, 'a') as f:
        f.write('------------------------------------------------------\n')
        f.write(f'Started: {datetime.datetime.now()}\n')
        f.write(f'Existing_entries: {len(existing_entries)}\n')
        if not SUMMARIZATION_ENABLED:
            f.write('Summaries skipped: OpenRouter summarisation disabled; keeping original content.\n')
    existing_entries = truncate_entries(existing_entries, max_entries=max_entries)
    # Be careful when the deleted ones are still in the feed, in that case, you will mess up the order of the entries.
    # Truncating old entries is for limiting the file size, 1000 is a safe number to avoid messing up the order.
    append_entries = []
    summarization_jobs: List[SummarizationJob] = []

    for rss_url in rss_urls:
        with open(log_file, 'a') as f:
            f.write(f"Fetching from {rss_url}\n")
            print(f"Fetching from {rss_url}")
        feed = fetch_feed(rss_url, log_file)['feed']
        if not feed:
            with open(log_file, 'a') as f:
                f.write(f"Fetch failed from {rss_url}\n")
            continue
        for entry in feed.entries:
            if cnt > max_entries:
                with open(log_file, 'a') as f:
                    f.write(f"Skip from: [{entry.title}]({entry.link})\n")
                break

            if entry.link.find('#replay') and entry.link.find('v2ex'):
                entry.link = entry.link.split('#')[0]

            if entry.link in [x.link for x in existing_entries]:
                continue

            if entry.link in [x.link for x in append_entries]:
                continue

            entry.title = generate_untitled(entry)

            try:
                entry.article = entry.content[0].value
            except:
                try: entry.article = entry.description
                except: entry.article = entry.title

            cleaned_article = clean_html(entry.article)

            if not filter_entry(entry, filter_apply, filter_type, filter_rule):
                with open(log_file, 'a') as f:
                    f.write(f"Filter: [{entry.title}]({entry.link})\n")
                continue


#            # format to Thu, 27 Jul 2023 13:13:42 +0000
#            if 'updated' in entry:
#                entry.updated = parse(entry.updated).strftime('%a, %d %b %Y %H:%M:%S %z')
#            if 'published' in entry:
#                entry.published = parse(entry.published).strftime('%a, %d %b %Y %H:%M:%S %z')

            cnt += 1
            if cnt > max_items:
                entry.summary = None
            elif SUMMARIZATION_ENABLED:
                token_length = len(cleaned_article)
                model_name = get_cfg(sec, 'model') or DEFAULT_MODEL or OPENROUTER_MODEL
                analysis_input = (
                    f"Feed: {feed_name}\n"
                    f"Title: {entry.title}\n"
                    f"Link: {entry.link}\n"
                    f"{cleaned_article}"
                )
                summarization_jobs.append(
                    SummarizationJob(
                        feed_name=feed_name,
                        entry=entry,
                        token_length=token_length,
                        article_input=analysis_input,
                        log_file=log_file,
                        model_name=model_name,
                    )
                )

            append_entries.append(entry)
            with open(log_file, 'a') as f:
                f.write(f"Append: [{entry.title}]({entry.link})\n")

    if SUMMARIZATION_ENABLED and summarization_jobs:
        print(f"Summarizing {len(summarization_jobs)} entries")

        def handle_summary_result(result: SummarizationResult) -> None:
            global TOTAL_SUMMARIZATION_COST
            job = result.job
            entry = job.entry
            if result.error or not result.analysis:
                entry.summary = None
                error_msg = result.error or ValueError("Missing analysis payload.")
                with open(job.log_file, 'a') as f:
                    f.write("Summarization failed, append the original article\n")
                    f.write(f"error: {error_msg}\n")
                print(f"Summarization failed for '{entry.title}': {error_msg}", file=sys.stderr)
                return

            analysis = result.analysis
            usage_info = result.usage
            entry.summary = analysis.to_summary_html()
            entry.analysis = analysis.model_dump()
            entry.analysis_json = analysis.to_json_str()
            store_analysis_snapshot(job.feed_name, entry, analysis, usage_info, job.model_name)

            cost_value = extract_cost(usage_info)
            usage_json = json.dumps(usage_info, ensure_ascii=False) if usage_info else "{}"
            with open(job.log_file, 'a') as f:
                f.write(f"Token length: {job.token_length}\n")
                f.write("Summarized using OpenRouter\n")
                f.write(f"Model: {job.model_name}\n")
                f.write(f"Usage: {usage_json}\n")
                if cost_value is not None:
                    f.write(f"Cost (USD): {cost_value:.6f}\n")

            if cost_value is not None:
                TOTAL_SUMMARIZATION_COST += cost_value
                print(
                    f"OpenRouter cost for '{entry.title}' (model={job.model_name}): "
                    f"${cost_value:.6f} | running total: ${TOTAL_SUMMARIZATION_COST:.6f}"
                )
            else:
                print(f"OpenRouter usage for '{entry.title}': {usage_json}")

        run_summaries_with_concurrency(tuple(summarization_jobs), handle_summary_result)

    with open(log_file, 'a') as f:
        f.write(f'append_entries: {len(append_entries)}\n')

    template = Template(open('template.xml').read())
    
    try:
        rss = template.render(feed=feed, append_entries=append_entries, existing_entries=existing_entries)
        with open(out_dir + '.xml', 'w') as f:
            f.write(rss)
        with open(log_file, 'a') as f:
            f.write(f'Finish: {datetime.datetime.now()}\n')
    except:
        with open (log_file, 'a') as f:
            f.write(f"error when rendering xml, skip {out_dir}\n")
            print(f"error when rendering xml, skip {out_dir}\n")

try:
    os.mkdir(BASE)
except:
    pass

feeds = []
links = []

for x in secs[1:]:
    output(x)
    feed = {"url": get_cfg(x, 'url').replace(',','<br>'), "name": get_cfg(x, 'name')}
    feeds.append(feed)  # for rendering index.html
    links.append("- "+ get_cfg(x, 'url').replace(',',', ') + " -> " + deployment_url + feed['name'] + ".xml\n")

def append_readme(readme, links):
    with open(readme, 'r') as f:
        readme_lines = f.readlines()
    while readme_lines[-1].startswith('- ') or readme_lines[-1] == '\n':
        readme_lines = readme_lines[:-1]  # remove 1 line from the end for each feed
    readme_lines.append('\n')
    readme_lines.extend(links)
    with open(readme, 'w') as f:
        f.writelines(readme_lines)

append_readme("README.md", links)
append_readme("README-zh.md", links)

# Rendering index.html used in my GitHub page, delete this if you don't need it.
# Modify template.html to change the style
with open(os.path.join(BASE, 'index.html'), 'w') as f:
    template = Template(open('template.html').read())
    html = template.render(update_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), feeds=feeds)
    f.write(html)

if SUMMARIZATION_ENABLED and TOTAL_SUMMARIZATION_COST:
    print(f"Total OpenRouter summarization cost: ${TOTAL_SUMMARIZATION_COST:.6f}")
