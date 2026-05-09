"""γ.5 synthetic tool-trace generator.

Builds chat-format training records that teach the model the v0.7 textual
tool-call protocol. Each record looks like:

  system: constitution + tools-block (per docs/PHASE_GAMMA_PROTOCOL.md)
  user: a question that legitimately needs a tool call (or doesn't,
             for negative examples)
  assistant: optional preamble + <tool>NAME(ARG)</tool>
             + <tool_result>REAL_RESULT</tool_result>
             + integrated final answer (biblically-grounded in tone)

Teacher: qwen2.5:32b via Ollama at localhost:11434, temp=0.4. The teacher
writes the integration text given (question, tool_result). Tool results
are REAL -- we execute the tool to get them, never simulated.

Usage:
    PYTHONPATH=. ~/.azriel/.venv/bin/python scripts/47_generate_tool_traces.py [N]

Default N=10 (validation). Output: ~/.azriel/data/synthetic/tool_traces.jsonl
(append-mode so multiple cron cycles accumulate).
"""
import json
import os
import random
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

from azriel.tools import REGISTRY, call, system_prompt_block

OUT_PATH = Path.home() / ".azriel" / "data" / "synthetic" / "tool_traces.jsonl"
SYSTEM_BASE = (Path.home() / ".azriel" / "AZRIEL_CONSTITUTION_SYSTEM.txt").read_text().strip()
SYSTEM = SYSTEM_BASE + "\n\n" + system_prompt_block()

OLLAMA = "http://localhost:11434/api/generate"
# Teacher overridable so we can use a smaller model (e.g. qwen2.5:14b) when
# the 30B-A3B serving model is hot on the same Metal device. The 32B teacher
# OOMs alongside the live server on a 96GB development machine.
TEACHER = os.environ.get("TRACE_TEACHER", "qwen2.5:32b")

BIBLE_REFS = [
    ("John 3:16", "What does John 3:16 say?"),
    ("John 3:16", "Quote John three sixteen for me."),
    ("Romans 8:28", "Tell me what Romans 8:28 teaches."),
    ("Romans 8:28-30", "What does Paul say in Romans 8:28-30?"),
    ("Genesis 1:1", "What is the opening verse of Genesis?"),
    ("Genesis 1:26-27", "Show me the creation of humanity in Genesis."),
    ("Genesis 3:15", "What is the protoevangelium in Genesis?"),
    ("Exodus 3:14", "Show me the divine name from Exodus 3."),
    ("Exodus 20:1-17", "Recite the Ten Commandments."),
    ("Deuteronomy 6:4-5", "Quote the Shema."),
    ("Joshua 1:9", "Encourage me with Joshua 1:9."),
    ("Psalms 1", "Recite Psalm 1."),
    ("Psalms 23", "Recite Psalm 23 for me."),
    ("Psalms 51:10", "Pray Psalm 51:10 with me."),
    ("Psalms 91", "Show me Psalm 91 for protection."),
    ("Psalms 119:105", "Quote Psalm 119:105."),
    ("Proverbs 3:5-6", "Show me Proverbs 3:5-6."),
    ("Isaiah 9:6", "Quote the messianic prophecy in Isaiah 9:6."),
    ("Isaiah 53:5", "What does Isaiah 53:5 say about the suffering servant?"),
    ("Isaiah 55:8-9", "Show me Isaiah 55:8-9."),
    ("Jeremiah 29:11", "Quote Jeremiah 29:11."),
    ("Matthew 5:3-12", "List the Beatitudes."),
    ("Matthew 6:9-13", "Recite the Lord's Prayer."),
    ("Matthew 28:19-20", "Quote the Great Commission."),
    ("Mark 16:15-18", "Show me the longer ending of Mark."),
    ("Luke 4:18-19", "What did Jesus read in the synagogue?"),
    ("John 1:1-3", "Quote the prologue of John."),
    ("John 14:6", "Show me John 14:6."),
    ("John 14:26", "What does Jesus say about the Comforter?"),
    ("John 16:13", "Quote John 16:13 about the Spirit of truth."),
    ("Acts 1:8", "Show me Acts 1:8."),
    ("Acts 2:1-4", "Recount Pentecost from Acts 2."),
    ("Acts 2:4", "Show me Acts 2:4."),
    ("Acts 2:38", "What does Peter say in Acts 2:38?"),
    ("Acts 4:31", "Show me Acts 4:31."),
    ("Acts 10:44-46", "What happened at Cornelius' house?"),
    ("Acts 19:6", "Show me Acts 19:6."),
    ("Romans 1:16-17", "Quote Romans 1:16-17."),
    ("Romans 5:8", "Show me Romans 5:8."),
    ("Romans 6:23", "Quote Romans 6:23."),
    ("Romans 8:38-39", "Show me Romans 8:38-39."),
    ("Romans 10:9-10", "What does Romans 10 say about confession?"),
    ("Romans 12:1-2", "Quote Romans 12:1-2."),
    ("Galatians 5:22-23", "List the fruit of the Spirit from Galatians 5."),
    ("Ephesians 2:8-9", "Show me Ephesians 2:8-9."),
    ("Ephesians 6:10-18", "Show me the armor of God passage."),
    ("Philippians 2:5-11", "Quote the Christ hymn from Philippians 2."),
    ("Philippians 4:6-7", "Quote Philippians 4:6-7."),
    ("Philippians 4:13", "Show me Philippians 4:13."),
    ("Colossians 3:1-4", "Quote Colossians 3:1-4."),
    ("Hebrews 4:12", "What does Hebrews 4:12 say about the word of God?"),
    ("Hebrews 11:1", "Define faith using Hebrews 11:1."),
    ("Hebrews 12:1-2", "Quote Hebrews 12:1-2."),
    ("James 1:2-4", "What does James say about trials?"),
    ("James 5:14-15", "Show me the elders praying for the sick."),
    ("Revelation 3:20", "Quote Revelation 3:20."),
    ("Revelation 21:1-4", "Show me the new heaven and new earth."),
]

CROSSREF_REFS = [
    ("John 3:16", "What are the cross-references for John 3:16?"),
    ("Romans 5:8", "What other verses connect to Romans 5:8?"),
    ("Acts 2:38", "Find me the cross-references for Acts 2:38."),
    ("Genesis 1:1", "What scriptures parallel Genesis 1:1?"),
    ("Hebrews 11:1", "Cross-reference Hebrews 11:1."),
    ("Isaiah 53:5", "Cross-reference Isaiah 53:5."),
    ("Matthew 28:19", "What scriptures relate to Matthew 28:19?"),
    ("Acts 1:8", "Find scriptures parallel to Acts 1:8."),
    ("Romans 8:28", "What verses connect to Romans 8:28?"),
    ("Galatians 5:22", "Cross-reference Galatians 5:22."),
    ("Ephesians 2:8", "What scriptures parallel Ephesians 2:8?"),
    ("John 14:6", "Find cross-references for John 14:6."),
    ("Genesis 3:15", "What scriptures connect to Genesis 3:15?"),
    ("Psalms 23:1", "Cross-reference Psalm 23:1."),
    ("Revelation 3:20", "Find scriptures parallel to Revelation 3:20."),
]

MEMORY_QUERIES = [
    ("Pentecost", "What do you remember about Pentecost?"),
    ("baptism Holy Spirit", "Recall what we have noted about Spirit baptism."),
    ("v0.6.0 release", "What do you remember about v0.6.0?"),
    ("user preference", "What is the user's doctrinal preference based on memory?"),
    ("Phase architecture", "What architectural notes are in your memory?"),
    ("tongues", "What do you remember about speaking in tongues?"),
    ("dispensationalism", "Recall what we have noted about dispensations."),
    ("safety floor", "What do you remember about Azriel's safety boundaries?"),
    ("training data", "What is in your memory about training data?"),
    ("LoRA adapter", "What memory do you have about LoRA adapters?"),
    ("doctrine", "What doctrinal positions are stored in memory?"),
    ("", "Recall details about ."),
    ("", "What do you remember about ?"),
    ("Qwen3", "What memory entries mention Qwen3?"),
    ("MLX", "What do you remember about MLX?"),
]

STRONGS_REFS = [
    ("H1", "What does Strong's H1 mean?"),
    ("H120", "Look up Strong's H120 -- what is the Hebrew word and root?"),
    ("H430", "Show me Strong's H430."),
    ("H3068", "What is Strong's H3068, the divine name?"),
    ("H113", "Look up H113 in Strong's lexicon."),
    ("H7225", "Define Strong's H7225 (the first word of Genesis)."),
    ("H1254", "Show me Strong's H1254 -- the verb 'create' in Genesis 1:1."),
    ("H7307", "What does Strong's H7307 (ruach) mean?"),
    ("H776", "Look up Strong's H776."),
    ("H7965", "What does Strong's H7965 (shalom) mean?"),
]

PDF_EXTRACT_PATHS = [
    ("missler/65_Jude/65_Jude_Commentary_Handbook.pdf|1-2", "Open the Missler Jude handbook and pull pages 1-2 for me."),
    ("missler/45_Romans/45_Romans_Commentary_Handbook.pdf|1-2", "Read the first two pages of Missler's Romans handbook."),
    ("missler/40_Matthew/40_Matthew_Commentary_Handbook.pdf|1-2", "Pull pages 1-2 from the Matthew handbook by Missler."),
    ("missler/49_Ephesians/49_Ephesians_Commentary_Handbook.pdf|1-2", "Show me the opening pages of Missler's Ephesians handbook."),
    ("missler/66_Revelation/66_Revelation_Commentary_Handbook.pdf|1-2", "Read pages 1-2 of the Missler Revelation handbook."),
]

WEATHER_QUERIES = [
    ("Phoenix", "What's the weather like in Phoenix right now?"),
    ("Tempe", "How's the weather in Tempe today?"),
    ("Tucson", "Will it rain in Tucson tomorrow?"),
    ("Flagstaff", "Is it cold in Flagstaff this week?"),
    ("Jerusalem", "What's the weather in Jerusalem today?"),
    ("Nashville", "Tell me the forecast for Nashville."),
    ("Dallas", "How hot is Dallas right now?"),
    ("Atlanta", "What's the weather in Atlanta?"),
    ("Seattle", "Is it raining in Seattle?"),
    ("Mesa", "Forecast for Mesa, Arizona."),
]

WEB_SEARCH_QUERIES = [
    ("global revival news 2026", "What's the latest news about Christian revivals?"),
    ("Asbury revival follow-up", "Find recent reporting on the Asbury revival aftermath."),
    ("biblical archaeology discoveries 2026", "Any new biblical archaeology finds this year?"),
    ("Christian persecution news", "What is the latest news on Christian persecution worldwide?"),
    ("Dead Sea Scrolls research", "Find recent research on the Dead Sea Scrolls."),
    ("Israel news today", "What is happening in Israel today?"),
    ("missions news Africa", "Find recent missions news from Africa."),
    ("worship music charts 2026", "What worship songs are charting this year?"),
]

WEB_FETCH_PAGES = [
    ("https://en.wikipedia.org/wiki/Pentecost", "Pull the Wikipedia article on Pentecost and give me the gist."),
    ("https://en.wikipedia.org/wiki/Azusa_Street_Revival", "Read the Azusa Street Revival article and summarize."),
    ("https://en.wikipedia.org/wiki/Holy_Spirit", "Fetch the Wikipedia entry on the Holy Spirit and summarize."),
    ("https://en.wikipedia.org/wiki/Christian_revival", "Read the article on Christian revivals and summarize."),
    ("https://en.wikipedia.org/wiki/Speaking_in_tongues", "Fetch the page on speaking in tongues and summarize."),
]

GITHUB_QUERIES = [
    ("repo:ggerganov/llama.cpp", "Find Georgi's llama.cpp repo on GitHub."),
    ("user:karpathy nanoGPT", "Find Karpathy's nanoGPT."),
    ("repo:ml-explore/mlx", "Find Apple's MLX repo."),
    ("user:ml-explore mlx-lm", "Find the mlx-lm package."),
    ("repo:Significant-Gravitas/AutoGPT", "Find the AutoGPT repository."),
]

MEMORY_INSERT_TEXTS = [
    ("user prefers the BSB translation", "Remember that I prefer the Berean Standard Bible translation."),
    ("user is on a Apple Silicon Mac with 96GB RAM", "Make a note that I am running on a development machine with 96GB."),
    ("user is building Azriel as a personal biblically-based AI", "Save the fact that I am building you as my personal biblically-based AI assistant."),
    ("user attends a local biblically-based church", "Remember that I attend a local biblically-based church."),
    ("user wants concise pastoral answers, not lectures", "Note for next time: I prefer concise pastoral answers, not long lectures."),
]

FS_LIST_PATHS = [
    (".", "What files do I have in my Azriel sandbox?"),
    (".", "Show me what's in my files folder."),
    (".", "List the documents I've saved with you."),
]

FS_READ_PATHS = [
    ("hello.txt", "Read the hello.txt file in my sandbox."),
    ("hello.txt", "What's in hello.txt?"),
]

IMAGE_SEARCH_QUERIES = [
    ("Pentecost stained glass", "Find me some stained glass images of Pentecost."),
    ("Holy Land aerial photos", "Find aerial photos of the Holy Land."),
    ("Sea of Galilee sunrise", "Show me images of the Sea of Galilee at sunrise."),
    ("Jerusalem old city", "Find pictures of Jerusalem's old city."),
    ("manuscript Codex Sinaiticus", "Find images of the Codex Sinaiticus."),
]

DOCUMENT_CREATE_INPUTS = [
    (
        "docx|sermon-notes|Sermon Notes\n\nText: Acts 2:1-4\n\nPoint 1: The promise of the Father.\n\nPoint 2: The wind and fire.\n\nPoint 3: The new tongues -- a sign for the nations.",
        "Create a docx of sermon notes on Acts 2:1-4 with three points.",
    ),
    (
        "xlsx|prayer-list|Name,Need,Date\nMom,healing,\nFriend J,job,\nSister K,wisdom,",
        "Make me a small xlsx prayer list with three rows.",
    ),
    (
        "pptx|gospel-outline|The Gospel\n---\nWho is Jesus?\nThe Son of God, the promised Messiah.\n---\nWhy did he come?\nTo seek and save the lost.\n---\nHow do we respond?\nRepent and believe.",
        "Generate a 4-slide pptx outline of the gospel: who, why, response.",
    ),
]

VISUALIZE_INPUTS = [
    (
        '<svg viewBox="0 0 100 60" xmlns="http://www.w3.org/2000/svg"><rect x="5" y="5" width="90" height="50" fill="none" stroke="#d97757" stroke-width="2"/><text x="50" y="35" text-anchor="middle" font-family="serif" font-size="10" fill="#d97757">Pentecost</text></svg>',
        "Draw me a simple banner with the word Pentecost.",
    ),
    (
        '<svg viewBox="0 0 120 80" xmlns="http://www.w3.org/2000/svg"><circle cx="60" cy="40" r="22" fill="none" stroke="#c2553a" stroke-width="2"/><circle cx="60" cy="40" r="14" fill="none" stroke="#c2553a" stroke-width="2"/><circle cx="60" cy="40" r="6" fill="#c2553a"/></svg>',
        "Show me a simple concentric-circles diagram for the Trinity centered on the Father.",
    ),
    (
        '<table style="border-collapse:collapse;font-family:serif"><tr><th style="border:1px solid #555;padding:4px">Fruit</th><th style="border:1px solid #555;padding:4px">Reference</th></tr><tr><td style="border:1px solid #555;padding:4px">Love</td><td style="border:1px solid #555;padding:4px">Gal 5:22</td></tr><tr><td style="border:1px solid #555;padding:4px">Joy</td><td style="border:1px solid #555;padding:4px">Gal 5:22</td></tr><tr><td style="border:1px solid #555;padding:4px">Peace</td><td style="border:1px solid #555;padding:4px">Gal 5:22</td></tr></table>',
        "Make a small table of the first three fruits of the Spirit.",
    ),
]

NEGATIVE_PROMPTS = [
    "Hello, how are you today?",
    "Why do you exist?",
    "What is salvation?",
    "How should I pray?",
    "What does it mean to be born again?",
    "Tell me about the Trinity.",
    "What is the gospel?",
    "How do I grow spiritually?",
    "What is repentance?",
    "Who is Jesus?",
]


def ollama_generate(prompt: str, max_tokens: int = 200, temperature: float = 0.4) -> str:
    body = {
        "model": TEACHER,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    req = Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as r:
        return json.loads(r.read()).get("response", "").strip()


def make_integration(question: str, tool_call: str, tool_result: str) -> str:
    """Ask the teacher to write the assistant's integration text."""
    p = (
        "You are completing an example of how Azriel (a biblically-based "
        "AI assistant) integrates a tool result into a final answer.\n\n"
        f"User asked: {question}\n"
        f"Azriel called: {tool_call}\n"
        f"Tool returned: {tool_result}\n\n"
        "Write Azriel's final 1-3 sentence answer that uses the tool result. "
        "Be concise and biblically grounded in tone. "
        "Do NOT call any more tools. Do NOT mention 'tool_result' or markup. "
        "Just the final natural answer:"
    )
    return ollama_generate(p, max_tokens=180)


def build_assistant_turn(preamble: str, tool_call: str, tool_result: str, integration: str) -> str:
    parts = []
    if preamble:
        parts.append(preamble)
    parts.append(f"<tool>{tool_call}</tool>")
    parts.append(f"<tool_result>{tool_result}</tool_result>")
    parts.append(integration)
    return "\n".join(parts)


def trace_bible(ref: str, question: str) -> dict:
    tool_call = f'bible_lookup("{ref}")'
    result = call("bible_lookup", ref)
    integration = make_integration(question, tool_call, result)
    asst = build_assistant_turn("Let me look that up.", tool_call, result, integration)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": asst},
    ], "tool": "bible_lookup", "ref": ref}


def trace_crossref(ref: str, question: str) -> dict:
    tool_call = f'crossref_lookup("{ref}")'
    result = call("crossref_lookup", ref)
    integration = make_integration(question, tool_call, result)
    asst = build_assistant_turn("Let me check the cross-references.", tool_call, result, integration)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": asst},
    ], "tool": "crossref_lookup", "ref": ref}


def trace_memory(query: str, question: str) -> dict:
    tool_call = f'memory_search("{query}")'
    result = call("memory_search", query)
    integration = make_integration(question, tool_call, result)
    asst = build_assistant_turn("Let me search my memory.", tool_call, result, integration)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": asst},
    ], "tool": "memory_search", "query": query}


def trace_strongs(ref: str, question: str) -> dict:
    tool_call = f'strongs_lookup("{ref}")'
    result = call("strongs_lookup", ref)
    integration = make_integration(question, tool_call, result)
    asst = build_assistant_turn("Let me check the lexicon.", tool_call, result, integration)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": asst},
    ], "tool": "strongs_lookup", "ref": ref}


def trace_pdf_extract(path: str, question: str) -> dict:
    tool_call = f'pdf_extract("{path}")'
    result = call("pdf_extract", path)
    integration = make_integration(question, tool_call, result)
    asst = build_assistant_turn("Reading that handbook now.", tool_call, result, integration)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": asst},
    ], "tool": "pdf_extract", "ref": path}


def trace_weather(location: str, question: str) -> dict:
    tool_call = f'weather("{location}")'
    result = call("weather", location)
    integration = make_integration(question, tool_call, result)
    asst = build_assistant_turn("Let me check the forecast.", tool_call, result, integration)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": asst},
    ], "tool": "weather", "ref": location}


def trace_web_search(query: str, question: str) -> dict:
    tool_call = f'web_search("{query}")'
    result = call("web_search", query)
    integration = make_integration(question, tool_call, result)
    asst = build_assistant_turn("Let me search the web.", tool_call, result, integration)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": asst},
    ], "tool": "web_search", "ref": query}


def trace_web_fetch(url: str, question: str) -> dict:
    tool_call = f'web_fetch("{url}")'
    result = call("web_fetch", url)
    integration = make_integration(question, tool_call, result)
    asst = build_assistant_turn("Let me read that page.", tool_call, result, integration)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": asst},
    ], "tool": "web_fetch", "ref": url}


def trace_github(query: str, question: str) -> dict:
    tool_call = f'github_query("{query}")'
    result = call("github_query", query)
    integration = make_integration(question, tool_call, result)
    asst = build_assistant_turn("Let me search GitHub.", tool_call, result, integration)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": asst},
    ], "tool": "github_query", "ref": query}


def trace_memory_insert(text: str, question: str) -> dict:
    tool_call = f'memory_insert("{text}")'
    result = call("memory_insert", text)
    integration = make_integration(question, tool_call, result)
    asst = build_assistant_turn("Saving that to memory.", tool_call, result, integration)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": asst},
    ], "tool": "memory_insert", "ref": text}


def trace_fs_list(path: str, question: str) -> dict:
    tool_call = f'fs_list("{path}")'
    result = call("fs_list", path)
    integration = make_integration(question, tool_call, result)
    asst = build_assistant_turn("Let me check the sandbox.", tool_call, result, integration)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": asst},
    ], "tool": "fs_list", "ref": path}


def trace_fs_read(path: str, question: str) -> dict:
    tool_call = f'fs_read("{path}")'
    result = call("fs_read", path)
    integration = make_integration(question, tool_call, result)
    asst = build_assistant_turn("Reading that file now.", tool_call, result, integration)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": asst},
    ], "tool": "fs_read", "ref": path}


def trace_image_search(query: str, question: str) -> dict:
    tool_call = f'image_search("{query}")'
    result = call("image_search", query)
    integration = make_integration(question, tool_call, result)
    asst = build_assistant_turn("Let me search for images.", tool_call, result, integration)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": asst},
    ], "tool": "image_search", "ref": query}


def trace_document_create(spec: str, question: str) -> dict:
    tool_call = f'document_create({json.dumps(spec)})'
    result = call("document_create", spec)
    integration = make_integration(question, tool_call, result)
    asst = build_assistant_turn("Generating that for you.", tool_call, result, integration)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": asst},
    ], "tool": "document_create", "ref": spec[:60]}


def trace_visualize(svg_or_html: str, question: str) -> dict:
    tool_call = f'visualize({json.dumps(svg_or_html)})'
    result = call("visualize", svg_or_html)
    integration = make_integration(question, tool_call, result)
    asst = build_assistant_turn("Rendering that for you.", tool_call, result, integration)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": asst},
    ], "tool": "visualize", "ref": svg_or_html[:60]}


def trace_negative(question: str) -> dict:
    """No tool call -- pure assistant response, teaches when NOT to fire."""
    p = (
        "You are Azriel, a biblically-based AI assistant. Answer the "
        f"following user question concisely (2-4 sentences) without calling any tools:\n\n"
        f"User: {question}\nAzriel:"
    )
    answer = ollama_generate(p, max_tokens=200)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ], "tool": None}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    random.seed(int(time.time()))

    # Build a balanced batch
    pool = []
    for ref, q in BIBLE_REFS:
        pool.append(("bible", ref, q))
    for ref, q in CROSSREF_REFS:
        pool.append(("crossref", ref, q))
    for query, q in MEMORY_QUERIES:
        pool.append(("memory", query, q))
    for ref, q in STRONGS_REFS:
        pool.append(("strongs", ref, q))
    for path, q in PDF_EXTRACT_PATHS:
        pool.append(("pdf_extract", path, q))
    for loc, q in WEATHER_QUERIES:
        pool.append(("weather", loc, q))
    for query, q in WEB_SEARCH_QUERIES:
        pool.append(("web_search", query, q))
    for url, q in WEB_FETCH_PAGES:
        pool.append(("web_fetch", url, q))
    for gq, q in GITHUB_QUERIES:
        pool.append(("github", gq, q))
    for txt, q in MEMORY_INSERT_TEXTS:
        pool.append(("memory_insert", txt, q))
    for path, q in FS_LIST_PATHS:
        pool.append(("fs_list", path, q))
    for path, q in FS_READ_PATHS:
        pool.append(("fs_read", path, q))
    for query, q in IMAGE_SEARCH_QUERIES:
        pool.append(("image_search", query, q))
    for spec, q in DOCUMENT_CREATE_INPUTS:
        pool.append(("document_create", spec, q))
    for svg, q in VISUALIZE_INPUTS:
        pool.append(("visualize", svg, q))
    for q in NEGATIVE_PROMPTS:
        pool.append(("negative", None, q))

    # Sampling strategy. Default = uniform random over the pool (proportional
    # to tool-pool size; bible_lookup dominates). With BALANCED_SAMPLING=1
    # we round-robin over tool kinds first, then fill with random remainder
    # -- guarantees each tool kind gets ~equal exposure regardless of how
    # many seeds are in its pool.
    if os.environ.get("BALANCED_SAMPLING") == "1":
        by_kind: dict[str, list] = {}
        for item in pool:
            by_kind.setdefault(item[0], []).append(item)
        kinds = list(by_kind)
        for k in kinds:
            random.shuffle(by_kind[k])
        per_kind = max(1, n // len(kinds))
        balanced = []
        for k in kinds:
            for i in range(per_kind):
                balanced.append(by_kind[k][i % len(by_kind[k])])
        # Fill remainder uniformly
        random.shuffle(pool)
        idx = 0
        while len(balanced) < n:
            balanced.append(pool[idx % len(pool)])
            idx += 1
        random.shuffle(balanced)
        pool = balanced[:n]
    elif n <= len(pool):
        random.shuffle(pool)
        pool = pool[:n]
    else:
        pool = [random.choice(pool) for _ in range(n)]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT_PATH.open("a") as f:
        for kind, arg, q in pool:
            t0 = time.time()
            try:
                if kind == "bible":
                    rec = trace_bible(arg, q)
                elif kind == "crossref":
                    rec = trace_crossref(arg, q)
                elif kind == "memory":
                    rec = trace_memory(arg, q)
                elif kind == "strongs":
                    rec = trace_strongs(arg, q)
                elif kind == "pdf_extract":
                    rec = trace_pdf_extract(arg, q)
                elif kind == "weather":
                    rec = trace_weather(arg, q)
                elif kind == "web_search":
                    rec = trace_web_search(arg, q)
                elif kind == "web_fetch":
                    rec = trace_web_fetch(arg, q)
                elif kind == "github":
                    rec = trace_github(arg, q)
                elif kind == "memory_insert":
                    rec = trace_memory_insert(arg, q)
                elif kind == "fs_list":
                    rec = trace_fs_list(arg, q)
                elif kind == "fs_read":
                    rec = trace_fs_read(arg, q)
                elif kind == "image_search":
                    rec = trace_image_search(arg, q)
                elif kind == "document_create":
                    rec = trace_document_create(arg, q)
                elif kind == "visualize":
                    rec = trace_visualize(arg, q)
                else:
                    rec = trace_negative(q)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                dt = time.time() - t0
                print(f" [{written}/{n}] {kind:9s} ({dt:.1f}s) {q[:60]}", flush=True)
            except Exception as e:
                print(f" FAIL {kind}: {e}", flush=True)

    print(f"\nwrote {written}/{n} traces to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
