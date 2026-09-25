from __future__ import annotations

import httpx
import streamlit as st
import os

st.set_page_config(page_title="FinSight - Financial QA", layout="wide")
st.title("FinSight")
st.caption("Evidence-Grounded Multi-hop QA on SEC Financial Filings")


with st.sidebar:
    st.header(":material/settings: Configuration")
    api_url = st.text_input(
        ":material/api: API_URL", value=os.getenv("API_URL", "http://localhost:8000")
    )
    st.divider()
    st.header(":material/filter_alt: Filters")
    company = st.text_input(
        ":material/business: Company (optional)",
        placeholder="e.g., Apple",
    )
    fiscal_year = st.number_input(
        ":material/calendar_today: FY - Fiscal Year (Optional)",
        min_value=2015,
        max_value=2026,
        value=None,
        placeholder="e.g., 2026",
    )
    include_trace = st.checkbox(":material/account_tree: Show trace", value=True)
    st.divider()
    st.header(":material/info: About")
    st.markdown("**Author** - [ducbao210](https://github.com/ducbao210)")
    st.markdown(
        """**FinSight** is a personal project demonstrating evidence-grounded QA over SEC financial filings.\n\n- **CRAG**: Decompose → Retrieve → Grade → Rewrite → Answer"""
    )

    st.markdown("""**Citation**

    @misc{islam2023financebench,
        title={FinanceBench: A New Benchmark for Financial Question Answering},
        author={Pranab Islam and Anand Kannappan and Douwe Kiela and Rebecca Qian and Nino Scherrer and Bertie Vidgen},
        year={2023},
        eprint={2311.11944},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
    """)

st.header(":material/question_answer: Ask a question")
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_area(
        "Your question about financial filings",
        placeholder="e.g., What was Apple's gross margin in FY2023, and how did it change from FY2022?",
        height=100,
        label_visibility="collapsed",
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.button(
        ":material/send: Ask",
        type="primary",
        use_container_width=True,
    )


with st.expander(":material/lightbulb: Example questions"):
    examples = [
        "What was the total revenue in FY2023?",
        "Gross margin of the company in FY2023 changed by how much compared to FY2022, and what reasons did management cite?",
        "What were the company's risk factors related to supply chain?",
        "Calculate the operating margin for FY2023.",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            question = ex
            submitted = True

if submitted and question.strip():
    with st.spinner(":material/search: Searching filings and generating answer..."):
        try:

            payload = {
                "question": question.strip(),
                "company": company or None,
                "fiscal_year": fiscal_year if fiscal_year else None,
                "include_trace": include_trace,
            }
            with httpx.Client(timeout=600.0) as client:
                resp = client.post(f"{api_url.rstrip('/')}/query", json=payload)
            if resp.status_code == 200:
                data = resp.json()
            else:
                st.error(
                    f":material/error: API error ({resp.status_code}): {resp.text}"
                )
                st.stop()
        except httpx.TimeoutException:
            st.error(
                ":material/hourglass_top: The backend is taking too long to respond "
                "or is still loading the model. Please try again later."
            )
            st.stop()
        except httpx.ConnectError as e:
            st.error(
                f":material/wifi_off: Could not connect to the API `{api_url}`: {e}"
            )
            st.stop()
        except httpx.HTTPError as e:
            st.error(f":material/error: HTTP error while calling the API: {e}")
            st.stop()
        except Exception as e:
            st.error(f":material/error: An unexpected error occurred: {e}")
            st.stop()

    status = data.get("status", "error")
    if status == "answer":
        st.success(":material/check_circle: Answer generated with evidence")
    elif status == "abstain":
        st.warning(":material/warning: Insufficient evidence — abstaining")
    else:
        st.error(":material/error: Failed to generate an answer")

    st.markdown("### :material/answer: Answer")
    model_name = data.get("model_name")
    if model_name:
        st.info(
            f"You're using **{model_name}** for this section. "
            "If you want to change it, please modify `MODEL_NAME` in `.env` "
            "or the default in `src/core/configs.py`."
        )
    st.markdown(data.get("answer", "No answer returned"))

    calculations = data.get("calculations", [])
    if calculations:
        with st.expander(
            ":material/calculate: Calculations",
            expanded=True,
        ):
            for calc in calculations:
                st.markdown(
                    f":material/functions: **Formula:** `{calc.get('formula', 'N/A')}`"
                )
                st.markdown(
                    f":material/trending_up: **Result:** `{calc.get('result', 'N/A')}`"
                )
                if calc.get("inputs"):
                    st.markdown(":material/input: **Inputs:**")
                    for inp in calc["inputs"]:
                        st.markdown(
                            f"- {inp.get('name','?')}: {inp.get('value','?')} {inp.get('unit','')} `[{inp.get('chunk_id','')}]`"
                        )
                st.divider()

    citations = data.get("citations", [])
    if citations:
        with st.expander(
            ":material/format_quote: Citations",
            expanded=True,
        ):
            for i, cite in enumerate(citations, 1):
                st.markdown(f":material/source: **[{i}] {cite.get('company','?')}**")
                st.markdown(
                    f":material/description: **Filing:** {cite.get('filing','?')}"
                )
                source_url = cite.get("source_url", "")
                if source_url:
                    st.markdown(f":material/link: [**Open document**]({source_url})")
                st.markdown(f":material/article: **Page:** {cite.get('page','?')}")
                st.markdown(
                    f":material/menu_book: **Section:** {cite.get('section','?')}"
                )
                st.markdown(
                    f":material/tag: **Chunk ID:** `{cite.get('chunk_id','?')}`"
                )
                snippet = cite.get("text_snippet", "")
                if snippet:
                    st.text_area(
                        "Evidence snippet",
                        value=snippet,
                        height=80,
                        disabled=True,
                        key=f"snippet_{i}",
                    )
                st.divider()

    trace = data.get("trace") or {}
    if status == "abstain":
        # An abstention hides which stage failed. Showing what retrieval found
        # separates "nothing was retrieved" from "the grader rejected it".
        with st.expander(
            ":material/troubleshoot: Why no answer? (retrieval diagnostics)",
            expanded=True,
        ):
            retrieval_log = trace.get("retrieval_log", [])
            if not retrieval_log:
                st.write("No retrieval attempt was recorded.")
            for attempt in retrieval_log:
                st.markdown(
                    f"**Hop {attempt.get('hop', '?')} / retry {attempt.get('retry', '?')}** "
                    f"- {attempt.get('chunks', 0)} chunks retrieved"
                )
                st.caption(attempt.get("query", ""))
                for item in attempt.get("top_chunks", []):
                    st.markdown(
                        f"- `{item.get('chunk_id','?')}` - {item.get('doc_id','?')} "
                        f"p.{item.get('page','?')} ({item.get('content_type','?')})"
                    )
            verdicts = trace.get("grader_verdicts", [])
            if verdicts:
                st.markdown("**Grader verdicts on the last attempt**")
                for verdict in verdicts:
                    st.markdown(
                        f"- `{verdict.get('chunk_id','?')}`: **{verdict.get('verdict','?')}** "
                        f"({verdict.get('score', 0)}) - {verdict.get('reason','')}"
                    )

    if trace and include_trace:
        with st.expander(
            ":material/account_tree: Full Trace",
            expanded=False,
        ):
            st.json(trace)

elif submitted and not question.strip():
    st.warning(":material/edit: Please enter a question.")
