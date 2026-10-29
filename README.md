---
title: IPC Navigator
emoji: ⚖️
colorFrom: indigo
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

⚖️ IPC Navigator

An AI-powered Streamlit app to explore the Indian Penal Code, 1860.

1. Start the app
bash
pip install -r requirements.txt
streamlit run app.py

It opens in your browser at http://localhost:8501. Keep ipc_act.txt (or ipc.txt) in the same folder as app.py.

2. Use the pages (left sidebar)

Code Lookup: Type a section number like 302 or 420. You get the section text, its punishment, bail and cognizable tags, the BNS equivalent and related sections. You can also type a range like 299-304, and typos like 42O are corrected.

Case Study: Describe an incident in plain words, for example: “A man threatened a shopkeeper and took money by force.” The app ranks the IPC sections that apply (extortion, robbery and so on). You can filter by chapter or bail type, and you can speak your description instead of typing it.

Compare: Pick two sections, such as 299 and 300, and see how they differ.

Browse: Open a chapter and read its sections one by one.

Chat: Ask follow-up questions like “explain section 420 in simple words”. This page needs a Groq key.

Analytics: See what people searched and how well each search method performed.

Who can use it
Law students: quick section lookup and comparison
Researchers: finding sections that match a scenario
General users: understanding which law applies to a situation