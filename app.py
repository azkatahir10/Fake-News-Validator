import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"

st.title("📰 Fake News Validator")

user_query = st.text_input("Enter a news headline or article:")

if st.button("Check"):
    if user_query.strip() == "":
        st.warning("⚠️ Please enter some text to validate.")
    else:
        try:
            res = requests.post(API_URL, json={"query": user_query})
            if res.status_code == 200:
                data = res.json()
                st.subheader("Verdict:")
                st.write(f"**{data['verdict']}** (Confidence: {data['confidence']}%)")

                st.subheader("Supporting Articles:")
                for art in data["articles"]:
                    st.markdown(
                        f"- **{'True' if art['label']==1 else 'Fake'}** | {art['title']}"
                    )
            else:
                st.error("❌ Backend error: " + res.text)
        except Exception as e:
            st.error(f"🚨 Could not connect to backend: {e}")
