import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from datetime import datetime
from engine import demo_traffic, analyze_traffic, train_demo_model, evaluate_demo_model, REQUIRED_FEATURES

st.set_page_config(page_title="CyberShield AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family: Inter, sans-serif;}
.stApp {background: linear-gradient(135deg,#081120 0%,#0b1428 50%,#101a32 100%); color:#eaf0ff;}
[data-testid="stSidebar"] {background: #0b1222; border-right: 1px solid #22304c;}
[data-testid="stMetric"] {background:linear-gradient(135deg,#111f38,#142641); padding:18px 18px; border:1px solid #263b5d; border-radius:16px;}
[data-testid="stMetricLabel"] {color:#9db0d0;}
[data-testid="stMetricValue"] {color:#f4f7ff;}
h1,h2,h3 {color:#f5f7ff !important;}
p, label, .stCaption {color:#bdc9df;}
div.stButton > button {border-radius:10px; border:1px solid #3c65a4; background:linear-gradient(90deg,#2563eb,#7c3aed); color:white; font-weight:650;}
div.stButton > button:hover {border-color:#8ab4ff; color:white;}
.block-container {padding-top:1.8rem; padding-bottom:3rem;}
.cs-banner {padding:24px 28px; border-radius:20px; border:1px solid #29436b; background:linear-gradient(110deg,rgba(37,99,235,.24),rgba(124,58,237,.16)); margin-bottom:18px;}
.cs-pill {display:inline-block;padding:5px 10px;border-radius:20px;background:#163d3c;color:#75f0cf;font-size:12px;font-weight:700;letter-spacing:.5px;}
.cs-small {color:#8fa5c8;font-size:13px;}
.cs-card {border:1px solid #263b5d;background:#101e34;border-radius:15px;padding:17px;}
hr {border-color:#263b5d;}
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "traffic" not in st.session_state:
    st.session_state.traffic = demo_traffic(240, seed=42)
if "results" not in st.session_state:
    st.session_state.results = None
if "model" not in st.session_state:
    st.session_state.model = None
if "report" not in st.session_state:
    st.session_state.report = None

def go(page):
    st.session_state.page = page

with st.sidebar:
    st.markdown("## 🛡️ CyberShield **AI**")
    st.markdown("<span class='cs-small'>NETWORK THREAT DEFENSE</span>", unsafe_allow_html=True)
    st.divider()
    pages = {
        "Dashboard":"▦  Dashboard",
        "Traffic Analysis":"⌁  Traffic Analysis",
        "Threat Intelligence":"◈  Threat Intelligence",
        "Incident Reports":"▤  Incident Reports",
        "Model Insights":"◎  Model Insights",
        "About Project":"ⓘ  About Project"
    }
    for p, label in pages.items():
        if st.button(label, key="nav_"+p, use_container_width=True, type="primary" if st.session_state.page == p else "secondary"):
            go(p)
            st.rerun()
    st.divider()
    st.markdown("<div class='cs-card'><span class='cs-pill'>● DEMO ENVIRONMENT</span><br><br><b>System status</b><br><span class='cs-small'>Analysis runs locally in this app.<br>No live network is being monitored.</span></div>", unsafe_allow_html=True)
    st.caption("Student prototype • v1.0")

page = st.session_state.page
st.markdown(f"<div class='cs-small'>CYBERSHIELD AI  /  {page.upper()}</div>", unsafe_allow_html=True)

if page == "Dashboard":
    st.markdown("""
    <div class="cs-banner">
      <span class="cs-pill">AI-POWERED NETWORK DEFENSE</span>
      <h1 style="margin:12px 0 6px 0;font-size:34px;">Security overview</h1>
      <p style="font-size:16px;margin:0;color:#b9c9e6;">Turn network-flow records into actionable security insights.</p>
    </div>""", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    results = st.session_state.results
    if results is None:
        df = st.session_state.traffic.copy()
        # Dashboard demo preview has illustrative labels from the sample generator.
        total = len(df)
        malicious = int(df["demo_label"].isin(["DDoS","PortScan","BruteForce","Botnet"]).sum()) if "demo_label" in df else 0
        high = max(1, int(malicious*.28))
        benign = total-malicious
    else:
        df = results.copy()
        total = len(df)
        malicious = int((df["prediction"] != "Benign").sum())
        benign = total-malicious
        high = int((df["severity"] == "High").sum())
    c1.metric("Records analyzed", f"{total:,}", "Traffic flows")
    c2.metric("Potential threats", f"{malicious:,}", f"{(malicious/max(total,1))*100:.1f}% of records")
    c3.metric("Benign records", f"{benign:,}", "Model classified")
    c4.metric("High-priority alerts", f"{high:,}", "Review recommended")
    st.markdown("### Threat landscape")
    left,right = st.columns([1.25,1])
    with left:
        if results is None:
            labels = st.session_state.traffic["demo_label"].value_counts().rename_axis("Category").reset_index(name="Records")
            labels = labels.rename(columns={"demo_label":"Category"})
            title = "Illustrative sample traffic mix"
        else:
            labels = results["prediction"].value_counts().rename_axis("Category").reset_index(name="Records")
            title = "Model prediction distribution"
        fig = px.bar(labels, x="Category", y="Records", color="Category",
                     color_discrete_sequence=["#4f8cff","#37d6b0","#ffb454","#ff647c","#a78bfa"],
                     template="plotly_dark", title=title)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False,margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        sev_counts = pd.DataFrame({"Severity":["Low","Medium","High"],"Alerts":[max(0,malicious-high), max(0,high//2), high]})
        fig2 = px.pie(sev_counts, names="Severity", values="Alerts", hole=.62,
                      color="Severity", color_discrete_map={"Low":"#37d6b0","Medium":"#ffb454","High":"#ff647c"},
                      template="plotly_dark", title="Priority distribution (illustrative)")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("### Recent activity")
    if results is not None:
        view = results[["timestamp","source_ip","destination_ip","prediction","severity","reason"]].head(8).copy()
        view.columns = ["Timestamp","Source IP","Destination IP","Prediction","Severity","Why flagged"]
    else:
        d = st.session_state.traffic.head(8).copy()
        view = pd.DataFrame({
            "Timestamp": d["timestamp"], "Source IP":d["source_ip"],"Destination IP":d["destination_ip"],
            "Prediction":d["demo_label"],"Severity":d["demo_severity"],"Why flagged":d["demo_reason"]
        })
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.info("Demo note: the initial dashboard uses generated sample records. Run Traffic Analysis to test the demo classifier or upload a compatible dataset.")

elif page == "Traffic Analysis":
    st.markdown("# Traffic analysis")
    st.write("Load illustrative sample traffic or upload a CSV of network-flow features. The prototype does not capture live packets.")
    a,b = st.columns([1,1])
    with a:
        if st.button("↻ Load sample traffic", use_container_width=True):
            st.session_state.traffic = demo_traffic(240, seed=random.randint(1,9999))
            st.session_state.results = None
            st.success("Sample traffic loaded.")
    with b:
        uploaded = st.file_uploader("Upload network-flow CSV", type=["csv"])
    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
            st.session_state.traffic = incoming
            st.session_state.results = None
            st.success(f"Loaded {len(incoming):,} rows.")
        except Exception as e:
            st.error(f"Could not read this CSV: {e}")
    st.markdown("### Data preview")
    st.dataframe(st.session_state.traffic.head(12), use_container_width=True, hide_index=True)
    st.caption("For real dataset inference, the CSV must contain the model's required numerical feature columns. Demo-generated records are clearly marked as illustrative.")
    with st.expander("Required input features for the demo model"):
        st.code(", ".join(REQUIRED_FEATURES))
        st.write("The demo model is trained on synthetic teaching data. It is useful to demonstrate the workflow, not to claim real-world detection performance.")
    if st.button("⚡ Analyze traffic", type="primary", use_container_width=True):
        with st.spinner("Preparing model and analyzing records..."):
            try:
                model = train_demo_model()
                result = analyze_traffic(st.session_state.traffic, model)
                st.session_state.model = model
                st.session_state.results = result
                st.success(f"Analysis complete: {len(result)} records processed.")
            except Exception as e:
                st.error(str(e))
    if st.session_state.results is not None:
        res = st.session_state.results
        st.markdown("### Analysis results")
        m1,m2,m3 = st.columns(3)
        m1.metric("Potential threats", int((res.prediction!="Benign").sum()))
        m2.metric("High severity", int((res.severity=="High").sum()))
        m3.metric("Benign", int((res.prediction=="Benign").sum()))
        st.dataframe(res[["timestamp","source_ip","destination_ip","prediction","severity","confidence","reason"]], use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download analyzed CSV", res.to_csv(index=False).encode("utf-8"), "cybershield_analysis.csv", "text/csv", use_container_width=True)

elif page == "Threat Intelligence":
    st.markdown("# Threat intelligence")
    st.write("Filter and inspect records flagged by the latest analysis.")
    res = st.session_state.results
    if res is None:
        st.warning("No model analysis is available yet. Go to Traffic Analysis and click **Analyze traffic**.")
    else:
        col1,col2,col3 = st.columns(3)
        severity = col1.multiselect("Severity", ["High","Medium","Low"], default=["High","Medium"])
        category = col2.multiselect("Prediction", sorted(res["prediction"].unique().tolist()), default=sorted(res["prediction"].unique().tolist()))
        search = col3.text_input("Search source/destination IP")
        filtered = res[res["severity"].isin(severity) & res["prediction"].isin(category)]
        if search:
            filtered = filtered[filtered["source_ip"].astype(str).str.contains(search, case=False) | filtered["destination_ip"].astype(str).str.contains(search, case=False)]
        st.dataframe(filtered[["timestamp","source_ip","destination_ip","protocol","prediction","severity","confidence","reason"]], use_container_width=True, hide_index=True)
        if not filtered.empty:
            st.markdown("### Investigate a record")
            idx = st.selectbox("Choose record", filtered.index.tolist(), format_func=lambda i: f"#{i} • {filtered.loc[i,'prediction']} • {filtered.loc[i,'source_ip']}")
            row = filtered.loc[idx]
            x,y = st.columns(2)
            with x:
                st.markdown("#### Alert details")
                st.write(f"**Prediction:** {row['prediction']}")
                st.write(f"**Severity:** {row['severity']}")
                st.write(f"**Confidence:** {row['confidence']:.1%}")
                st.write(f"**Reason:** {row['reason']}")
            with y:
                st.markdown("#### Flow features")
                feature_cols = [f for f in REQUIRED_FEATURES if f in row.index]
                st.json({f: float(row[f]) for f in feature_cols})
        else:
            st.info("No records match the selected filters.")

elif page == "Incident Reports":
    st.markdown("# Incident reports")
    res = st.session_state.results
    if res is None:
        st.warning("Run an analysis first to create an incident report.")
    else:
        threats = res[res.prediction != "Benign"].copy()
        high = int((threats.severity=="High").sum())
        medium = int((threats.severity=="Medium").sum())
        low = int((threats.severity=="Low").sum())
        st.markdown(f"""<div class="cs-banner"><span class="cs-pill">GENERATED REPORT</span>
        <h2>Network Threat Analysis Summary</h2>
        <p>Generated: {datetime.now().strftime('%d %b %Y, %H:%M')} · Scope: {len(res)} analyzed flow records</p></div>""", unsafe_allow_html=True)
        a,b,c,d = st.columns(4)
        a.metric("Total records",len(res))
        b.metric("Potential threats",len(threats))
        c.metric("High",high)
        d.metric("Medium",medium)
        st.markdown("### Key findings")
        if threats.empty:
            st.success("No records were classified as potential threats in this analysis.")
        else:
            st.write(f"- The model flagged **{len(threats)}** records as potentially malicious.")
            st.write(f"- Severity distribution: **{high} High**, **{medium} Medium**, **{low} Low**.")
            st.write("- Findings require analyst verification; predictions are not proof of compromise.")
            st.markdown("### Top flagged records")
            st.dataframe(threats.sort_values("severity", key=lambda s:s.map({"High":3,"Medium":2,"Low":1}), ascending=False)[["source_ip","destination_ip","prediction","severity","reason"]].head(15), use_container_width=True, hide_index=True)
        report = [
            "CYBERSHIELD AI — NETWORK THREAT ANALYSIS SUMMARY",
            f"Generated: {datetime.now().isoformat(timespec='minutes')}",
            f"Records analyzed: {len(res)}",
            f"Potentially malicious records: {len(threats)}",
            f"High severity: {high}; Medium severity: {medium}; Low severity: {low}",
            "",
            "Important limitation: this student prototype uses a synthetic teaching model unless separately trained and evaluated on a real, documented network intrusion dataset. Findings require analyst verification."
        ]
        st.download_button("⬇️ Export text report", "\n".join(report), "incident_report.txt", "text/plain", use_container_width=True)

elif page == "Model Insights":
    st.markdown("# Model insights")
    st.write("Understand how the classifier works and inspect a demonstration evaluation.")
    st.markdown("""
    <div class="cs-card">
    <h3>Random Forest classifier</h3>
    <p>A Random Forest combines predictions from multiple decision trees. Each tree learns patterns from training examples, and the forest combines their outputs.</p>
    <p><b>Why this model?</b> It is a useful baseline for tabular data, handles nonlinear relationships, and is relatively easy to explain in a student viva.</p>
    </div><br>
    """, unsafe_allow_html=True)
    st.warning("The scores below are calculated on synthetic teaching data only. They are not evidence of real-world cybersecurity performance.")
    if st.button("Run demo evaluation", type="primary"):
        with st.spinner("Training and evaluating the demonstration model..."):
            metrics, cm, labels = evaluate_demo_model()
            st.session_state.demo_metrics = metrics
            st.session_state.demo_cm = cm
            st.session_state.demo_labels = labels
    if "demo_metrics" in st.session_state:
        metrics = st.session_state.demo_metrics
        c1,c2,c3 = st.columns(3)
        c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        c2.metric("Macro precision", f"{metrics['precision']:.1%}")
        c3.metric("Macro recall", f"{metrics['recall']:.1%}")
        st.write("Confusion matrix — rows are actual labels; columns are predicted labels.")
        cm_df = pd.DataFrame(st.session_state.demo_cm, index=st.session_state.demo_labels, columns=st.session_state.demo_labels)
        st.dataframe(cm_df, use_container_width=True)
    st.markdown("### Pipeline")
    st.code("CSV flow records → validate features → preprocessing → Random Forest → predicted class → severity rule → dashboard", language="text")
    st.markdown("### Key limitations")
    st.write("- Synthetic data is for demonstrating the application flow only.")
    st.write("- Real performance must be measured on a documented dataset and held-out test data.")
    st.write("- Dataset drift, class imbalance, false positives, and false negatives can affect results.")
    st.write("- This app does not block traffic or monitor a live network.")

elif page == "About Project":
    st.markdown("# About CyberShield AI")
    st.markdown("""
    <div class="cs-banner">
      <span class="cs-pill">PROJECT P30</span>
      <h1 style="font-size:32px;">AI Cybersecurity Threat Defense Agent</h1>
      <p>Assist security teams by analyzing network-flow records, surfacing potential threats, and prioritizing alerts for investigation.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("### Problem statement")
    st.write("Security teams need help identifying and prioritizing suspicious network activity among large volumes of traffic records.")
    st.markdown("### Proposed solution")
    st.write("A web-based prototype that accepts network-flow CSV data, applies a machine-learning classifier, assigns transparent severity levels, and generates investigation summaries.")
    st.markdown("### Technology stack")
    st.write("Python · Streamlit · Pandas · NumPy · scikit-learn · Plotly")
    st.markdown("### Current scope")
    st.write("This version is an interactive student prototype. It uses synthetic traffic for its built-in demo. It does not capture live network packets, block attacks, or claim validated real-world accuracy.")
    st.markdown("### Future work")
    st.write("Train and validate on CICIDS2017 or UNSW-NB15, add live flow ingestion in an authorized lab, persist incidents in a database, and add analyst feedback.")

st.markdown("---")
st.markdown("<div class='cs-small'>CYBERSHIELD AI • Educational prototype • Use only with data and networks you are authorized to analyze.</div>", unsafe_allow_html=True)
