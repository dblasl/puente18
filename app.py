
import io
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="PUENTE 18+", page_icon="🌉", layout="wide")

# ---------------------------
# Visual theme
# ---------------------------
st.markdown("""
<style>
:root { --navy:#0B1F33; --purple:#4F46E5; --green:#16A34A; --amber:#F59E0B; --red:#DC2626; --muted:#64748B; }
.block-container {padding-top: 1.0rem; padding-bottom: 2rem;}
.kpi {
    border: 1px solid #E2E8F0; border-radius: 16px; padding: 16px 18px;
    background: #FFFFFF; box-shadow: 0 6px 18px rgba(15,23,42,.06);
}
.kpi .label {font-size: .78rem;color:#64748B;margin-bottom:5px;}
.kpi .value {font-size: 1.65rem;font-weight: 750;color:#0B1F33;}
.kpi .sub {font-size:.72rem;color:#64748B;margin-top:4px;}
.badge {display:inline-block;padding:5px 10px;border-radius:999px;font-weight:700;font-size:.78rem;}
.badge-green {background:#DCFCE7;color:#166534}.badge-amber{background:#FEF3C7;color:#92400E}.badge-red{background:#FEE2E2;color:#991B1B}.badge-gray{background:#E2E8F0;color:#334155}
.section-title {font-size:1.02rem;font-weight:750;color:#0B1F33;margin:8px 0 10px;}
.small-muted {color:#64748B;font-size:.78rem;}
.patient-card {border:1px solid #E2E8F0;border-radius:14px;padding:14px;background:#fff;margin-bottom:8px;}
.metric-note {font-size:.75rem;color:#64748B;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

def load_demo_data():
    p = pd.read_csv(DATA / "pacientes_sinteticos.csv")
    f = pd.read_csv(DATA / "establecimientos_capacidad.csv")
    return p, f

def risk_badge(level):
    if level == "Alto":
        return '<span class="badge badge-red">🔴 Alto</span>'
    if level == "Moderado":
        return '<span class="badge badge-amber">🟡 Moderado</span>'
    return '<span class="badge badge-green">🟢 Bajo</span>'

def status_badge(status):
    if status == "Disponible":
        return '<span class="badge badge-green">🟢 Disponible</span>'
    if status == "Capacidad limitada":
        return '<span class="badge badge-amber">🟡 Limitada</span>'
    if status == "Sin capacidad reportada":
        return '<span class="badge badge-red">🔴 Sin capacidad</span>'
    return '<span class="badge badge-gray">⚪ Desactualizado</span>'

def calculate_rule_score(row):
    score = 0
    score += 25 if str(row["adult_receiver_identified"]).lower() in ["no","0","false"] else 0
    score += 20 if str(row["appointment_confirmed"]).lower() in ["no","0","false"] else 0
    score += 20 if float(row["readiness_score"]) < 50 else 0
    score += 15 if str(row["summary_complete"]).lower() in ["no","0","false"] else 0
    score += 10 if float(row["no_shows_last_12m"]) >= 2 else 0
    score += 10 if float(row["months_to_18"]) <= 2 else 0
    score += 8 if float(row["complexity"] == "Alta") else 0
    return int(min(score,100))

def apply_rules(df):
    out = df.copy()
    out["rule_score_live"] = out.apply(calculate_rule_score, axis=1)
    out["rule_level_live"] = np.select(
        [out["rule_score_live"] >= 65, out["rule_score_live"] >= 38],
        ["Alto","Moderado"],
        default="Bajo"
    )
    return out

def build_synthetic_training_data(n=1600, seed=18):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "age_years": rng.choice([16,17,18], n, p=[.30,.50,.20]),
        "months_to_18": rng.uniform(0,24,n),
        "complexity": rng.choice(["Baja","Media","Alta"], n, p=[.10,.45,.45]),
        "readiness_score": rng.integers(20,97,n),
        "caregiver_support": rng.integers(25,100,n),
        "no_shows_last_12m": rng.poisson(0.8,n),
        "adult_receiver_identified": rng.choice([0,1], n, p=[.48,.52]),
        "appointment_confirmed": rng.choice([0,1], n, p=[.42,.58]),
        "summary_complete": rng.choice([0,1], n, p=[.28,.72]),
    })
    # Synthetic demonstration target: continuity interruption (NOT clinical data).
    lin = (
        0.08*(d["complexity"]=="Alta").astype(int)
        +0.95*(1-d["adult_receiver_identified"])
        +0.78*(1-d["appointment_confirmed"])
        +0.56*(1-d["summary_complete"])
        +0.055*(50-d["readiness_score"]).clip(lower=0)
        +0.055*(d["no_shows_last_12m"].clip(upper=4))
        +0.55*(d["months_to_18"]<2).astype(int)
        +rng.normal(0,0.45,n)
    )
    prob = 1/(1+np.exp(-lin+1.1))
    d["transition_interruption"] = rng.binomial(1, np.clip(prob,.04,.96))
    return d

@st.cache_resource
def train_demo_model():
    train = build_synthetic_training_data()
    X = train.drop(columns=["transition_interruption"])
    y = train["transition_interruption"]
    cat = ["complexity"]
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])
    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.25,random_state=18,stratify=y)
    model.fit(Xtr,ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:,1])
    return model, auc, train

def model_predict(df, model):
    x = df[[
        "age_years","months_to_18","complexity","readiness_score","caregiver_support",
        "no_shows_last_12m"
    ]].copy()
    x["adult_receiver_identified"] = (df["adult_receiver_identified"].str.lower()=="sí").astype(int)
    x["appointment_confirmed"] = (df["appointment_confirmed"].str.lower()=="sí").astype(int)
    x["summary_complete"] = (df["summary_complete"].str.lower()=="sí").astype(int)
    p = model.predict_proba(x)[:,1]
    return (p*100).round(0).astype(int)

def render_kpi(label, value, sub, accent="#4F46E5"):
    st.markdown(
        f"""<div class="kpi" style="border-top:4px solid {accent}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div></div>""",
        unsafe_allow_html=True
    )

if "patients" not in st.session_state:
    demo_p, demo_f = load_demo_data()
    st.session_state.patients = apply_rules(demo_p)
    st.session_state.facilities = demo_f

patients = st.session_state.patients
facilities = st.session_state.facilities

# ---------------------------
# Sidebar / role
# ---------------------------
with st.sidebar:
    st.markdown("## 🌉 PUENTE 18+")
    st.caption("Prototipo de coordinación de transición pediátrico-adulta")
    role = st.selectbox("Tipo de usuario", [
        "Médico/a", "Enfermería", "Coordinación de transición", "DIRIS / DIRESA", "Tutor / joven"
    ])
    st.divider()
    page = st.radio("Módulo", [
        "Dashboard",
        "Ficha de transición",
        "Buscar derivación",
        "Capacidad de red",
        "Autoaprendizaje",
        "Modelo predictivo",
        "Carga de datos"
    ])
    st.divider()
    st.caption("🧪 DEMO — datos 100% sintéticos")
    st.caption("No usar historias clínicas reales.")

# ---------------------------
# Dashboard
# ---------------------------
if page == "Dashboard":
    st.title("Centro de Gestión de Transición")
    st.caption(f"Vista: {role} · Última actualización demo: 15/08/2026 09:30")
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: render_kpi("Pacientes en transición", len(patients), "base demo", "#4F46E5")
    with c2: render_kpi("Próximos a 18 años", int((patients["months_to_18"]<=3).sum()), "≤ 3 meses", "#F59E0B")
    with c3: render_kpi("Alta complejidad", int((patients["complexity"]=="Alta").sum()), "seguimiento continuo", "#DC2626")
    with c4: render_kpi("Sin receptor", int((patients["adult_receiver_identified"]=="No").sum()), "requieren gestión", "#DC2626")
    with c5: render_kpi("Sin cita", int((patients["appointment_confirmed"]=="No").sum()), "pendientes", "#F59E0B")

    st.markdown("### Estado de la transición")
    left, mid, right = st.columns([1.1,1.2,1.2])
    with left:
        lvl = patients["rule_level_live"].value_counts().reindex(["Bajo","Moderado","Alto"]).fillna(0)
        fig = px.pie(values=lvl.values, names=["🟢 Bajo","🟡 Moderado","🔴 Alto"], hole=.62)
        fig.update_layout(height=290, margin=dict(l=0,r=0,t=10,b=0), legend=dict(orientation="h",y=-.05))
        st.plotly_chart(fig, use_container_width=True)
    with mid:
        funnel = pd.DataFrame({
            "Etapa":["Identificados","Evaluados","Preparados","Con plan","Referidos","Cita confirmada","Primera atención"],
            "Pacientes":[len(patients), len(patients)-2, int(len(patients)*.82), int(len(patients)*.71),
                         int((patients["referral_sent"]=="Sí").sum()), int((patients["appointment_confirmed"]=="Sí").sum()), int((patients["appointment_confirmed"]=="Sí").sum())-4]
        })
        fig2 = px.funnel(funnel, y="Etapa", x="Pacientes")
        fig2.update_layout(height=290, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig2, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Alertas recientes</div>', unsafe_allow_html=True)
        alerts = [
            ("🔴", "Pacientes próximos a 18 años", f"{int((patients['months_to_18']<=2).sum())} requieren revisión"),
            ("🟡", "Sin cita confirmada", f"{int((patients['appointment_confirmed']=='No').sum())} pendientes"),
            ("🔴", "Sin receptor adulto", f"{int((patients['adult_receiver_identified']=='No').sum())} casos"),
            ("⚪", "Capacidad desactualizada", "1 registro supera 48 h")
        ]
        for icon, title, txt in alerts:
            st.markdown(f"**{icon} {title}**<br><span class='small-muted'>{txt}</span>", unsafe_allow_html=True)
            st.markdown("---")

    st.markdown("### Pacientes prioritarios")
    top = patients.sort_values(["rule_score_live","months_to_18"], ascending=[False, True]).head(8).copy()
    cols = ["patient_id","name","age_years","months_to_18","complexity","rule_score_live","rule_level_live","specialty_required"]
    st.dataframe(top[cols].rename(columns={
        "patient_id":"ID","name":"Paciente","age_years":"Edad","months_to_18":"Meses para 18",
        "complexity":"Complejidad","rule_score_live":"Riesgo","rule_level_live":"Semáforo","specialty_required":"Especialidad"
    }), use_container_width=True, hide_index=True)

# ---------------------------
# Transition profile
# ---------------------------
elif page == "Ficha de transición":
    st.title("Ficha de transición 360°")
    st.caption("La ficha está diseñada para coordinar la transición, no para reemplazar la historia clínica electrónica.")
    patient_id = st.selectbox("Seleccionar paciente", patients["patient_id"].tolist())
    row = patients.loc[patients["patient_id"]==patient_id].iloc[0]

    st.markdown(f"### {row['name']} · `{row['patient_id']}`")
    st.caption("DEMO — identidad, clínica y eventos completamente sintéticos")
    a,b,c,d = st.columns(4)
    with a: render_kpi("Edad", f"{int(row['age_years'])} años {int(row['age_months_total']%12)} meses", f"{row['months_to_18']:.1f} meses para 18", "#4F46E5")
    with b: render_kpi("Complejidad", row["complexity"], row["condition"], "#DC2626" if row["complexity"]=="Alta" else "#F59E0B")
    with c: render_kpi("Readiness", f"{row['readiness_score']}%", "preparación", "#16A34A")
    with d: render_kpi("Riesgo", f"{int(row['rule_score_live'])}/100", row["rule_level_live"], "#DC2626" if row["rule_level_live"]=="Alto" else "#F59E0B")

    st.markdown("#### Información clínica relevante para transición")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.markdown("**Condición principal**")
        st.info(row["condition"])
        st.markdown("**Especialidad adulta requerida**")
        st.info(row["specialty_required"])
    with col2:
        st.markdown("**Medicaciones**")
        st.info(row["current_medications"])
        st.markdown("**Alergias**")
        st.info(row["allergies"])
    with col3:
        st.markdown("**Cuidador principal**")
        st.info(row["caregiver"])
        st.markdown("**Última atención**")
        st.info(row["last_visit"])

    st.markdown("#### Estado de transición")
    steps = [
        ("Identificación", True),
        ("Evaluación", True),
        ("Preparación", row["readiness_score"]>=60),
        ("Plan", row["adult_receiver_identified"]=="Sí"),
        ("Referencia", row["referral_sent"]=="Sí"),
        ("Cita", row["appointment_confirmed"]=="Sí"),
        ("Primera atención", row["appointment_confirmed"]=="Sí" and row["adult_receiver_identified"]=="Sí"),
    ]
    for label, ok in steps:
        st.write(("✅" if ok else "⬜") + " " + label)

    st.markdown("#### ¿Por qué aparece esta alerta?")
    factors = []
    if row["adult_receiver_identified"]=="No": factors.append("No hay receptor adulto identificado.")
    if row["appointment_confirmed"]=="No": factors.append("No hay cita adulta confirmada.")
    if row["readiness_score"]<50: factors.append("Nivel de preparación inferior a 50%.")
    if row["summary_complete"]=="No": factors.append("Resumen clínico de transición incompleto.")
    if row["no_shows_last_12m"]>=2: factors.append("Historial reciente de inasistencias.")
    if row["months_to_18"]<=2: factors.append("Ventana crítica: quedan 2 meses o menos para los 18 años.")
    if not factors: factors.append("No se identificaron alertas principales en las reglas de demo.")
    for x in factors:
        st.markdown(f"- {x}")

# ---------------------------
# Referral navigator
# ---------------------------
elif page == "Buscar derivación":
    st.title("🔎 Navegador de derivación")
    st.caption("Motor de compatibilidad basado en reglas de demostración y capacidad reportada.")
    c1,c2,c3 = st.columns(3)
    with c1:
        condition = st.selectbox("Condición", sorted(patients["condition"].unique()))
    with c2:
        specialty = st.selectbox("Especialidad adulta", sorted(patients["specialty_required"].unique()))
    with c3:
        complexity = st.selectbox("Complejidad", ["Alta","Media","Baja"], index=0)

    c4,c5 = st.columns([1,1])
    with c4:
        diris = st.selectbox("Red / DIRIS", ["Todas"] + sorted(facilities["diris"].unique()))
    with c5:
        only_capacity = st.checkbox("Mostrar solo capacidad disponible", value=False)

    if st.button("Buscar establecimientos compatibles", type="primary"):
        f = facilities[
            (facilities["specialty"]==specialty) &
            (facilities["adult_service"]=="Sí")
        ].copy()
        def match_score(r):
            s = 45
            if r["complexity_supported"] == complexity or (complexity=="Alta" and r["complexity_supported"]=="Alta"): s += 20
            s += 25 if r["capacity_status"]=="Disponible" else (10 if r["capacity_status"]=="Capacidad limitada" else 0)
            s += max(0, 10 - min(int(r["distance_km_demo"]),10))
            return min(99, s)
        f["compatibility"] = f.apply(match_score, axis=1)
        if diris != "Todas":
            f = f[f["diris"]==diris]
        if only_capacity:
            f = f[f["capacity_status"]=="Disponible"]
        f = f.sort_values("compatibility", ascending=False)
        if f.empty:
            st.warning("No hay opciones reportadas con los filtros actuales.")
        else:
            st.success(f"{len(f)} opciones compatibles encontradas en datos sintéticos.")
            for _, r in f.iterrows():
                left,right = st.columns([3,1])
                with left:
                    st.markdown(f"### {r['facility_name']}")
                    st.write(f"**{r['specialty']} · {r['complexity_supported']} · {r['diris']}**")
                    st.write(f"Capacidad reportada: **{r['reported_capacity']}** · Última actualización: **{r['last_updated']}**")
                with right:
                    st.metric("Compatibilidad", f"{int(r['compatibility'])}%")
                    st.markdown(status_badge(r["capacity_status"]), unsafe_allow_html=True)
                st.divider()

# ---------------------------
# Network capacity
# ---------------------------
elif page == "Capacidad de red":
    st.title("🏥 Capacidad reportada de la red")
    st.caption("Esta vista simula información compartida por establecimientos / DIRIS / DIRESA.")
    spec_summary = facilities.groupby(["specialty","capacity_status"]).size().reset_index(name="n")
    fig = px.bar(spec_summary, x="specialty", y="n", color="capacity_status", barmode="stack",
                 category_orders={"capacity_status":["Disponible","Capacidad limitada","Sin capacidad reportada"]})
    fig.update_layout(height=400, margin=dict(l=0,r=0,t=10,b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    heat = facilities.pivot_table(index="specialty", columns="diris", values="reported_capacity", aggfunc="sum", fill_value=0)
    fig2 = px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale="YlGn")
    fig2.update_layout(height=420, margin=dict(l=0,r=0,t=20,b=10))
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        facilities[["facility_name","diris","specialty","complexity_supported","reported_capacity","capacity_status","last_updated"]]
        .sort_values(["specialty","reported_capacity"], ascending=[True,False]),
        use_container_width=True, hide_index=True
    )

# ---------------------------
# Learning module
# ---------------------------
elif page == "Autoaprendizaje":
    st.title("🧠 Mi transición: módulo inicial")
    st.caption("Microaprendizaje para jóvenes y tutores. Contenido educativo de demo; no sustituye orientación clínica.")
    target = st.radio("¿Para quién es este módulo?", ["Joven", "Tutor / cuidador"], horizontal=True)

    quiz_data = json.loads((DATA/"quiz.json").read_text(encoding="utf-8"))
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "quiz_result" not in st.session_state:
        st.session_state.quiz_result = None

    st.info("Objetivo: fortalecer progresivamente conocimientos y habilidades prácticas para una transición segura.")

    for q in quiz_data:
        choice = st.radio(
            f"{q['id']}. {q['question']}",
            q["options"], key=f"q_{q['id']}"
        )
        st.session_state.quiz_answers[q["id"]] = q["options"].index(choice)

    if st.button("Calificar módulo", type="primary"):
        correct = sum(st.session_state.quiz_answers.get(q["id"]) == q["answer"] for q in quiz_data)
        score = round(correct/len(quiz_data)*100)
        st.session_state.quiz_result = (correct, score)

    if st.session_state.quiz_result:
        correct, score = st.session_state.quiz_result
        st.success(f"Resultado: {score}% · {correct}/{len(quiz_data)} respuestas correctas")
        if score >= 80:
            st.balloons()
            st.markdown("### 🟢 Buen nivel de preparación")
            st.write("El módulo sugiere continuar practicando gestión de citas, comunicación y toma de decisiones.")
        elif score >= 60:
            st.markdown("### 🟡 En desarrollo")
            st.write("Conviene reforzar algunas habilidades antes de la transferencia.")
        else:
            st.markdown("### 🔴 Necesita acompañamiento")
            st.write("El resultado sugiere priorizar educación y acompañamiento del equipo.")
        st.caption("La puntuación es educativa y no constituye un diagnóstico ni una evaluación clínica validada.")

# ---------------------------
# Predictive model
# ---------------------------
elif page == "Modelo predictivo":
    st.title("📈 Modelo predictivo — demostración técnica")
    st.warning("Este modelo se entrena exclusivamente con datos sintéticos generados para la demo. No está validado clínicamente y no debe usarse para decisiones reales.")
    model, auc, train = train_demo_model()
    st.metric("AUC de validación interna (sintético)", f"{auc:.2f}")
    pred = model_predict(patients, model)
    view = patients[["patient_id","name","age_years","complexity","readiness_score","adult_receiver_identified","appointment_confirmed"]].copy()
    view["ml_risk_probability"] = pred
    view["ml_risk_level"] = np.select([pred>=65,pred>=38],["Alto","Moderado"],default="Bajo")
    st.dataframe(view.sort_values("ml_risk_probability", ascending=False), use_container_width=True, hide_index=True)
    st.markdown("### ¿Cómo interpretar?")
    st.write("El modelo de demostración estima probabilidad de interrupción de continuidad a partir de variables operativas y de preparación. En una implementación real, se debería definir el outcome con especialistas, entrenar con datos históricos anonimizados y validar externamente.")

# ---------------------------
# Data upload
# ---------------------------
elif page == "Carga de datos":
    st.title("📤 Cargar registros sintéticos")
    st.caption("Puedes subir Excel o CSV. El prototipo intenta mapear las columnas esperadas.")
    st.markdown("**Columnas mínimas recomendadas:** patient_id, name, age_years, months_to_18, condition, specialty_required, complexity, readiness_score, caregiver_support, no_shows_last_12m, summary_complete, adult_receiver_identified, appointment_confirmed, referral_sent")
    file = st.file_uploader("Sube `pacientes.xlsx` o `pacientes.csv`", type=["xlsx","xls","csv"])
    if file is not None:
        try:
            if file.name.lower().endswith(".csv"):
                new_df = pd.read_csv(file)
            else:
                new_df = pd.read_excel(file)
            required = {"patient_id","name","age_years","months_to_18","condition","specialty_required","complexity","readiness_score","caregiver_support","no_shows_last_12m","summary_complete","adult_receiver_identified","appointment_confirmed","referral_sent"}
            missing = required - set(new_df.columns)
            if missing:
                st.error("Faltan columnas: " + ", ".join(sorted(missing)))
            else:
                st.session_state.patients = apply_rules(new_df)
                patients = st.session_state.patients
                st.success(f"Se cargaron {len(new_df)} registros sintéticos.")
                st.dataframe(new_df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
    else:
        st.download_button(
            "Descargar plantilla de pacientes",
            data=(DATA/"pacientes_sinteticos.csv").read_bytes(),
            file_name="plantilla_pacientes_sinteticos.csv",
            mime="text/csv"
        )
