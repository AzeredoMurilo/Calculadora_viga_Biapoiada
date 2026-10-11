import streamlit as st
import pandas as pd
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sympy.physics.continuum_mechanics.beam import Beam
import io
from fpdf import FPDF
import time
import unicodedata
import re
import os

# ==========================================
# INTEGRADORES E FORMATADORES
# ==========================================
def cumtrapz_np(y, x):
    return np.concatenate(([0], np.cumsum((y[:-1] + y[1:]) / 2 * np.diff(x))))

def clean_expr(expr, sig_figs=4):
    try:
        return expr.xreplace({n: sp.Float(n.evalf(sig_figs)) for n in expr.atoms(sp.Number)})
    except:
        return expr

def format_macaulay(eq_sympy):
    eq_str = str(eq_sympy)
    eq_str = re.sub(r'SingularityFunction\(([^,]+),\s*([^,]+),\s*([^)]+)\)', r'<\1 - \2>^\3', eq_str)
    return eq_str.replace("**", "^").replace("*", "")

# ==========================================
# CONFIGURAÇÃO E ESTADO DA SESSÃO
# ==========================================
st.set_page_config(page_title="Análise Estrutural", layout="wide")

if 'cargas_salvas' not in st.session_state:
    st.session_state.cargas_salvas = []
if 'resultados_analiticos' not in st.session_state:
    st.session_state.resultados_analiticos = None
if 'resultados_flambagem' not in st.session_state:
    st.session_state.resultados_flambagem = None
if 'last_sis' not in st.session_state:
    st.session_state.last_sis = "Métrico (SI)"

# ==========================================
# 1. MENU LATERAL: MÓDULO E UNIDADES
# ==========================================
st.sidebar.title("⚙️ Plataforma de Análise")

modulo_selecionado = st.sidebar.radio("Selecione o Módulo:", 
                                      ["1. Flexão Transversal (Vigas)", "2. Flambagem Axial (Colunas)"])

st.sidebar.markdown("---")
sis = st.sidebar.radio("Sistema de Unidades", ["Métrico (SI)", "Imperial (US)"], 
                       index=0 if st.session_state.last_sis == "Métrico (SI)" else 1)

if st.session_state.last_sis != sis:
    st.session_state.cargas_salvas = []
    st.session_state.resultados_analiticos = None
    st.session_state.resultados_flambagem = None
    st.session_state.last_sis = sis
    st.rerun()

if sis == "Métrico (SI)":
    u_L, u_F, u_M, u_q, u_sec, u_A, u_I, u_E, u_stress, u_defl = "m", "kN", "kN.m", "kN/m", "mm", "cm²", "cm⁴", "GPa", "MPa", "mm"
    f_pos, f_F, f_M, f_q, f_sec, f_I, f_E, f_Sy = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
else:
    u_L, u_F, u_M, u_q, u_sec, u_A, u_I, u_E, u_stress, u_defl = "ft", "kip", "kip.ft", "kip/ft", "in", "in²", "in⁴", "ksi", "ksi", "in"
    f_pos, f_F, f_M, f_q = 0.3048, 4.4482216, 1.3558179, 14.5939029
    f_sec, f_I, f_E, f_Sy = 25.4, 41.62314, 0.006894757, 6.894757

# ==========================================
# 2. MENU LATERAL: MATERIAL E SECÇÃO
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("Material")
E_input_ui = st.sidebar.number_input(f"Módulo de Elasticidade E ({u_E})", min_value=1.0, value=200.0 if sis=="Métrico (SI)" else 29000.0, step=10.0 if sis=="Métrico (SI)" else 1000.0)
Sy_input_ui = st.sidebar.number_input(f"Tensão de Escoamento Sy ({u_stress})", min_value=1.0, value=250.0 if sis=="Métrico (SI)" else 36.0, step=10.0 if sis=="Métrico (SI)" else 1.0)

st.sidebar.markdown("---")
st.sidebar.header("Secção Transversal")
tipo_secao = st.sidebar.selectbox("Geometria", 
    ["Barra Maciça Retangular", "Entrada Manual", "Barra Maciça Circular", "Perfil I / H", "Tubo Retangular / Quadrado", "Tubo Circular"]
)

H_val_ui, I_input_ui, B_ui, tw_ui, tf_ui, t_ui = 200.0, 800.0, 100.0, 5.0, 5.0, 5.0
I_min_input_ui, A_input_ui = 400.0, 50.0
H_inicial_ui, H_final_ui = 200.0, 200.0
variacao_h, alinhamento_cunha = "Constante", "Simétrica"

if tipo_secao == "Barra Maciça Retangular":
    if modulo_selecionado == "1. Flexão Transversal (Vigas)":
        variacao_h = st.sidebar.radio("Variação de Altura (H)", ["Constante", "Variável (Cunha)"])
    B_ui = st.sidebar.number_input(f"Base B ({u_sec})", min_value=0.01, value=50.0 if sis=="Métrico (SI)" else 2.0, step=10.0 if sis=="Métrico (SI)" else 0.5)
    
    if variacao_h == "Constante":
        H_val_ui = st.sidebar.number_input(f"Altura H ({u_sec})", min_value=0.01, value=100.0 if sis=="Métrico (SI)" else 4.0, step=10.0 if sis=="Métrico (SI)" else 0.5)
        H_inicial_ui, H_final_ui = H_val_ui, H_val_ui
    else:
        alinhamento_cunha = st.sidebar.radio("Alinhamento", ["Simétrica", "Reta no Topo (Nervura Inferior)", "Reta na Base (Nervura Superior)"])
        H_inicial_ui = st.sidebar.number_input(f"Altura Inicial H0 ({u_sec})", min_value=0.01, value=200.0 if sis=="Métrico (SI)" else 8.0, step=10.0 if sis=="Métrico (SI)" else 0.5)
        H_final_ui = st.sidebar.number_input(f"Altura Final H1 ({u_sec})", min_value=0.01, value=100.0 if sis=="Métrico (SI)" else 4.0, step=10.0 if sis=="Métrico (SI)" else 0.5)
        H_val_ui = H_inicial_ui 

elif tipo_secao == "Entrada Manual":
    I_input_ui = st.sidebar.number_input(f"Inércia Eixo Forte Iz ({u_I})", min_value=0.01, value=800.0 if sis=="Métrico (SI)" else 20.0, step=10.0)
    if modulo_selecionado == "2. Flambagem Axial (Colunas)":
        I_min_input_ui = st.sidebar.number_input(f"Inércia Eixo Fraco Iy ({u_I})", min_value=0.01, value=400.0 if sis=="Métrico (SI)" else 10.0, step=10.0)
        A_input_ui = st.sidebar.number_input(f"Área da Secção ({u_A})", min_value=0.01, value=50.0 if sis=="Métrico (SI)" else 8.0, step=5.0)
    H_val_ui = st.sidebar.number_input(f"Altura Total da Secção ({u_sec})", min_value=0.01, value=200.0 if sis=="Métrico (SI)" else 8.0, step=10.0)
    
elif tipo_secao == "Barra Maciça Circular":
    H_val_ui = st.sidebar.number_input(f"Diâmetro D ({u_sec})", min_value=0.01, value=50.0 if sis=="Métrico (SI)" else 2.0, step=10.0 if sis=="Métrico (SI)" else 0.5)
elif tipo_secao == "Perfil I / H":
    H_val_ui = st.sidebar.number_input(f"Altura Total H ({u_sec})", min_value=0.01, value=200.0 if sis=="Métrico (SI)" else 8.0, step=10.0 if sis=="Métrico (SI)" else 0.5)
    B_ui = st.sidebar.number_input(f"Largura da Mesa B ({u_sec})", min_value=0.01, value=100.0 if sis=="Métrico (SI)" else 4.0, step=10.0 if sis=="Métrico (SI)" else 0.5)
    tw_ui = st.sidebar.number_input(f"Espessura da Alma tw ({u_sec})", min_value=0.01, value=6.35 if sis=="Métrico (SI)" else 0.25, step=1.0 if sis=="Métrico (SI)" else 0.1)
    tf_ui = st.sidebar.number_input(f"Espessura da Mesa tf ({u_sec})", min_value=0.01, value=9.0 if sis=="Métrico (SI)" else 0.35, step=1.0 if sis=="Métrico (SI)" else 0.1)
elif tipo_secao == "Tubo Retangular / Quadrado":
    H_val_ui = st.sidebar.number_input(f"Altura Externa H ({u_sec})", min_value=0.01, value=100.0 if sis=="Métrico (SI)" else 4.0, step=10.0 if sis=="Métrico (SI)" else 0.5)
    B_ui = st.sidebar.number_input(f"Base Externa B ({u_sec})", min_value=0.01, value=100.0 if sis=="Métrico (SI)" else 4.0, step=10.0 if sis=="Métrico (SI)" else 0.5)
    t_ui = st.sidebar.number_input(f"Espessura da Parede t ({u_sec})", min_value=0.01, value=5.0 if sis=="Métrico (SI)" else 0.2, step=1.0 if sis=="Métrico (SI)" else 0.1)
elif tipo_secao == "Tubo Circular":
    H_val_ui = st.sidebar.number_input(f"Diâmetro Externo D ({u_sec})", min_value=0.01, value=114.3 if sis=="Métrico (SI)" else 4.5, step=10.0 if sis=="Métrico (SI)" else 0.5)
    t_ui = st.sidebar.number_input(f"Espessura da Parede t ({u_sec})", min_value=0.01, value=6.02 if sis=="Métrico (SI)" else 0.237, step=1.0 if sis=="Métrico (SI)" else 0.1)

H_val, B = H_val_ui * f_sec, B_ui * f_sec
H_inicial, H_final = H_inicial_ui * f_sec, H_final_ui * f_sec
tw, tf, t = tw_ui * f_sec, tf_ui * f_sec, t_ui * f_sec

if tipo_secao == "Barra Maciça Retangular": I_input_cm4 = ((B/10) * (H_val/10)**3)/12
elif tipo_secao == "Entrada Manual": I_input_cm4 = I_input_ui * f_I
elif tipo_secao == "Barra Maciça Circular": I_input_cm4 = (np.pi * (H_val/10)**4) / 64
elif tipo_secao == "Perfil I / H": I_input_cm4 = ((B/10) * (H_val/10)**3)/12 - (((B/10) - (tw/10)) * ((H_val/10) - 2*(tf/10))**3)/12
elif tipo_secao == "Tubo Retangular / Quadrado": I_input_cm4 = ((B/10) * (H_val/10)**3)/12 - (((B/10) - 2*(t/10)) * ((H_val/10) - 2*(t/10))**3)/12
elif tipo_secao == "Tubo Circular": I_input_cm4 = (np.pi * ((H_val/10)**4 - ((H_val/10) - 2*(t/10))**4)) / 64

E_val, Sy_val = E_input_ui * f_E * 1e9, Sy_input_ui * f_Sy * 1e6
I_val, c_val = I_input_cm4 * 1e-8, (H_val / 1000) / 2 

H_m, B_m, tw_m, tf_m, t_m = H_val/1000.0, B/1000.0, tw/1000.0, tf/1000.0, t/1000.0
if tipo_secao == "Barra Maciça Retangular": A_val, I_min_val = B_m * H_m, min(I_val, (H_m * B_m**3)/12.0)
elif tipo_secao == "Entrada Manual":
    A_val = (A_input_ui * (1.0 if sis == "Métrico (SI)" else 6.4516)) * 1e-4 if 'A_input_ui' in locals() else 0.01
    I_min_val = (I_min_input_ui * f_I) * 1e-8 if 'I_min_input_ui' in locals() else I_val
elif tipo_secao == "Barra Maciça Circular": A_val, I_min_val = np.pi * (H_m / 2)**2, I_val
elif tipo_secao == "Perfil I / H": A_val, I_min_val = 2*B_m*tf_m + (H_m-2*tf_m)*tw_m, min(I_val, 2*(tf_m*B_m**3)/12.0 + ((H_m-2*tf_m)*tw_m**3)/12.0)
elif tipo_secao == "Tubo Retangular / Quadrado": A_val, I_min_val = B_m*H_m - (B_m-2*t_m)*(H_m-2*t_m), min(I_val, (H_m*B_m**3)/12.0 - ((H_m-2*t_m)*(B_m-2*t_m)**3)/12.0)
elif tipo_secao == "Tubo Circular": A_val, I_min_val = np.pi*((H_m/2.0)**2 - ((H_m/2.0)-t_m)**2), I_val

# ==========================================
# RENDERIZAÇÃO DA SECÇÃO TRANSVERSAL
# ==========================================
fig_secao_global = go.Figure()
offset_cota = max(H_val_ui, B_ui if 'B_ui' in locals() else H_val_ui) * 0.15 

if tipo_secao == "Entrada Manual":
    fig_secao_global.add_annotation(text="Secção Genérica", x=0, y=0, showarrow=False, font=dict(size=14))
else:
    if tipo_secao == "Barra Maciça Retangular":
        fig_secao_global.add_trace(go.Scatter(x=[-B_ui/2, B_ui/2, B_ui/2, -B_ui/2, -B_ui/2], y=[-H_val_ui/2, -H_val_ui/2, H_val_ui/2, H_val_ui/2, -H_val_ui/2], fill='toself', fillcolor='lightgray', line=dict(color='black', width=2)))
    elif tipo_secao == "Barra Maciça Circular":
        theta = np.linspace(0, 2*np.pi, 100)
        fig_secao_global.add_trace(go.Scatter(x=(H_val_ui/2)*np.cos(theta), y=(H_val_ui/2)*np.sin(theta), fill='toself', fillcolor='lightgray', line=dict(color='black', width=2)))
    elif tipo_secao == "Perfil I / H":
        fig_secao_global.add_trace(go.Scatter(x=[-B_ui/2, B_ui/2, B_ui/2, tw_ui/2, tw_ui/2, B_ui/2, B_ui/2, -B_ui/2, -B_ui/2, -tw_ui/2, -tw_ui/2, -B_ui/2, -B_ui/2], y=[-H_val_ui/2, -H_val_ui/2, -H_val_ui/2+tf_ui, -H_val_ui/2+tf_ui, H_val_ui/2-tf_ui, H_val_ui/2-tf_ui, H_val_ui/2, H_val_ui/2, H_val_ui/2-tf_ui, H_val_ui/2-tf_ui, -H_val_ui/2+tf_ui, -H_val_ui/2+tf_ui, -H_val_ui/2], fill='toself', fillcolor='lightgray', line=dict(color='black', width=2)))
    elif tipo_secao == "Tubo Retangular / Quadrado":
        fig_secao_global.add_trace(go.Scatter(x=[-B_ui/2, B_ui/2, B_ui/2, -B_ui/2, -B_ui/2], y=[-H_val_ui/2, -H_val_ui/2, H_val_ui/2, H_val_ui/2, -H_val_ui/2], fill='toself', fillcolor='lightgray', line=dict(color='black', width=2)))
        fig_secao_global.add_trace(go.Scatter(x=[-B_ui/2+t_ui, B_ui/2-t_ui, B_ui/2-t_ui, -B_ui/2+t_ui, -B_ui/2+t_ui], y=[-H_val_ui/2+t_ui, -H_val_ui/2+t_ui, H_val_ui/2-t_ui, H_val_ui/2-t_ui, -H_val_ui/2+t_ui], fill='toself', fillcolor='white', line=dict(color='black', width=2)))
    elif tipo_secao == "Tubo Circular":
        theta = np.linspace(0, 2*np.pi, 100)
        fig_secao_global.add_trace(go.Scatter(x=(H_val_ui/2)*np.cos(theta), y=(H_val_ui/2)*np.sin(theta), fill='toself', fillcolor='lightgray', line=dict(color='black', width=2)))
        fig_secao_global.add_trace(go.Scatter(x=((H_val_ui/2)-t_ui)*np.cos(theta), y=((H_val_ui/2)-t_ui)*np.sin(theta), fill='toself', fillcolor='white', line=dict(color='black', width=2)))

fig_secao_global.update_layout(height=320, margin=dict(l=10, r=30, t=10, b=10), showlegend=False, plot_bgcolor="white", xaxis=dict(showgrid=False, zeroline=False, visible=False), yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1))

# ==========================================
# MÓDULO 1: FLEXÃO TRANSVERSAL
# ==========================================
if modulo_selecionado == "1. Flexão Transversal (Vigas)":
    st.sidebar.markdown("---")
    st.sidebar.header("Cargas Transversais")
    tipo_carga = st.sidebar.selectbox("Tipo de Carga", ["Pontual", "Momento Fletor", "Distribuída (Uniforme)", "Triangular / Trapezoidal"])
    w1, w2, posicao_f = None, None, None

    if tipo_carga == "Triangular / Trapezoidal":
        w1 = st.sidebar.number_input(f"Magnitude Inicial w1 ({u_q})", value=0.0, step=1.0)
        w2 = st.sidebar.number_input(f"Magnitude Final w2 ({u_q})", value=-10.0 if sis=="Métrico (SI)" else -1.0, step=1.0)
        posicao_i = st.sidebar.number_input(f"Posição Inicial - x ({u_L})", min_value=0.0, value=0.0, step=0.5)
        posicao_f = st.sidebar.number_input(f"Posição Final - x ({u_L})", min_value=posicao_i+0.01, value=posicao_i+(3.0 if sis=="Métrico (SI)" else 10.0), step=0.5)
        valor_exibicao = f"{w1} até {w2}"
    else:
        valor = st.sidebar.number_input(f"Magnitude ({u_F} ou {u_M})", value=-10.0 if sis=="Métrico (SI)" else -2.0, step=1.0)
        posicao_i = st.sidebar.number_input(f"Posição Inicial - x ({u_L})", min_value=0.0, value=2.0 if sis=="Métrico (SI)" else 5.0, step=0.5)
        if tipo_carga == "Distribuída (Uniforme)":
            posicao_f = st.sidebar.number_input(f"Posição Final - x ({u_L})", min_value=posicao_i+0.01, value=posicao_i+(2.0 if sis=="Métrico (SI)" else 5.0), step=0.5)
        valor_exibicao = valor

    if st.sidebar.button("➕ Adicionar Carga"):
        st.session_state.cargas_salvas.append({"Tipo": tipo_carga, "Valor Exibição": valor_exibicao, "Valor Numérico": valor if tipo_carga != "Triangular / Trapezoidal" else None, "w1": w1, "w2": w2, "Posição Inicial": posicao_i, "Posição Final": posicao_f})
        st.session_state.resultados_analiticos = None
        st.rerun()

    if st.sidebar.button("🔄 Limpar Estudo"):
        st.session_state.cargas_salvas = []
        st.session_state.resultados_analiticos = None
        st.rerun()

    st.title("🏗️ Módulo de Flexão de Vigas")
    col_comp, col_tipo = st.columns([1, 2])
    comprimento_ui = col_comp.number_input(f"Comprimento L ({u_L})", min_value=0.01, value=10.0 if sis=="Métrico (SI)" else 30.0, step=0.005)
    tipo_viga = col_tipo.selectbox("Apoios:", ["Biapoiada nas extremidades", "Engastada à Esquerda (x=0)", "Biapoiada Personalizada (com ou sem balanço)"])

    pos_apoio1_ui, pos_apoio2_ui = 0.0, float(comprimento_ui)
    if tipo_viga == "Biapoiada Personalizada (com ou sem balanço)":
        cA, cB = st.columns(2)
        pos_apoio1_ui = cA.number_input(f"Apoio 1 (x em {u_L})", min_value=0.0, max_value=float(comprimento_ui), value=min((2.0 if sis=="Métrico (SI)" else 5.0), float(comprimento_ui)))
        pos_apoio2_ui = cB.number_input(f"Apoio 2 (x em {u_L})", min_value=0.0, max_value=float(comprimento_ui), value=min((8.0 if sis=="Métrico (SI)" else 25.0), float(comprimento_ui)))

    comprimento, pos_apoio1, pos_apoio2 = comprimento_ui * f_pos, pos_apoio1_ui * f_pos, pos_apoio2_ui * f_pos

    st.write("---")
    col_viga, col_secao = st.columns([2.2, 1])

    with col_viga:
        fig_esquema = go.Figure()
        if variacao_h == "Variável (Cunha)":
            h_escala = 0.3 
            H_vet_plot = H_inicial_ui + (H_final_ui - H_inicial_ui) * (np.array([0, float(comprimento_ui)]) / float(comprimento_ui))
            if "Topo" in alinhamento_cunha: y_top, y_bot = np.array([h_escala, h_escala]), h_escala - (H_vet_plot / H_inicial_ui) * (2 * h_escala)
            elif "Base" in alinhamento_cunha: y_bot, y_top = np.array([-h_escala, -h_escala]), -h_escala + (H_vet_plot / H_inicial_ui) * (2 * h_escala)
            else: y_top, y_bot = (H_vet_plot / H_inicial_ui) * h_escala, -(H_vet_plot / H_inicial_ui) * h_escala
            fig_esquema.add_trace(go.Scatter(x=[0, float(comprimento_ui), float(comprimento_ui), 0, 0], y=[y_top[0], y_top[1], y_bot[1], y_bot[0], y_top[0]], fill='toself', fillcolor='darkgray', line=dict(color='black', width=2), hoverinfo='none'))
        else:
            fig_esquema.add_trace(go.Scatter(x=[0, float(comprimento_ui)], y=[0, 0], mode='lines', line=dict(color='darkgray', width=12), hoverinfo='none'))

        if tipo_viga == "Engastada à Esquerda (x=0)": fig_esquema.add_trace(go.Scatter(x=[0, 0], y=[-0.5, 0.5], mode='lines', line=dict(color='black', width=8), hoverinfo='none'))
        else:
            x_ap = [0, float(comprimento_ui)] if tipo_viga == "Biapoiada nas extremidades" else [pos_apoio1_ui, pos_apoio2_ui]
            fig_esquema.add_trace(go.Scatter(x=x_ap, y=[-0.15, -0.15], mode='markers', marker=dict(symbol='triangle-up', size=18, color='darkorange'), hoverinfo='none'))

        for c in st.session_state.cargas_salvas:
            tipo, xi, texto, xf = c["Tipo"], c["Posição Inicial"], str(c["Valor Exibição"]), c["Posição Final"] if c["Posição Final"] is not None else c["Posição Inicial"]
            if tipo == "Pontual":
                val = float(c["Valor Numérico"])
                is_down = (val < 0)
                ay_val, color = (0.8, 'red') if is_down else (-0.8, 'blue')
                fig_esquema.add_annotation(x=xi, y=0, ax=xi, ay=ay_val, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor=color)
                fig_esquema.add_annotation(x=xi, y=ay_val + (0.15 if is_down else -0.15), text=f"{val} {u_F}", showarrow=False, font=dict(color=color, size=12))
            elif tipo == "Momento Fletor":
                val = float(c["Valor Numérico"])
                color, dir_str, yshift_val = ("purple", "↻", 30) if val < 0 else ("green", "↺", -30)
                fig_esquema.add_annotation(x=xi, y=0, text=f"{dir_str} {val} {u_M}", showarrow=False, font=dict(color=color, size=14), yshift=yshift_val)
                fig_esquema.add_trace(go.Scatter(x=[xi], y=[0], mode='markers', marker=dict(symbol='circle-open', size=25, color=color, line_width=3), hoverinfo='none'))
            elif tipo in ["Distribuída (Uniforme)", "Triangular / Trapezoidal"]:
                w1_val, w2_val = float(c["w1"] if c["w1"] is not None else c["Valor Numérico"]), float(c["w2"] if c["w2"] is not None else c["Valor Numérico"])
                mx = max(abs(w1_val), abs(w2_val))
                mx = mx if mx > 0 else 1
                h1, h2 = (abs(w1_val)/mx)*0.5, (abs(w2_val)/mx)*0.5
                mult, color, fill_color = (1, 'red', 'rgba(255, 0, 0, 0.2)') if ((w1_val + w2_val) / 2) <= 0 else (-1, 'blue', 'rgba(0, 0, 255, 0.2)')
                fig_esquema.add_trace(go.Scatter(x=[xi, xi, xf, xf, xi], y=[0, h1*mult, h2*mult, 0, 0], fill='toself', fillcolor=fill_color, line=dict(color=color, width=1.5), hoverinfo='none'))
                fig_esquema.add_annotation(x=(xi+xf)/2, y=max(h1, h2)*mult + (0.15*mult), text=texto, showarrow=False, font=dict(color=color, size=12))

        fig_esquema.update_layout(yaxis=dict(range=[-0.8, 1.5], showgrid=False, zeroline=False, showticklabels=False), xaxis=dict(range=[-0.5, float(comprimento_ui)+0.5], showgrid=True, title=f"Posição na viga (x) em {u_L}"), height=320, margin=dict(l=10, r=10, t=40, b=30), showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig_esquema, use_container_width=True)

    with col_secao:
        st.plotly_chart(fig_secao_global, use_container_width=True)

    if len(st.session_state.cargas_salvas) > 0:
        st.dataframe(pd.DataFrame(st.session_state.cargas_salvas)[["Tipo", "Valor Exibição", "Posição Inicial", "Posição Final"]], use_container_width=True)
    else:
        st.info("Adicione cargas na barra lateral para calcular os diagramas de flexão.")
        
    if st.button("🚀 Processar Diagramas de Flexão", type="primary") and len(st.session_state.cargas_salvas) > 0:
        with st.spinner("A integrar equações de Flexão..."):
            E, I_sym = sp.symbols('E I')
            R_A, R_B, M_A = sp.symbols('R_A R_B M_A')
            viga = Beam(comprimento, E, I_sym)
            
            reacoes_desconhecidas = []
            if tipo_viga == "Biapoiada nas extremidades":
                viga.bc_deflection = [(0, 0), (comprimento, 0)]
                viga.apply_load(R_A, 0, -1); viga.apply_load(R_B, comprimento, -1)
                reacoes_desconhecidas = [R_A, R_B]
            elif tipo_viga == "Engastada à Esquerda (x=0)":
                viga.bc_deflection = [(0, 0)]; viga.bc_slope = [(0, 0)]
                viga.apply_load(R_A, 0, -1); viga.apply_load(M_A, 0, -2)
                reacoes_desconhecidas = [R_A, M_A]
            elif tipo_viga == "Biapoiada Personalizada (com ou sem balanço)":
                viga.bc_deflection = [(pos_apoio1, 0), (pos_apoio2, 0)]
                viga.apply_load(R_A, pos_apoio1, -1); viga.apply_load(R_B, pos_apoio2, -1)
                reacoes_desconhecidas = [R_A, R_B]

            for carga in st.session_state.cargas_salvas:
                tipo, pi, pf = carga["Tipo"], carga["Posição Inicial"] * f_pos, carga["Posição Final"] * f_pos if carga["Posição Final"] is not None else None
                if tipo == "Pontual": viga.apply_load(carga["Valor Numérico"] * f_F, pi, -1)
                elif tipo == "Momento Fletor": viga.apply_load(carga["Valor Numérico"] * f_M, pi, -2)
                elif tipo == "Distribuída (Uniforme)":
                    viga.apply_load(carga["Valor Numérico"] * f_q, pi, 0)
                    if pf < comprimento: viga.apply_load(-carga["Valor Numérico"] * f_q, pf, 0)
                elif tipo == "Triangular / Trapezoidal":
                    w1_val, w2_val = carga["w1"] * f_q, carga["w2"] * f_q
                    q_val = (w2_val - w1_val) / (pf - pi)
                    if w1_val != 0: viga.apply_load(w1_val, pi, 0)
                    if q_val != 0: viga.apply_load(q_val, pi, 1)
                    if pf <= comprimento:
                        if w2_val != 0: viga.apply_load(-w2_val, pf, 0)
                        if q_val != 0: viga.apply_load(-q_val, pf, 1)
            
            viga.solve_for_reaction_loads(*reacoes_desconhecidas)
            reacoes = viga.reaction_loads
            x = sp.Symbol('x')
            
            eq_cortante_macaulay, eq_momento_macaulay = viga.shear_force(), viga.bending_moment()
            eq_V_clean, eq_M_clean = clean_expr(eq_cortante_macaulay), clean_expr(eq_momento_macaulay)
            eq_V_latex, eq_M_latex = sp.latex(eq_V_clean), sp.latex(eq_M_clean)
            
            func_V = sp.lambdify(x, eq_cortante_macaulay.rewrite(sp.Piecewise), 'numpy')
            func_M = sp.lambdify(x, eq_momento_macaulay.rewrite(sp.Piecewise), 'numpy')
            
            pontos_notaveis = [0.0, float(comprimento)]
            if tipo_viga == "Biapoiada Personalizada (com ou sem balanço)": pontos_notaveis.extend([pos_apoio1, pos_apoio2])
            for c in st.session_state.cargas_salvas:
                pontos_notaveis.append(c["Posição Inicial"] * f_pos)
                if c["Posição Final"] is not None: pontos_notaveis.append(c["Posição Final"] * f_pos)
                    
            x_vetor = np.sort(np.unique(np.concatenate((np.linspace(0, float(comprimento), 800), pontos_notaveis))))
            V_vetor, M_vetor = np.zeros_like(x_vetor), np.zeros_like(x_vetor)
            
            for k, val in enumerate(x_vetor):
                try:
                    res_v = func_V(val)
                    V_vetor[k] = float(res_v) if not isinstance(res_v, np.ndarray) and not np.isinf(res_v) else float(func_V(val+1e-7))
                except: pass
                try:
                    res_m = func_M(val)
                    M_vetor[k] = float(res_m) if not isinstance(res_m, np.ndarray) and not np.isinf(res_m) else float(func_M(val+1e-7))
                except: pass
                    
            idx_max_V, idx_max_M = np.argmax(np.abs(V_vetor)), np.argmax(np.abs(M_vetor))
            pos_V_max, pos_M_max, max_M_kNm = x_vetor[idx_max_V], x_vetor[idx_max_M], np.abs(M_vetor)[idx_max_M]
            
            if variacao_h == "Variável (Cunha)":
                H_vetor_mm = H_inicial + (H_final - H_inicial) * (x_vetor / float(comprimento))
                I_vetor_m4 = (B/1000) * (H_vetor_mm/1000)**3 / 12
                c_vetor_m = (H_vetor_mm / 1000) / 2
                tensao_vetor_MPa = (np.abs(M_vetor) * 1000 * c_vetor_m) / I_vetor_m4 / 1e6
                idx_max_tensao = np.argmax(tensao_vetor_MPa)
                tensao_max_MPa, posicao_tensao_max = tensao_vetor_MPa[idx_max_tensao], x_vetor[idx_max_tensao]
                
                kappa = (M_vetor * 1000) / (E_val * I_vetor_m4)
                y_rel = cumtrapz_np(cumtrapz_np(kappa, x_vetor), x_vetor)
                
                if tipo_viga == "Engastada à Esquerda (x=0)": y_vetor_m = y_rel
                else:
                    idx_ap1, idx_ap2 = np.argmin(np.abs(x_vetor - pos_apoio1)), np.argmin(np.abs(x_vetor - pos_apoio2))
                    C1 = -(y_rel[idx_ap2] - y_rel[idx_ap1]) / (x_vetor[idx_ap2] - x_vetor[idx_ap1]) if x_vetor[idx_ap2] != x_vetor[idx_ap1] else 0
                    y_vetor_m = y_rel + C1 * x_vetor + (-y_rel[idx_ap1] - C1 * x_vetor[idx_ap1])
                y_vetor_mm = y_vetor_m * 1000
                eq_Y_latex = r"\text{Resolvido via Dupla Integracao Numerica: } \iint \frac{M(x)}{E \cdot I(x)} \, dx^2"
            else:
                eq_flecha_macaulay = viga.deflection().subs({E: E_val, I_sym: I_val})
                eq_Y_latex = sp.latex(clean_expr(eq_flecha_macaulay, sig_figs=4))
                func_y = sp.lambdify(x, eq_flecha_macaulay.rewrite(sp.Piecewise), 'numpy')
                y_vetor_m = np.zeros_like(x_vetor)
                for k, val in enumerate(x_vetor):
                    try: y_vetor_m[k] = float(func_y(val))
                    except: pass
                y_vetor_mm = y_vetor_m * 1000 
                tensao_max_MPa, posicao_tensao_max = (((max_M_kNm * 1000) * c_val) / I_val) / 1e6, pos_M_max 
                
            idx_max_Y = np.argmax(np.abs(y_vetor_mm))
            pos_Y_max, max_flecha = x_vetor[idx_max_Y], np.abs(y_vetor_mm)[idx_max_Y]
            
            out_L, out_F, out_M, out_stress, out_defl = 1.0/f_pos, 1.0/f_F, 1.0/f_M, (1.0/6.894757 if sis=="Imperial (US)" else 1.0), (1.0/25.4 if sis=="Imperial (US)" else 1.0)
            
            txt_reacoes = []
            if tipo_viga == "Engastada à Esquerda (x=0)":
                txt_reacoes.append(f"Reação R_A (x=0): {float(reacoes[R_A]) * out_F:.2f} {u_F}")
                txt_reacoes.append(f"Momento M_A (Engaste): {float(reacoes[M_A]) * out_M:.2f} {u_M}")
            elif tipo_viga == "Biapoiada Personalizada (com ou sem balanço)":
                txt_reacoes.append(f"Reação R_A (x={pos_apoio1_ui:.2f}{u_L}): {float(reacoes[R_A]) * out_F:.2f} {u_F}")
                txt_reacoes.append(f"Reação R_B (x={pos_apoio2_ui:.2f}{u_L}): {float(reacoes[R_B]) * out_F:.2f} {u_F}")
            else:
                txt_reacoes.append(f"Reação R_A (x=0): {float(reacoes[R_A]) * out_F:.2f} {u_F}")
                txt_reacoes.append(f"Reação R_B (x={comprimento_ui:.2f}{u_L}): {float(reacoes[R_B]) * out_F:.2f} {u_F}")

            fig_plotly = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=(f"Cortante ({u_F})", f"Momento Fletor ({u_M})", f"Linha Elástica ({u_defl})"))
            fig_plotly.add_trace(go.Scatter(x=x_vetor*out_L, y=V_vetor*out_F, fill='tozeroy', line=dict(color='royalblue')), row=1, col=1)
            fig_plotly.add_vline(x=pos_V_max*out_L, line_dash="dash", line_color="black", annotation_text=f" Máx", row=1, col=1)
            fig_plotly.add_trace(go.Scatter(x=x_vetor*out_L, y=M_vetor*out_M, fill='tozeroy', line=dict(color='firebrick')), row=2, col=1)
            fig_plotly.add_vline(x=pos_M_max*out_L, line_dash="dash", line_color="black", annotation_text=f" Máx", row=2, col=1)
            fig_plotly.update_yaxes(autorange="reversed", row=2, col=1)
            fig_plotly.add_trace(go.Scatter(x=x_vetor*out_L, y=y_vetor_mm*out_defl, fill='tozeroy', line=dict(color='seagreen')), row=3, col=1)
            fig_plotly.add_vline(x=pos_Y_max*out_L, line_dash="dash", line_color="black", annotation_text=f" Máx", row=3, col=1)
            fig_plotly.update_layout(height=700, showlegend=False, hovermode="x unified", margin=dict(l=40, r=40, t=40, b=40))

            st.session_state.resultados_analiticos = {
                'reacoes': txt_reacoes,
                'max_M_disp': max_M_kNm * out_M, 'pos_M_max_disp': pos_M_max * out_L,
                'tensao_max_disp': tensao_max_MPa * out_stress, 'posicao_tensao_max_disp': posicao_tensao_max * out_L,
                'max_flecha_disp': max_flecha * out_defl, 'pos_Y_max_disp': pos_Y_max * out_L,
                'fig_plotly': fig_plotly,
                'eq_V_latex': eq_V_latex, 'eq_M_latex': eq_M_latex, 'eq_Y_latex': eq_Y_latex
            }
            st.rerun()

    if st.session_state.resultados_analiticos:
        res = st.session_state.resultados_analiticos
        st.subheader("Resultados Analíticos")
        cols_reac = st.columns(len(res['reacoes']))
        for i, reac in enumerate(res['reacoes']): cols_reac[i].success(f"**{reac}**")
            
        with st.expander("Ver Equações Analíticas Completas (LaTeX)"):
            st.latex(f"V(x) = {res['eq_V_latex']}")
            st.latex(f"M(x) = {res['eq_M_latex']}")
            st.latex(f"y(x) = {res['eq_Y_latex']}")
            
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Momento Máx (x={res['pos_M_max_disp']:.2f}{u_L})", f"{res['max_M_disp']:.2f} {u_M}")
        c2.metric(f"Tensão Máx de Flexão (x={res['posicao_tensao_max_disp']:.2f}{u_L})", f"{res['tensao_max_disp']:.1f} {u_stress}")
        c3.metric(f"Deflexão Máx (x={res['pos_Y_max_disp']:.2f}{u_L})", f"{res['max_flecha_disp']:.2f} {u_defl}")

        st.plotly_chart(res['fig_plotly'], use_container_width=True)

# ==========================================
# MÓDULO 2: FLAMBAGEM AXIAL (COLUNAS)
# ==========================================
elif modulo_selecionado == "2. Flambagem Axial (Colunas)":
    st.sidebar.markdown("---")
    st.sidebar.header("Carga e Fixação")
    
    comprimento_coluna_ui = st.sidebar.number_input(f"Comprimento da Coluna L ({u_L})", min_value=0.01, value=3.0 if sis=="Métrico (SI)" else 10.0, step=0.1)
    condicao_apoio = st.sidebar.selectbox("Fator K", ["Articulado - Articulado (K = 1.0)", "Engastado - Livre / Poste (K = 2.0)", "Engastado - Engastado (K = 0.5)", "Engastado - Articulado (K = 0.7)"])
    carga_axial_ui = st.sidebar.number_input(f"Carga de Compressão P ({u_F})", min_value=0.0, value=100.0 if sis=="Métrico (SI)" else 20.0, step=10.0)

    st.title("🏛️ Módulo de Flambagem (Colunas)")
    
    col_vazio, col_secao_f = st.columns([1.5, 1])
    with col_secao_f: st.plotly_chart(fig_secao_global, use_container_width=True)
        
    with col_vazio:
        st.info("👈 Defina os parâmetros da coluna para visualizar os limites de estabilidade (Euler/Johnson).")
        if st.button("🚀 Calcular Flambagem", type="primary"):
            L_coluna = comprimento_coluna_ui * f_pos
            K_val = 1.0
            if "K = 2.0" in condicao_apoio: K_val = 2.0
            elif "K = 0.5" in condicao_apoio: K_val = 0.5
            elif "K = 0.7" in condicao_apoio: K_val = 0.7
                
            L_efetivo = K_val * L_coluna
            
            if I_min_val > 0 and A_val > 0:
                raio_giracao = np.sqrt(I_min_val / A_val)
                esbeltez = L_efetivo / raio_giracao
                lambda_c = np.sqrt((2 * np.pi**2 * E_val) / Sy_val)
                
                if esbeltez >= lambda_c:
                    teoria_usada = "Equação de Euler (Coluna Esbelta)"
                    tensao_critica_Pa = (np.pi**2 * E_val) / (esbeltez**2)
                else:
                    teoria_usada = "Parábola de Johnson (Coluna Curta/Intermediária)"
                    tensao_critica_Pa = Sy_val - (Sy_val**2 / (4 * np.pi**2 * E_val)) * (esbeltez**2)
                    
                P_cr_N = tensao_critica_Pa * A_val
                P_cr_kN, tensao_critica_MPa = P_cr_N / 1000.0, tensao_critica_Pa / 1e6
            else:
                P_cr_kN, esbeltez, lambda_c, tensao_critica_MPa, teoria_usada = 0.0, 0.0, 0.0, 0.0, "N/A"
                P_cr_N, raio_giracao, tensao_critica_Pa = 0.0, 0.0, 0.0
                
            out_F = 1.0 / f_F
            out_stress = 1.0 / 6.894757 if sis == "Imperial (US)" else 1.0
            
            P_cr_disp, carga_aplicada_kN = P_cr_kN * out_F, carga_axial_ui / out_F
            tensao_cr_disp, Sy_disp = tensao_critica_MPa * out_stress, (Sy_val / 1e6) * out_stress
            fator_seguranca = (P_cr_kN / carga_aplicada_kN) if carga_aplicada_kN > 0 else float('inf')
            
            y_grafico = np.linspace(0, comprimento_coluna_ui, 100)
            if K_val == 1.0: x_grafico = np.sin(np.pi * y_grafico / comprimento_coluna_ui)
            elif K_val == 2.0: x_grafico = 1 - np.cos(np.pi * y_grafico / (2 * comprimento_coluna_ui))
            elif K_val == 0.5: x_grafico = 1 - np.cos(2 * np.pi * y_grafico / comprimento_coluna_ui)
            elif K_val == 0.7: x_grafico = (y_grafico/comprimento_coluna_ui) - np.sin(4.4934 * y_grafico / comprimento_coluna_ui)/4.4934
                
            fig_col = go.Figure()
            fig_col.add_trace(go.Scatter(x=np.zeros_like(y_grafico), y=y_grafico, mode='lines', line=dict(color='gray', width=3, dash='dash'), name='Eixo Original'))
            fig_col.add_trace(go.Scatter(x=x_grafico, y=y_grafico, mode='lines', line=dict(color='red', width=5), name='Modo de Flambagem'))
            if K_val == 1.0: fig_col.add_trace(go.Scatter(x=[0, 0], y=[0, comprimento_coluna_ui], mode='markers', marker=dict(symbol='triangle-up', size=15, color='orange')))
            elif K_val == 2.0: fig_col.add_trace(go.Scatter(x=[-0.5, 0.5], y=[0, 0], mode='lines', line=dict(color='black', width=8)))
            elif K_val == 0.5:
                fig_col.add_trace(go.Scatter(x=[-0.5, 0.5], y=[0, 0], mode='lines', line=dict(color='black', width=8)))
                fig_col.add_trace(go.Scatter(x=[-0.5, 0.5], y=[comprimento_coluna_ui, comprimento_coluna_ui], mode='lines', line=dict(color='black', width=8)))
            elif K_val == 0.7:
                fig_col.add_trace(go.Scatter(x=[-0.5, 0.5], y=[0, 0], mode='lines', line=dict(color='black', width=8)))
                fig_col.add_trace(go.Scatter(x=[0], y=[comprimento_coluna_ui], mode='markers', marker=dict(symbol='triangle-up', size=15, color='orange')))

            fig_col.update_layout(title="Modo Visual (Vista do Eixo Fraco)", xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=True, title=f"Altura ({u_L})"), height=600, width=400, showlegend=False, plot_bgcolor="white")
            
            st.session_state.resultados_flambagem = {
                'P_cr_disp': P_cr_disp, 'tensao_cr_disp': tensao_cr_disp, 'esbeltez': esbeltez, 'lambda_c': lambda_c,
                'teoria': teoria_usada, 'fator_seguranca': fator_seguranca, 'carga_ui': carga_axial_ui,
                'fig_col': fig_col, 'A_val': A_val, 'I_min_val': I_min_val, 'raio_giracao': raio_giracao,
                'L_efetivo': L_efetivo, 'E_val': E_val, 'Sy_val': Sy_val, 'K_val': K_val,
                'L_coluna': L_coluna, 'tensao_critica_Pa': tensao_critica_Pa, 'P_cr_N': P_cr_N,
                'P_cr_kN': P_cr_kN, 'carga_aplicada_kN': carga_aplicada_kN
            }

    if st.session_state.resultados_flambagem:
        res_f = st.session_state.resultados_flambagem
        st.write("---")
        c1_f, c2_f = st.columns([1.5, 1])
        with c1_f:
            st.markdown(f"**Método Empregado:** `{res_f['teoria']}`")
            st.metric(f"Carga Máxima de Ruptura (Pcr)", f"{res_f['P_cr_disp']:.2f} {u_F}")
            
            if res_f['carga_ui'] > 0:
                fs = res_f['fator_seguranca']
                if fs < 1.0: st.error(f"🚨 **Fator de Segurança Global: {fs:.2f}**\n\n**COLAPSO IMINENTE!** A coluna falhará.")
                elif fs <= 2.0: st.warning(f"⚠️ **Fator de Segurança Global: {fs:.2f}**\n\nA coluna não falha matematicamente, mas o coeficiente está abaixo do padrão.")
                else: st.success(f"✅ **Fator de Segurança Global: {fs:.2f}**\n\nEstrutura segura contra flambagem e escoamento.")
                    
            with st.expander("Ver Memória de Cálculo Detalhada (Passo a Passo)"):
                st.latex(f"A = {res_f['A_val']:.6e} \\text{{ m}}^2")
                st.latex(f"I_{{min}} = {res_f['I_min_val']:.6e} \\text{{ m}}^4")
                st.latex(f"r = \\sqrt{{\\frac{{I_{{min}}}}{{A}}}} = {res_f['raio_giracao']:.4f} \\text{{ m}}")
                st.latex(f"L_e = K \\cdot L = {res_f['K_val']} \\cdot {res_f['L_coluna']:.3f} = {res_f['L_efetivo']:.3f} \\text{{ m}}")
                st.latex(f"\\lambda = \\frac{{L_e}}{{r}} = {res_f['esbeltez']:.2f}")
                st.latex(f"\\lambda_c = \\sqrt{{\\frac{{2 \\pi^2 E}}{{S_y}}}} = {res_f['lambda_c']:.2f}")
                if res_f['esbeltez'] >= res_f['lambda_c']: st.latex(f"\\sigma_{{cr}} = \\frac{{\\pi^2 E}}{{\\lambda^2}} = {res_f['tensao_critica_Pa']:.2e} \\text{{ Pa}}")
                else: st.latex(f"\\sigma_{{cr}} = S_y - \\frac{{S_y^2}}{{4 \\pi^2 E}} \\lambda^2 = {res_f['tensao_critica_Pa']:.2e} \\text{{ Pa}}")
                st.latex(f"P_{{cr}} = \\sigma_{{cr}} \\cdot A = {res_f['P_cr_kN']:.2f} \\text{{ kN}}")

        with c2_f:
            st.plotly_chart(res_f['fig_col'], use_container_width=False)
