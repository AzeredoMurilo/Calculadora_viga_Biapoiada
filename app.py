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

# Configuração da página do Streamlit
st.set_page_config(page_title="Análise de Vigas", layout="wide")

if 'cargas_salvas' not in st.session_state:
    st.session_state.cargas_salvas = []

# ==========================================
# 1. INTERFACE DE ENTRADA (BARRA LATERAL)
# ==========================================
st.sidebar.header("1. Adicionar Nova Carga")
tipo_carga = st.sidebar.selectbox("Tipo de Carga", 
    ["Pontual", "Momento Fletor", "Distribuída (Uniforme)", "Triangular / Trapezoidal"]
)

w1, w2, posicao_f = None, None, None

if tipo_carga == "Triangular / Trapezoidal":
    st.sidebar.markdown("*Para Triangular: deixe um valor como 0.*")
    w1 = st.sidebar.number_input("Magnitude Inicial w1 (kN/m)", value=0.0, step=1.0)
    w2 = st.sidebar.number_input("Magnitude Final w2 (kN/m)", value=-10.0, step=1.0)
    posicao_i = st.sidebar.number_input("Posição Inicial - x (m)", min_value=0.0, value=0.0, step=0.5)
    posicao_f = st.sidebar.number_input("Posição Final - x (m)", min_value=posicao_i+0.1, value=posicao_i+3.0, step=0.5)
    valor_exibicao = f"{w1} até {w2}"
else:
    valor = st.sidebar.number_input("Magnitude (kN ou kN.m)", value=-10.0, step=1.0)
    posicao_i = st.sidebar.number_input("Posição Inicial - x (m)", min_value=0.0, value=2.0, step=0.5)
    if tipo_carga == "Distribuída (Uniforme)":
        posicao_f = st.sidebar.number_input("Posição Final - x (m)", min_value=posicao_i+0.1, value=posicao_i+2.0, step=0.5)
    valor_exibicao = valor

if st.sidebar.button("➕ Adicionar Carga"):
    nova_carga = {
        "Tipo": tipo_carga, "Valor Exibição": valor_exibicao,
        "Valor Numérico": valor if tipo_carga != "Triangular / Trapezoidal" else None,
        "w1": w1, "w2": w2, "Posição Inicial": posicao_i, "Posição Final": posicao_f
    }
    st.session_state.cargas_salvas.append(nova_carga)
    st.sidebar.success("Carga adicionada!")

# ==========================================
# 2. SEÇÃO TRANSVERSAL E INÉRCIA
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("2. Propriedades da Seção")
E_input = st.sidebar.number_input("Módulo de Elasticidade E (GPa)", value=200.0, step=10.0)
tipo_secao = st.sidebar.selectbox("Geometria da Seção", 
    ["Entrada Manual", "Barra Maciça Retangular", "Barra Maciça Circular", "Perfil I / H", "Tubo Retangular / Quadrado", "Tubo Circular"]
)

H_val = 200.0 
I_input = 800.0 

if tipo_secao == "Entrada Manual":
    I_input = st.sidebar.number_input("Inércia I (cm⁴)", value=800.0, step=10.0)
    H_val = st.sidebar.number_input("Altura Total da Seção (mm)", value=200.0, step=10.0)
elif tipo_secao == "Barra Maciça Retangular":
    H_val = st.sidebar.number_input("Altura H (mm)", value=100.0, step=10.0)
    B = st.sidebar.number_input("Base B (mm)", value=50.0, step=10.0)
    I_input = ((B/10) * (H_val/10)**3)/12
    st.sidebar.info(f"📐 Inércia Calculada: {I_input:.2f} cm⁴")
elif tipo_secao == "Barra Maciça Circular":
    H_val = st.sidebar.number_input("Diâmetro D (mm)", value=50.0, step=10.0)
    I_input = (np.pi * (H_val/10)**4) / 64
    st.sidebar.info(f"📐 Inércia Calculada: {I_input:.2f} cm⁴")
elif tipo_secao == "Perfil I / H":
    H_val = st.sidebar.number_input("Altura Total H (mm)", value=200.0, step=10.0)
    B = st.sidebar.number_input("Largura da Mesa B (mm)", value=100.0, step=10.0)
    tw = st.sidebar.number_input("Espessura da Alma tw (mm)", value=6.35, step=1.0)
    tf = st.sidebar.number_input("Espessura da Mesa tf (mm)", value=9.0, step=1.0)
    I_input = ((B/10) * (H_val/10)**3)/12 - (((B/10) - (tw/10)) * ((H_val/10) - 2*(tf/10))**3)/12
    st.sidebar.info(f"📐 Inércia Calculada: {I_input:.2f} cm⁴")
elif tipo_secao == "Tubo Retangular / Quadrado":
    H_val = st.sidebar.number_input("Altura Externa H (mm)", value=100.0, step=10.0)
    B = st.sidebar.number_input("Base Externa B (mm)", value=100.0, step=10.0)
    t = st.sidebar.number_input("Espessura da Parede t (mm)", value=5.0, step=1.0)
    I_input = ((B/10) * (H_val/10)**3)/12 - (((B/10) - 2*(t/10)) * ((H_val/10) - 2*(t/10))**3)/12
    st.sidebar.info(f"📐 Inércia Calculada: {I_input:.2f} cm⁴")
elif tipo_secao == "Tubo Circular":
    H_val = st.sidebar.number_input("Diâmetro Externo D (mm)", value=114.3, step=10.0)
    t = st.sidebar.number_input("Espessura da Parede t (mm)", value=6.02, step=1.0)
    I_input = (np.pi * ((H_val/10)**4 - ((H_val/10) - 2*(t/10))**4)) / 64
    st.sidebar.info(f"📐 Inércia Calculada: {I_input:.2f} cm⁴")

E_val = E_input * 1e9  # Pa
I_val = I_input * 1e-8 # m⁴
c_val = (H_val / 1000) / 2 # Fibra extrema em metros

# ==========================================
# 3. TELA PRINCIPAL
# ==========================================
st.title("🏗️ Verificador Estrutural de Vigas")
st.write("Cálculo de Esforços, Tensões, Deflexões e Rotação via SymPy + Plotly")

col_comp, col_tipo = st.columns([1, 2])
comprimento = col_comp.number_input("Comprimento total da viga (m)", min_value=1.0, value=10.0, step=1.0)
tipo_viga = col_tipo.selectbox("Selecione os Apoios:", 
    ["Biapoiada nas extremidades", "Engastada à Esquerda (x=0)", "Biapoiada Personalizada (com ou sem balanço)"]
)

pos_apoio1, pos_apoio2 = 0.0, float(comprimento)
if tipo_viga == "Biapoiada Personalizada (com ou sem balanço)":
    colA, colB = st.columns(2)
    pos_apoio1 = colA.number_input("Posição do Apoio 1 (x)", min_value=0.0, max_value=float(comprimento), value=2.0)
    pos_apoio2 = colB.number_input("Posição do Apoio 2 (x)", min_value=0.0, max_value=float(comprimento), value=8.0)

st.write("---")
if len(st.session_state.cargas_salvas) > 0:
    df_exibicao = pd.DataFrame(st.session_state.cargas_salvas)[["Tipo", "Valor Exibição", "Posição Inicial", "Posição Final"]]
    st.dataframe(df_exibicao, use_container_width=True)
    if st.button("🗑️ Limpar Todas as Cargas"):
        st.session_state.cargas_salvas = []
        st.rerun()
else:
    st.info("Nenhuma carga inserida. Use o menu lateral para adicionar.")

# ==========================================
# 4. FUNÇÃO GERADORA DE PDF (Com Matplotlib interno)
# ==========================================
def gerar_relatorio_pdf(x_vetor, V_vetor, M_vetor, y_vetor_mm, df_cargas, reacoes_texto, metricas_texto):
    # Gerando os gráficos estáticos silenciosamente na memória para o PDF
    fig_mpl, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
    ax1.plot(x_vetor, V_vetor, color='blue'); ax1.fill_between(x_vetor, V_vetor, 0, alpha=0.2, color='blue')
    ax1.axhline(0, color='black', linewidth=1); ax1.set_title("Esforço Cortante (V)"); ax1.grid(True)
    
    ax2.plot(x_vetor, M_vetor, color='red'); ax2.fill_between(x_vetor, M_vetor, 0, alpha=0.2, color='red')
    ax2.axhline(0, color='black', linewidth=1); ax2.invert_yaxis(); ax2.set_title("Momento Fletor (M)"); ax2.grid(True)

    ax3.plot(x_vetor, y_vetor_mm, color='green'); ax3.fill_between(x_vetor, y_vetor_mm, 0, alpha=0.2, color='green')
    ax3.axhline(0, color='black', linewidth=1); ax3.set_title("Deflexão (mm)"); ax3.grid(True)
    
    plt.tight_layout()
    img_buffer = io.BytesIO()
    fig_mpl.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    plt.close(fig_mpl)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Relatorio de Analise Estrutural", ln=True, align='C')
    
    pdf.set_font("Arial", 'B', 12)
    pdf.ln(5)
    pdf.cell(200, 10, txt="1. Propriedades e Metricas Maximas:", ln=True)
    pdf.set_font("Arial", '', 11)
    for m in metricas_texto: pdf.cell(200, 8, txt=m, ln=True)

    pdf.set_font("Arial", 'B', 12)
    pdf.ln(5)
    pdf.cell(200, 10, txt="2. Reacoes de Apoio Encontradas:", ln=True)
    pdf.set_font("Arial", '', 11)
    for r in reacoes_texto: pdf.cell(200, 8, txt=r, ln=True)
    
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="3. Diagramas Estruturais (V, M e Linha Elastica):", ln=True)
    pdf.image(img_buffer, x=10, y=30, w=190)
    
    return bytes(pdf.output())

# ==========================================
# 5. CÁLCULO E PLOTAGEM (BACKEND)
# ==========================================
st.write("---")

if st.button("🚀 Calcular Estrutura") and len(st.session_state.cargas_salvas) > 0:
    
    with st.spinner("Processando tensores e integrais de Macaulay..."):
        E, I = sp.symbols('E I')
        R_A, R_B, M_A = sp.symbols('R_A R_B M_A')
        viga = Beam(comprimento, E, I)
        
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
            tipo, pi, pf = carga["Tipo"], carga["Posição Inicial"], carga["Posição Final"]
            if tipo == "Pontual": viga.apply_load(carga["Valor Numérico"], pi, -1)
            elif tipo == "Momento Fletor": viga.apply_load(carga["Valor Numérico"], pi, -2)
            elif tipo == "Distribuída (Uniforme)":
                viga.apply_load(carga["Valor Numérico"], pi, 0)
                if pf < comprimento: viga.apply_load(-carga["Valor Numérico"], pf, 0)
            elif tipo == "Triangular / Trapezoidal":
                w1, w2 = carga["w1"], carga["w2"]
                q = (w2 - w1) / (pf - pi)
                if w1 != 0: viga.apply_load(w1, pi, 0)
                if q != 0: viga.apply_load(q, pi, 1)
                if pf <= comprimento:
                    if w2 != 0: viga.apply_load(-w2, pf, 0)
                    if q != 0: viga.apply_load(-q, pf, 1)
        
        viga.solve_for_reaction_loads(*reacoes_desconhecidas)
        reacoes = viga.reaction_loads
        
        x = sp.Symbol('x')
        eq_cortante = viga.shear_force().rewrite(sp.Piecewise)
        eq_momento = viga.bending_moment().rewrite(sp.Piecewise)
        eq_rotacao = viga.slope().subs({E: E_val, I: I_val}).rewrite(sp.Piecewise) 
        eq_flecha = viga.deflection().subs({E: E_val, I: I_val}).rewrite(sp.Piecewise)
        
        func_V = sp.lambdify(x, eq_cortante, 'numpy')
        func_M = sp.lambdify(x, eq_momento, 'numpy')
        func_theta = sp.lambdify(x, eq_rotacao, 'numpy')
        func_y = sp.lambdify(x, eq_flecha, 'numpy')
        
        x_vetor = np.linspace(0, float(comprimento), 800) 
        
        V_vetor = np.ones_like(x_vetor) * func_V(x_vetor) if np.isscalar(func_V(x_vetor)) else func_V(x_vetor)
        M_vetor = np.ones_like(x_vetor) * func_M(x_vetor) if np.isscalar(func_M(x_vetor)) else func_M(x_vetor)
        theta_vetor = np.ones_like(x_vetor) * func_theta(x_vetor) if np.isscalar(func_theta(x_vetor)) else func_theta(x_vetor)
        y_vetor_m = np.ones_like(x_vetor) * func_y(x_vetor) if np.isscalar(func_y(x_vetor)) else func_y(x_vetor)
        y_vetor_mm = y_vetor_m * 1000 
        
        txt_reacoes = []
        col_r1, col_r2, col_r3 = st.columns(3)
        if tipo_viga == "Engastada à Esquerda (x=0)":
            txt_reacoes.append(f"R_A (x=0): {float(reacoes[R_A]):.2f} kN")
            txt_reacoes.append(f"M_A (Engaste): {float(reacoes[M_A]):.2f} kN.m")
            col_r1.success(f"**{txt_reacoes[0]}**")
            col_r2.warning(f"**{txt_reacoes[1]}**")
        elif tipo_viga == "Biapoiada Personalizada (com ou sem balanço)":
            txt_reacoes.append(f"R_A (x={pos_apoio1}): {float(reacoes[R_A]):.2f} kN")
            txt_reacoes.append(f"R_B (x={pos_apoio2}): {float(reacoes[R_B]):.2f} kN")
            col_r1.success(f"**{txt_reacoes[0]}**")
            col_r2.success(f"**{txt_reacoes[1]}**")
        else:
            txt_reacoes.append(f"R_A (x=0): {float(reacoes[R_A]):.2f} kN")
            txt_reacoes.append(f"R_B (x={comprimento}): {float(reacoes[R_B]):.2f} kN")
            col_r1.success(f"**{txt_reacoes[0]}**")
            col_r2.success(f"**{txt_reacoes[1]}**")

        max_M_kNm = np.max(np.abs(M_vetor))
        tensao_max_MPa = (((max_M_kNm * 1000) * c_val) / I_val) / 1e6 
        max_flecha = np.max(np.abs(y_vetor_mm))
        max_rotacao = np.max(np.abs(theta_vetor))

        txt_metricas = [
            f"Momento Maximo Absoluto: {max_M_kNm:.2f} kN.m",
            f"Tensao Maxima de Flexao: {tensao_max_MPa:.1f} MPa",
            f"Deflexao Maxima Absoluta: {max_flecha:.2f} mm"
        ]

        st.write("### Esforços e Deslocamentos Máximos (Valores Absolutos)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Momento Máximo |M|", f"{max_M_kNm:.2f} kN.m")
        c2.metric("Tensão Máxima (σ)", f"{tensao_max_MPa:.1f} MPa")
        c3.metric("Deflexão Máxima", f"{max_flecha:.2f} mm")
        c4.metric("Rotação Máxima (θ)", f"{max_rotacao:.5f} rad")

        # ==========================================
        # 6. PLOTAGEM INTERATIVA (PLOTLY)
        # ==========================================
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Diagrama de Esforço Cortante (V)", 
                                            "Diagrama de Momento Fletor (M)", 
                                            "Linha Elástica / Deflexão"))

        # Cortante
        fig.add_trace(go.Scatter(x=x_vetor, y=V_vetor, fill='tozeroy', mode='lines', 
                                 line=dict(color='royalblue', width=2), name='Cortante (kN)'), row=1, col=1)
        fig.update_yaxes(title_text="Força (kN)", zeroline=True, zerolinecolor='black', row=1, col=1)

        # Momento (Invertido)
        fig.add_trace(go.Scatter(x=x_vetor, y=M_vetor, fill='tozeroy', mode='lines', 
                                 line=dict(color='firebrick', width=2), name='Momento (kN.m)'), row=2, col=1)
        fig.update_yaxes(title_text="Momento (kN.m)", autorange="reversed", zeroline=True, zerolinecolor='black', row=2, col=1)

        # Deflexão
        fig.add_trace(go.Scatter(x=x_vetor, y=y_vetor_mm, fill='tozeroy', mode='lines', 
                                 line=dict(color='seagreen', width=2), name='Deflexão (mm)'), row=3, col=1)
        fig.update_yaxes(title_text="Deflexão (mm)", zeroline=True, zerolinecolor='black', row=3, col=1)
        fig.update_xaxes(title_text="Posição x (m)", row=3, col=1)

        # Configurações do layout geral
        fig.update_layout(height=850, showlegend=False, hovermode="x unified",
                          margin=dict(l=40, r=40, t=40, b=40))

        st.plotly_chart(fig, use_container_width=True)

        # ==========================================
        # 7. BOTÃO DE DOWNLOAD DO PDF
        # ==========================================
        pdf_bytes = gerar_relatorio_pdf(x_vetor, V_vetor, M_vetor, y_vetor_mm, 
                                        pd.DataFrame(st.session_state.cargas_salvas), txt_reacoes, txt_metricas)
        st.write("---")
        st.download_button(
            label="📄 Baixar Relatório em PDF",
            data=pdf_bytes,
            file_name="Relatorio_Viga.pdf",
            mime="application/pdf",
            type="primary"
        )
