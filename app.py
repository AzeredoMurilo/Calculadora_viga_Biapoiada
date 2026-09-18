import streamlit as st
import pandas as pd
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.continuum_mechanics.beam import Beam

# Configuração da página do Streamlit
st.set_page_config(page_title="Análise de Vigas", layout="wide")

# ==========================================
# 1. INICIALIZAÇÃO DO ESTADO (MEMÓRIA)
# ==========================================
if 'cargas_salvas' not in st.session_state:
    st.session_state.cargas_salvas = []

# ==========================================
# 2. INTERFACE DE ENTRADA (BARRA LATERAL)
# ==========================================
st.sidebar.header("1. Adicionar Nova Carga")

tipo_carga = st.sidebar.selectbox("Tipo de Carga", 
    ["Pontual", "Momento Fletor", "Distribuída (Uniforme)", "Triangular / Trapezoidal"]
)

w1, w2 = None, None
posicao_f = None

if tipo_carga == "Triangular / Trapezoidal":
    st.sidebar.markdown("*Para Triangular: deixe um dos valores como 0.*")
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
        "Tipo": tipo_carga,
        "Valor Exibição": valor_exibicao,
        "Valor Numérico": valor if tipo_carga != "Triangular / Trapezoidal" else None,
        "w1": w1,
        "w2": w2,
        "Posição Inicial": posicao_i,
        "Posição Final": posicao_f
    }
    st.session_state.cargas_salvas.append(nova_carga)
    st.sidebar.success("Carga adicionada!")

# ==========================================
# NOVO: SEÇÃO TRANSVERSAL E INÉRCIA
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("2. Propriedades da Seção")
E_input = st.sidebar.number_input("Módulo de Elasticidade E (GPa)", value=200.0, step=10.0, help="Aço = ~200 GPa | Concreto = ~25 GPa")

tipo_secao = st.sidebar.selectbox("Geometria da Seção Transversal", 
    ["Entrada Manual (cm⁴)", "Perfil I / H", "Tubo Retangular / Quadrado", "Tubo Circular"]
)

I_input = 800.0 # Valor padrão

if tipo_secao == "Entrada Manual (cm⁴)":
    I_input = st.sidebar.number_input("Momento de Inércia I (cm⁴)", value=800.0, step=10.0)

elif tipo_secao == "Perfil I / H":
    H = st.sidebar.number_input("Altura Total H (mm)", value=200.0, step=10.0)
    B = st.sidebar.number_input("Largura da Mesa B (mm)", value=100.0, step=10.0)
    tw = st.sidebar.number_input("Espessura da Alma tw (mm)", value=6.35, step=1.0)
    tf = st.sidebar.number_input("Espessura da Mesa tf (mm)", value=9.0, step=1.0)
    
    # Conversão de mm para cm
    H_c, B_c, tw_c, tf_c = H/10, B/10, tw/10, tf/10
    # Inércia Eixo X (Maior)
    I_input = (B_c * H_c**3)/12 - ((B_c - tw_c) * (H_c - 2*tf_c)**3)/12
    st.sidebar.info(f"📐 Inércia (Ix) Calculada: **{I_input:.2f} cm⁴**")

elif tipo_secao == "Tubo Retangular / Quadrado":
    H = st.sidebar.number_input("Altura Externa H (mm)", value=100.0, step=10.0)
    B = st.sidebar.number_input("Base Externa B (mm)", value=100.0, step=10.0)
    t = st.sidebar.number_input("Espessura da Parede t (mm)", value=5.0, step=1.0)
    
    H_c, B_c, t_c = H/10, B/10, t/10
    h_c = H_c - 2*t_c # Altura interna
    b_c = B_c - 2*t_c # Base interna
    
    I_input = (B_c * H_c**3)/12 - (b_c * h_c**3)/12
    st.sidebar.info(f"📐 Inércia (Ix) Calculada: **{I_input:.2f} cm⁴**")

elif tipo_secao == "Tubo Circular":
    D = st.sidebar.number_input("Diâmetro Externo D (mm)", value=114.3, step=10.0)
    t = st.sidebar.number_input("Espessura da Parede t (mm)", value=6.02, step=1.0)
    
    D_c, t_c = D/10, t/10
    d_c = D_c - 2*t_c # Diâmetro interno
    
    I_input = (np.pi * (D_c**4 - d_c**4)) / 64
    st.sidebar.info(f"📐 Inércia (I) Calculada: **{I_input:.2f} cm⁴**")

# Conversão final para o motor do SymPy (Sistema Internacional)
E_val = E_input * 1e9  # GPa para Pa
I_val = I_input * 1e-8 # cm4 para m4

# ==========================================
# 3. TELA PRINCIPAL
# ==========================================
st.title("🏗️ Aplicativo de Análise de Vigas")
st.write("Esforço Cortante, Momento Fletor e Linha Elástica via Macaulay")

st.subheader("Configuração da Estrutura")
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

st.write("### Cargas Atuais na Viga")
if len(st.session_state.cargas_salvas) > 0:
    df_exibicao = pd.DataFrame(st.session_state.cargas_salvas)[["Tipo", "Valor Exibição", "Posição Inicial", "Posição Final"]]
    st.dataframe(df_exibicao, use_container_width=True)
    
    if st.button("🗑️ Limpar Todas as Cargas"):
        st.session_state.cargas_salvas = []
        st.rerun()
else:
    st.info("Nenhuma carga inserida. Use o menu lateral para adicionar.")

# ==========================================
# 4. CÁLCULO E PLOTAGEM (BACKEND)
# ==========================================
st.write("---")

if st.button("🚀 Calcular Diagramas") and len(st.session_state.cargas_salvas) > 0:
    
    with st.spinner("Resolvendo estática e integrais (incluindo rampas)..."):
        E, I = sp.symbols('E I')
        R_A, R_B, M_A = sp.symbols('R_A R_B M_A')
        
        viga = Beam(comprimento, E, I)
        
        reacoes_desconhecidas = []
        if tipo_viga == "Biapoiada nas extremidades":
            viga.bc_deflection = [(0, 0), (comprimento, 0)]
            viga.apply_load(R_A, 0, -1)
            viga.apply_load(R_B, comprimento, -1)
            reacoes_desconhecidas = [R_A, R_B]
        elif tipo_viga == "Engastada à Esquerda (x=0)":
            viga.bc_deflection = [(0, 0)]
            viga.bc_slope = [(0, 0)]
            viga.apply_load(R_A, 0, -1)
            viga.apply_load(M_A, 0, -2)
            reacoes_desconhecidas = [R_A, M_A]
        elif tipo_viga == "Biapoiada Personalizada (com ou sem balanço)":
            viga.bc_deflection = [(pos_apoio1, 0), (pos_apoio2, 0)]
            viga.apply_load(R_A, pos_apoio1, -1)
            viga.apply_load(R_B, pos_apoio2, -1)
            reacoes_desconhecidas = [R_A, R_B]

        for carga in st.session_state.cargas_salvas:
            tipo = carga["Tipo"]
            pi = carga["Posição Inicial"]
            pf = carga["Posição Final"]
            
            if tipo == "Pontual":
                v = carga["Valor Numérico"]
                viga.apply_load(v, pi, -1)
            elif tipo == "Momento Fletor":
                v = carga["Valor Numérico"]
                viga.apply_load(v, pi, -2)
            elif tipo == "Distribuída (Uniforme)":
                v = carga["Valor Numérico"]
                viga.apply_load(v, pi, 0)
                if pf < comprimento:
                    viga.apply_load(-v, pf, 0)
            elif tipo == "Triangular / Trapezoidal":
                w1 = carga["w1"]
                w2 = carga["w2"]
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
        eq_flecha_numerica = viga.deflection().subs({E: E_val, I: I_val}).rewrite(sp.Piecewise)
        
        func_V = sp.lambdify(x, eq_cortante, 'numpy')
        func_M = sp.lambdify(x, eq_momento, 'numpy')
        func_y = sp.lambdify(x, eq_flecha_numerica, 'numpy')
        
        x_vetor = np.linspace(0, float(comprimento), 800) 
        
        V_vetor = np.ones_like(x_vetor) * func_V(x_vetor) if np.isscalar(func_V(x_vetor)) else func_V(x_vetor)
        M_vetor = np.ones_like(x_vetor) * func_M(x_vetor) if np.isscalar(func_M(x_vetor)) else func_M(x_vetor)
        y_vetor_m = np.ones_like(x_vetor) * func_y(x_vetor) if np.isscalar(func_y(x_vetor)) else func_y(x_vetor)
        y_vetor_mm = y_vetor_m * 1000 
        
        max_flecha = np.max(np.abs(y_vetor_mm))
        
        col1, col2, col3 = st.columns(3)
        if tipo_viga == "Engastada à Esquerda (x=0)":
            col1.success(f"**R_A (x=0):** {float(reacoes[R_A]):.2f} kN")
            col2.warning(f"**M_A (Engaste):** {float(reacoes[M_A]):.2f} kN.m")
        elif tipo_viga == "Biapoiada Personalizada (com ou sem balanço)":
            col1.success(f"**R_A (x={pos_apoio1}):** {float(reacoes[R_A]):.2f} kN")
            col2.success(f"**R_B (x={pos_apoio2}):** {float(reacoes[R_B]):.2f} kN")
        else:
            col1.success(f"**R_A (x=0):** {float(reacoes[R_A]):.2f} kN")
            col2.success(f"**R_B (x={comprimento}):** {float(reacoes[R_B]):.2f} kN")
            
        col3.error(f"**Flecha Máx:** {max_flecha:.2f} mm") 

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        ax1.plot(x_vetor, V_vetor, color='blue')
        ax1.fill_between(x_vetor, V_vetor, 0, alpha=0.2, color='blue')
        ax1.axhline(0, color='black', linewidth=1)
        ax1.set_title("Diagrama de Esforço Cortante (V)")
        ax1.set_ylabel("Força (kN)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        ax2.plot(x_vetor, M_vetor, color='red')
        ax2.fill_between(x_vetor, M_vetor, 0, alpha=0.2, color='red')
        ax2.axhline(0, color='black', linewidth=1)
        ax2.invert_yaxis() 
        ax2.set_title("Diagrama de Momento Fletor (M)")
        ax2.set_ylabel("Momento (kN.m)")
        ax2.grid(True, linestyle='--', alpha=0.6)

        ax3.plot(x_vetor, y_vetor_mm, color='green')
        ax3.fill_between(x_vetor, y_vetor_mm, 0, alpha=0.2, color='green')
        ax3.axhline(0, color='black', linewidth=1)
        ax3.invert_yaxis() 
        ax3.set_title("Linha Elástica / Deflexão")
        ax3.set_ylabel("Deflexão (mm)")
        ax3.set_xlabel("Posição x ao longo da viga (m)")
        ax3.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        st.pyplot(fig)
