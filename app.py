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

tipo_carga = st.sidebar.selectbox("Tipo de Carga", ["Pontual", "Momento Fletor", "Distribuída (Uniforme)"])
valor = st.sidebar.number_input("Magnitude (kN ou kN.m)", value=-10.0, step=1.0, 
                                help="Use valores negativos para cargas apontando para baixo.")
posicao_i = st.sidebar.number_input("Posição Inicial - x (m)", min_value=0.0, value=2.0, step=0.5)

posicao_f = None
if tipo_carga == "Distribuída (Uniforme)":
    posicao_f = st.sidebar.number_input("Posição Final - x (m)", min_value=posicao_i, value=posicao_i+2.0, step=0.5)

if st.sidebar.button("➕ Adicionar Carga"):
    nova_carga = {
        "Tipo": tipo_carga,
        "Valor": valor,
        "Posição Inicial": posicao_i,
        "Posição Final": posicao_f
    }
    st.session_state.cargas_salvas.append(nova_carga)
    st.sidebar.success("Carga adicionada!")

# -- NOVA SEÇÃO: PROPRIEDADES DA VIGA --
st.sidebar.markdown("---")
st.sidebar.header("2. Propriedades da Seção")
st.sidebar.write("Necessárias para o cálculo da flecha.")
E_input = st.sidebar.number_input("Módulo de Elasticidade E (GPa)", value=200.0, step=10.0, help="Aço padrão = ~200 GPa")
I_input = st.sidebar.number_input("Inércia I (cm⁴)", value=800.0, step=10.0, help="Inércia da seção transversal")

# Conversão para as unidades base do Sistema Internacional (Pascal e Metros)
E_val = E_input * 1e9  # GPa para Pa
I_val = I_input * 1e-8 # cm4 para m4

# ==========================================
# 3. TELA PRINCIPAL (VISUALIZAÇÃO DE DADOS)
# ==========================================
st.title("🏗️ Aplicativo de Análise de Vigas")
st.write("Esforço Cortante, Momento Fletor e Linha Elástica")

comprimento = st.number_input("Comprimento total da viga biapoiada (m)", min_value=1.0, value=10.0, step=1.0)

st.write("### Cargas Atuais na Viga")
if len(st.session_state.cargas_salvas) > 0:
    df_cargas = pd.DataFrame(st.session_state.cargas_salvas)
    st.dataframe(df_cargas, use_container_width=True)
    
    if st.button("🗑️ Limpar Todas as Cargas"):
        st.session_state.cargas_salvas = []
        st.rerun()
else:
    st.info("Nenhuma carga inserida. Use o menu lateral para adicionar cargas.")

# ==========================================
# 4. CÁLCULO E PLOTAGEM (BACKEND)
# ==========================================
st.write("---")

if st.button("🚀 Calcular Diagramas") and len(st.session_state.cargas_salvas) > 0:
    
    with st.spinner("Resolvendo integrais e calculando deslocamentos..."):
        E, I = sp.symbols('E I')
        R_A, R_B = sp.symbols('R_A R_B')
        viga = Beam(comprimento, E, I)
        
        viga.bc_deflection = [(0, 0), (comprimento, 0)]
        viga.apply_load(R_A, 0, -1)
        viga.apply_load(R_B, comprimento, -1)
        
        for carga in st.session_state.cargas_salvas:
            v = carga["Valor"]
            pi = carga["Posição Inicial"]
            pf = carga["Posição Final"]
            
            if carga["Tipo"] == "Pontual":
                viga.apply_load(v, pi, -1)
            elif carga["Tipo"] == "Momento Fletor":
                viga.apply_load(v, pi, -2)
            elif carga["Tipo"] == "Distribuída (Uniforme)":
                viga.apply_load(v, pi, 0)
                if pf < comprimento:
                    viga.apply_load(-v, pf, 0)
        
        viga.solve_for_reaction_loads(R_A, R_B)
        reacoes = viga.reaction_loads
        
        # --- PREPARAÇÃO DAS EQUAÇÕES MATEMÁTICAS ---
        x = sp.Symbol('x')
        eq_cortante = viga.shear_force().rewrite(sp.Piecewise)
        eq_momento = viga.bending_moment().rewrite(sp.Piecewise)
        
        # NOVA EQUAÇÃO: Linha Elástica (Flecha)
        # Precisamos substituir E e I pelos valores reais para termos números exatos
        eq_flecha_simbolica = viga.deflection()
        eq_flecha_numerica = eq_flecha_simbolica.subs({E: E_val, I: I_val}).rewrite(sp.Piecewise)
        
        func_V = sp.lambdify(x, eq_cortante, 'numpy')
        func_M = sp.lambdify(x, eq_momento, 'numpy')
        func_y = sp.lambdify(x, eq_flecha_numerica, 'numpy') # Função da flecha
        
        # Vetorização
        x_vetor = np.linspace(0, float(comprimento), 500)
        
        V_vetor = np.ones_like(x_vetor) * func_V(x_vetor) if np.isscalar(func_V(x_vetor)) else func_V(x_vetor)
        M_vetor = np.ones_like(x_vetor) * func_M(x_vetor) if np.isscalar(func_M(x_vetor)) else func_M(x_vetor)
        
        # O resultado do SymPy é em Metros. Multiplicamos por 1000 para Milímetros.
        # Também invertemos o sinal para o gráfico plotar para baixo a deflexão negativa (convenção)
        y_vetor_m = np.ones_like(x_vetor) * func_y(x_vetor) if np.isscalar(func_y(x_vetor)) else func_y(x_vetor)
        y_vetor_mm = y_vetor_m * 1000 
        
        # Extrair valores máximos
        max_flecha = np.max(np.abs(y_vetor_mm))
        
        # Mostra resultados numéricos em cards
        col1, col2, col3 = st.columns(3)
        col1.success(f"**R_A (x=0):** {float(reacoes[R_A]):.2f} kN")
        col2.success(f"**R_B (x={comprimento}):** {float(reacoes[R_B]):.2f} kN")
        col3.error(f"**Flecha Máxima:** {max_flecha:.2f} mm") # Destacado em vermelho (error block)
        
        # --- DESENHO COM MATPLOTLIB (Agora com 3 gráficos) ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Gráfico 1: Cortante
        ax1.plot(x_vetor, V_vetor, color='blue')
        ax1.fill_between(x_vetor, V_vetor, 0, alpha=0.2, color='blue')
        ax1.axhline(0, color='black', linewidth=1)
        ax1.set_title("Diagrama de Esforço Cortante (V)")
        ax1.set_ylabel("Força (kN)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Gráfico 2: Momento
        ax2.plot(x_vetor, M_vetor, color='red')
        ax2.fill_between(x_vetor, M_vetor, 0, alpha=0.2, color='red')
        ax2.axhline(0, color='black', linewidth=1)
        ax2.invert_yaxis() 
        ax2.set_title("Diagrama de Momento Fletor (M)")
        ax2.set_ylabel("Momento (kN.m)")
        ax2.grid(True, linestyle='--', alpha=0.6)

        # Gráfico 3: Linha Elástica (NOVO)
        ax3.plot(x_vetor, y_vetor_mm, color='green')
        ax3.fill_between(x_vetor, y_vetor_mm, 0, alpha=0.2, color='green')
        ax3.axhline(0, color='black', linewidth=1)
        ax3.invert_yaxis() # Inverte para que a "barriga" da viga vá para baixo
        ax3.set_title("Linha Elástica / Deflexão")
        ax3.set_ylabel("Deflexão (mm)")
        ax3.set_xlabel("Posição x ao longo da viga (m)")
        ax3.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        st.pyplot(fig)