import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, List
import plotly.graph_objects as go


# Configuração da página
st.set_page_config(
    page_title="Analisador de TRL - UNESP AUIN",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Classe principal do analisador TRL
class TRLAnalyzer:
    def __init__(self, model: str = "llm"):
        # self.api_key = api_key        
        self.api_key = st.secrets["ds_apikey"]
        self.model = model
        self.base_url = "https://api.deepseek.com/v1/chat/completions"

        self.trl_definitions = {
            1: {
                "name": "Princípios Básicos Observados",
                "description": "Observação e relato dos princípios científicos básicos",
                "characteristics": [
                    "Pesquisa científica básica iniciada",
                    "Princípios fundamentais observados e relatados",
                    "Ideia/conceito em estágio muito inicial"
                ],
                "color": "#FF6B6B"
            },
            2: {
                "name": "Conceito de Tecnologia Formulado",
                "description": "Formulação do conceito de aplicação tecnológica",
                "characteristics": [
                    "Conceito de aplicação inventado/formulado",
                    "Hipóteses de aplicação prática estabelecidas",
                    "Estudos teóricos para validar conceito"
                ],
                "color": "#FFA726"
            },
            3: {
                "name": "Prova de Conceito Experimental",
                "description": "Prova experimental de conceito-chave",
                "characteristics": [
                    "Prova de conceito em laboratório",
                    "Componentes críticos validados experimentalmente",
                    "Análise analítica e experimental de conceito"
                ],
                "color": "#FFD54F"
            },
            4: {
                "name": "Validação em Ambiente de Laboratório",
                "description": "Validação de componentes em ambiente de laboratório",
                "characteristics": [
                    "Componentes integrados em ambiente de laboratório",
                    "Validação de integração básica",
                    "Testes de funcionamento em condições controladas"
                ],
                "color": "#AED581"
            },
            5: {
                "name": "Validação em Ambiente Relevante",
                "description": "Validação em ambiente relevante/simulado",
                "characteristics": [
                    "Protótipo de baixa fidelidade em ambiente relevante",
                    "Testes em condições simuladas do mundo real",
                    "Validação de interfaces e integração"
                ],
                "color": "#4DB6AC"
            },
            6: {
                "name": "Demonstração em Ambiente Relevante",
                "description": "Demonstração de protótipo em ambiente relevante",
                "characteristics": [
                    "Protótipo de alta fidelidade em ambiente relevante",
                    "Demonstração operacional em ambiente simulado",
                    "Validação de desempenho em condições operacionais"
                ],
                "color": "#4FC3F7"
            },
            7: {
                "name": "Demonstração em Ambiente Operacional",
                "description": "Demonstração de protótipo em ambiente operacional real",
                "characteristics": [
                    "Protótipo operacional em ambiente real",
                    "Testes em condições operacionais reais",
                    "Validação com usuários finais"
                ],
                "color": "#7986CB"
            },
            8: {
                "name": "Sistema Completo Qualificado",
                "description": "Sistema completo qualificado e testado",
                "characteristics": [
                    "Tecnologia demonstrada em forma final",
                    "Testes e qualificação completos",
                    "Pronto para implementação operacional"
                ],
                "color": "#BA68C8"
            },
            9: {
                "name": "Sistema Comprovado em Operação Real",
                "description": "Sistema comprovado em operação real bem-sucedida",
                "characteristics": [
                    "Tecnologia em operação real bem-sucedida",
                    "Feedback do mercado e usuários incorporado",
                    "Operação rotineira e confiável"
                ],
                "color": "#4CAF50"
            }
        }

    def _call_deepseek_api(self, prompt: str, system_message: str = None) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 2000
        }

        try:
            with st.spinner("Conectando e cruzando tipologias LLM e técnicas de pré-processamento..."):
                response = requests.post(self.base_url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"Erro na chamada da conexão: {str(e)}")

    def _create_trl_prompt(self, project_text: str) -> str:
        base_prompt = f"""
        Analise o seguinte resumo de projeto e determine seu Nível de Maturidade Tecnológica (TRL) considerando todos os 9 níveis:

        RESUMO DO PROJETO:
        {project_text}

        INSTRUÇÕES:
        1. Analise cuidadosamente o texto e identifique evidências de maturidade tecnológica
        2. Classifique em um dos 9 níveis TRL (1-9)
        3. Liste as características específicas do projeto que justificam este nível
        4. Forneça recomendações específicas para avançar para o próximo nível TRL

        FORMATO DE RESPOSTA REQUERIDO (JSON):
        {{
            "trl_level": número_entre_1_e_9,
            "confidence_score": número_entre_0_e_1,
            "justification": "explicação_detalhada",
            "matching_characteristics": ["característica1", "característica2", ...],
            "next_steps": ["passo1", "passo2", ...],
            "recommendations": "recomendações_específicas"
        }}
        """

        # Adaptações baseadas no modelo de inferência
        if self.model == "llm_rag":
            base_prompt += "\n\nCONTEXTO ADICIONAL (RAG): Utilize conhecimentos de casos similares para uma análise mais precisa."
        elif self.model == "llm_graphrag":
            base_prompt += "\n\nCONTEXTO ADICIONAL (GraphRAG): Analise relações e dependências entre componentes tecnológicos."
        elif self.model == "kag":
            base_prompt += "\n\nCONTEXTO ADICIONAL (KAG): Aplique conhecimento específico de domínio para avaliação técnica."
        elif self.model == "xlstm":
            base_prompt += "\n\nCONTEXTO ADICIONAL (xLSTM): Utilize memória de longo prazo para análise temporal de maturidade."

        return base_prompt

    def analyze_trl(self, project_text: str) -> Dict:
        system_message = """
        Você é um especialista em avaliação de maturidade tecnológica (TRL). 
        Sua tarefa é analisar resumos de projetos e determinar seu nível TRL com precisão.
        Forneça justificativas detalhadas baseadas em evidências concretas do texto.
        Seja objetivo e técnico em sua análise.
        """

        prompt = self._create_trl_prompt(project_text)

        try:
            with st.spinner("🔍 Analisando maturidade tecnológica..."):
                response = self._call_deepseek_api(prompt, system_message)

            # Extrai JSON da resposta
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            result = json.loads(json_str)

            # Adiciona informações adicionais sobre o nível TRL
            trl_level = result["trl_level"]
            result["trl_definition"] = self.trl_definitions[trl_level]
            result["analysis_timestamp"] = datetime.now().isoformat()
            result["model_used"] = self.model

            return result

        except Exception as e:
            return {
                "error": str(e),
                "trl_level": None,
                "confidence_score": 0.0,
                "justification": "Erro na análise",
                "matching_characteristics": [],
                "next_steps": []
            }


# Funções auxiliares para visualização
def create_trl_gauge(trl_level: int, confidence_score: float):
    """Cria um gauge visual para o nível TRL"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=trl_level,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"NÍVEL TRL - {confidence_score * 100:.1f}% Confiança"},
        delta={'reference': trl_level - 1, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [1, 9], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [1, 3], 'color': 'lightcoral'},
                {'range': [3, 6], 'color': 'lightyellow'},
                {'range': [6, 9], 'color': 'lightgreen'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': trl_level}}))

    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def create_trl_roadmap(current_level: int, next_steps: List[str]):
    st.subheader("🎯 Roadmap de Evolução TRL")

    cols = st.columns(3)
    levels_display = [
        (1, 3, "Pesquisa Básica"),
        (4, 6, "Desenvolvimento"),
        (7, 9, "Implementação")
    ]

    for idx, (start, end, phase) in enumerate(levels_display):
        with cols[idx]:
            st.markdown(f"**{phase} (TRL {start}-{end})**")
            for level in range(start, end + 1):
                if level == current_level:
                    st.markdown(f"🔸 **TRL {level} - ATUAL**")
                elif level < current_level:
                    st.markdown(f"✅ TRL {level} - Concluído")
                else:
                    st.markdown(f"⏳ TRL {level} - Futuro")


def display_analysis_results(result: Dict):
    if "error" in result:
        st.error(f"❌ Erro na análise: {result['error']}")
        return

    trl_level = result["trl_level"]
    trl_info = result["trl_definition"]
    confidence = result.get("confidence_score", 0)

    # Header com resultado principal
    col1, col2 = st.columns([1, 2])

    with col1:
        create_trl_gauge(trl_level, confidence)

    with col2:
        st.markdown(f"""
        ### 📊 Resultado da Análise
        **Nível TRL:** `{trl_level} - {trl_info['name']}`  
        **Confiança:** `{confidence * 100:.1f}%`  
        **Modelo Utilizado:** `{result.get('model_used', 'N/A')}`  
        **Data da Análise:** `{datetime.fromisoformat(result['analysis_timestamp']).strftime('%d/%m/%Y %H:%M')}`
        """)

    # Descrição do nível TRL
    st.markdown("---")
    st.subheader("📋 Descrição do Nível TRL")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{trl_info['name']}**")
        st.info(trl_info['description'])

        st.markdown("**Características Típicas:**")
        for char in trl_info['characteristics']:
            st.markdown(f"• {char}")

    with col2:
        st.markdown("**Características Identificadas:**")
        for char in result.get('matching_characteristics', []):
            st.markdown(f"✅ {char}")

    # Justificativa
    st.markdown("---")
    st.subheader("🎯 Justificativa da Análise")
    st.write(result.get('justification', 'Não disponível'))

    # Próximos passos
    st.markdown("---")
    st.subheader("🚀 Próximos Passos Recomendados")

    next_steps = result.get('next_steps', [])
    if next_steps:
        for i, step in enumerate(next_steps, 1):
            st.markdown(f"{i}. {step}")
    else:
        st.warning("Nenhum próximo passo identificado.")

    # Recomendações específicas
    if 'recommendations' in result and result['recommendations']:
        st.markdown("---")
        st.subheader("💡 Recomendações Específicas")
        st.write(result['recommendations'])

    # Roadmap visual
    create_trl_roadmap(trl_level, next_steps)


# Interface principal do Streamlit
def main():
    # Sidebar
    st.sidebar.title("Analisador de TRL")
    st.sidebar.title("UNESP AUIN - v1.1")
    st.sidebar.markdown("---")

    # Seleção do modelo
    st.sidebar.subheader("🤖 Modelo de Inferência")
    model_options = {
        "LLM (Padrão)": "llm",
        "LLM + RAG": "llm_rag",
        "LLM + GraphRAG": "llm_graphrag",
        "KAG": "kag",
        "xLSTM": "xlstm"
    }

    selected_model = st.sidebar.selectbox(
        "Selecione o modelo:",
        options=list(model_options.keys()),
        index=0
    )
    model_key = model_options[selected_model]

    # Informações na sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Sobre o TRL")
    st.sidebar.markdown("""
    **Technology Readiness Level (TRL)** é uma métrica para avaliar 
    a maturidade de uma tecnologia durante sua fase de desenvolvimento.

    - **TRL 1-3**: Pesquisa básica
    - **TRL 4-6**: Desenvolvimento tecnológico  
    - **TRL 7-9**: Implementação e operação
    """)

    # Área principal
    st.title("Analisador de Maturidade Tecnológica (TRL) - UNESP AUIN")
    st.markdown("Analise o nível de maturidade tecnológica do seu projeto usando IA")

    # Entrada do texto do projeto
    st.subheader("Insira o Resumo do Projeto")

    input_method = st.radio(
        "Método de entrada:",
        ["Texto direto", "Upload de arquivo"],
        horizontal=True
    )

    project_text = ""

    if input_method == "Texto direto":
        project_text = st.text_area(
            "Cole o resumo do projeto aqui:",
            height=200,
            placeholder="Descreva o projeto, estágio de desenvolvimento, testes realizados, resultados obtidos, próximos passos..."
        )
    else:
        uploaded_file = st.file_uploader(
            "Faça upload de um arquivo de texto",
            type=['txt', 'md', 'docx', 'pdf']
        )
        if uploaded_file is not None:
            if uploaded_file.type == "text/plain":
                project_text = str(uploaded_file.read(), "utf-8")
            else:
                st.warning("Formatos .docx e .pdf em desenvolvimento. Use arquivo .txt por enquanto.")

    # Botão de análise
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "Analisar TRL do Projeto",
            type="primary",
            use_container_width=True,
            # disabled=not (api_key and project_text.strip())
            disabled = not (project_text.strip())
        )

    # Processamento da análise
    if analyze_button and project_text.strip():
        analyzer = TRLAnalyzer(model_key)
        with st.spinner("Processando análise com IA..."):
            result = analyzer.analyze_trl(project_text)

        display_analysis_results(result)

        st.markdown("---")
        st.subheader("Exportar Resultados")

        col1, col2 = st.columns(2)

        with col1:
            # Download JSON
            json_str = json.dumps(result, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 Baixar JSON Completo",
                data=json_str,
                file_name=f"trl_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        with col2:
            # Download Resumo
            summary = f"""
            RELATÓRIO DE ANÁLISE TRL
            Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}
            Modelo: {selected_model}

            RESULTADO:
            TRL Nível: {result.get('trl_level', 'N/A')} - {result.get('trl_definition', {}).get('name', 'N/A')}
            Confiança: {result.get('confidence_score', 0) * 100:.1f}%

            PRÓXIMOS PASSOS:
            {chr(10).join(f'- {step}' for step in result.get('next_steps', []))}
            """

            st.download_button(
                label="Baixar Resumo (TXT)",
                data=summary,
                file_name=f"trl_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

    # Exemplo de uso
    elif not project_text:
        st.markdown("---")
        st.subheader("Exemplo de Uso")

        with st.expander("Clique para ver um exemplo"):
            example_text = """
            Desenvolvimento de um novo sistema de baterias de estado sólido para veículos elétricos.

            STATUS ATUAL:
            - Completamos simulações computacionais validadas experimentalmente
            - Demonstramos os princípios fundamentais em ambiente de laboratório
            - Desenvolvemos protótipos em escala de bancada com desempenho promissor
            - Realizamos testes de ciclo de vida preliminares (100+ ciclos)
            - Estabelecemos parceria com fabricante de materiais

            PRÓXIMOS PASSOS:
            - Otimização da composição dos materiais
            - Testes de estabilidade térmica e segurança
            - Scale-up para protótipos de célula completa
            - Validação em condições simuladas de veículo elétrico
            """

            st.text_area("Texto de exemplo:", value=example_text, height=200)

            if st.button("Usar Exemplo", key="example"):
                st.session_state.example_used = example_text
                st.rerun()


if __name__ == "__main__":
    main()
