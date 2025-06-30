# Instalación requerida:
# 1. Instalar Ollama: https://ollama.ai/
# 2. Descargar modelo: ollama pull llama3:8b
# 3. pip install requests

import requests
import json
import pandas as pd
from typing import List, Dict
import time
import duckdb
import os
from dotenv import load_dotenv 

load_dotenv()
DB_NAME = os.getenv("DB_NAME")
DB_PATH = os.getenv("DB_PATH")
DATA_PATH = os.getenv("DATA_PATH")

class OllamaDescriptionAnalyzer:
    def __init__(self, model_name="llama3:8b", base_url="http://localhost:11434"):
        """
        Analizador de descripciones de trabajo usando Ollama con modelos locales
        
        Modelos recomendados:
        - llama3:8b: Excelente balance (4.7GB VRAM)
        - llama3:13b: Mejor calidad (7.4GB VRAM) 
        - codellama:7b: Especializado en código (3.8GB VRAM)
        - mistral:7b: Rápido y eficiente (4.1GB VRAM)
        - llama3:70b: Máxima calidad (40GB VRAM - solo para GPUs potentes)
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Verificar que Ollama esté corriendo
        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                available_models = [m['name'] for m in models['models']]
                print(f"Ollama conectado. Modelos disponibles: {available_models}")
                if model_name not in available_models:
                    print(f"⚠️  Modelo {model_name} no encontrado. Descárgalo con: ollama pull {model_name}")
                else:
                    print(f"✅ Usando modelo: {model_name}")
            else:
                print("❌ Ollama no está corriendo. Ejecuta: ollama serve")
        except requests.exceptions.RequestException:
            print("❌ No se puede conectar a Ollama. Asegúrate de que esté instalado y corriendo.")
    
    def create_analysis_prompt(self, descriptions: List[str]) -> str:
        """
        Creates optimized prompt for job description analysis
        """
        prompt = f"""Analyze these job descriptions and extract information EXACTLY in the specified categories.

DESCRIPTIONS TO ANALYZE:
{json.dumps(descriptions, indent=2)}

SENIORITY CATEGORIES:
- Entry Level: 0-2 years experience, junior, recent graduate, trainee, assistant, copywriter
- Mid Level: 3-7 years experience, specialist, analyst, coordinator, any years
- Senior Level: 8-15 years experience, senior, lead, principal, extensive experience, many years
- Executive: 15+ years experience, director, VP, CEO, CTO, manager, leadership roles, executive, head of department

FUNCTIONAL CATEGORIES:
- Technology: software, developer, engineer, IT, data scientist, programmer, technical
- Sales_Marketing: sales, marketing, account, business development, social media, brand
- Finance: financial, accounting, audit, treasury, investment, budgeting
- HR: human resources, recruiter, talent, training, people management
- Operations: operations, project manager, supply chain, logistics, process improvement
- Research: scientist, researcher, R&D, clinical, analysis, studies
- Customer_Service: customer service, support, help desk, client relations
- Product: product manager, product owner, product development
- Other: any other not classifiable

RESPONSIBILITY LEVEL:
- Low: Individual tasks, following instructions, contributor role
- Medium: Independent projects, limited decision making
- High: Project leadership, strategic decisions
- Very_High: Organizational responsibility, company-wide impact

ADDITIONAL VALIDATION:
- If you assign Executive or Director, there must be clear textual evidence of that level
- Prefer more conservative levels when there is ambiguity
- Years of experience must match the assigned level

RESPONSE FORMAT (valid JSON only):
{{
  "results": [
    {{
      "seniority_level": "one of the 4 seniority categories. If say Junior is entry level, if says director, is director",
      "functional_area": "one of the 9 functional categories. HR and human resources are the same, is HR"",
      "salary_expectation": "Low/Medium/High/Very_High. If dertermine high or very_high, please valide whith seniority_level ",
      "management_level": "None/Team_Lead/Manager/Director. Only this levels",
      "responsibility_level": "Low/Medium/High/Very_High. If dertermine high or very_high, please valide whith seniority_level",
      "years_experience": "number extracted from description or estimated",
      "has_leadership": true/false,
      "has_strategic_role": true/false,
      "has_budget_responsibility": true/false,
      "has_client_facing": true/false,
      "requires_technical_skills": true/false,
      "education_level": "High_School/Bachelor/Master/PhD/Other",
      "key_skills": ["skill1", "skill2", "skill3"],
      "validation_notes": "brief explanation of why you assigned these levels",
      "confidence": 0.95
    }}
  ]
}}

Respond with ONLY the JSON, no additional text."""

        return prompt
    
    def query_ollama(self, prompt: str, max_retries: int = 3) -> str:
        """
        Consulta al modelo Ollama
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Baja temperatura para consistencia
                "top_p": 0.9,
                "num_predict": 3000,  # Máximo tokens de respuesta (aumentado para más info)
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, timeout=180)
                if response.status_code == 200:
                    result = response.json()
                    return result['response'].strip()
                else:
                    print(f"Error HTTP {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"Intento {attempt + 1} falló: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return None
    
    def parse_analysis_response(self, response: str, original_descriptions: List[str]) -> List[Dict]:
        """
        Parsea la respuesta del LLM
        """
        try:
            # Limpiar respuesta
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            # Parsear JSON
            parsed = json.loads(response)
            
            if 'results' in parsed:
                return parsed['results']
            else:
                return parsed if isinstance(parsed, list) else [parsed]
        
        except json.JSONDecodeError as e:
            print(f"Error parseando JSON: {e}")
            print(f"Respuesta recibida: {response[:500]}...")
            
            # Fallback: crear respuestas por defecto
            fallback_results = []
            for i, desc in enumerate(original_descriptions):
                fallback_results.append({
                    "description_id": i+1,
                    "seniority_level": "Mid Level",
                    "functional_area": "Other", 
                    "salary_expectation": "Medium",
                    "management_level": "None",
                    "responsibility_level": "Medium",
                    "years_experience": "Unknown",
                    "has_leadership": False,
                    "has_strategic_role": False,
                    "has_budget_responsibility": False,
                    "has_client_facing": False,
                    "requires_technical_skills": False,
                    "education_level": "Other",
                    "key_skills": [],
                    "validation_notes": "Fallback due to parsing error",
                    "confidence": 0.5
                })
            return fallback_results
    
    def analyze_descriptions(self, descriptions: List[str], batch_size: int = 3) -> List[Dict]:
        """
        Analiza descripciones de trabajo en lotes
        """
        all_results = []
        
        # Procesar en lotes pequeños para evitar timeouts (reducido para descripciones largas)
        for i in range(0, len(descriptions), batch_size):
            batch = descriptions[i:i + batch_size]
            batch_clean = [str(desc) if pd.notna(desc) else 'No description available' for desc in batch]
            
            print(f"Procesando lote {i//batch_size + 1}/{(len(descriptions)-1)//batch_size + 1}: {len(batch)} descripciones")
            
            # Crear prompt y consultar
            prompt = self.create_analysis_prompt(batch_clean)
            response = self.query_ollama(prompt)
            
            if response:
                batch_results = self.parse_analysis_response(response, batch_clean)
                all_results.extend(batch_results)
                print(f"  ✅ Analizadas {len(batch_results)} descripciones")
            else:
                print(f"  ❌ Error en lote {i//batch_size + 1}")
                # Agregar fallbacks
                for j, desc in enumerate(batch_clean):
                    all_results.append({
                        "description_id": str(1 + j),
                        "seniority_level": "Mid Level",
                        "functional_area": "Other",
                        "salary_expectation": "Medium",
                        "management_level": "None",
                        "responsibility_level": "Medium",
                        "years_experience": "Unknown",
                        "has_leadership": False,
                        "has_strategic_role": False,
                        "has_budget_responsibility": False,
                        "has_client_facing": False,
                        "requires_technical_skills": False,
                        "education_level": "Other",
                        "key_skills": [],
                        "validation_notes": "Fallback due to query error",
                        "confidence": 0.0
                    })
            
            # Pausa entre lotes
            time.sleep(2)
        
        return all_results
    
    def process_dataframe(self, df: pd.DataFrame, description_col: str = 'description') -> pd.DataFrame:
        """
        Procesa DataFrame completo
        """
        # Obtener descripciones (no necesariamente únicas ya que cada una puede tener matices)
        descriptions = df[description_col].dropna().tolist()
        print(f"Analizando {len(descriptions)} descripciones con {self.model_name}...")
        
        # Analizar descripciones
        analyses = self.analyze_descriptions(descriptions)
        
        # Aplicar al DataFrame
        df_result = df.copy()
        
        # Mapear resultados por índice
        valid_indices = df[description_col].dropna().index.tolist()
        
        for i, analysis in enumerate(analyses):
            if i < len(valid_indices):
                idx = valid_indices[i]
                df_result.loc[idx, 'llm_seniority'] = analysis.get('seniority_level', 'Unknown')
                df_result.loc[idx, 'llm_functional_area'] = analysis.get('functional_area', 'Other')
                df_result.loc[idx, 'llm_salary_expectation'] = analysis.get('salary_expectation', 'Medium')
                df_result.loc[idx, 'llm_management_level'] = analysis.get('management_level', 'None')
                df_result.loc[idx, 'llm_responsibility_level'] = analysis.get('responsibility_level', 'Medium')
                df_result.loc[idx, 'llm_years_experience'] = analysis.get('years_experience', 'Unknown')
                df_result.loc[idx, 'llm_has_leadership'] = analysis.get('has_leadership', False)
                df_result.loc[idx, 'llm_has_strategic_role'] = analysis.get('has_strategic_role', False)
                df_result.loc[idx, 'llm_has_budget_responsibility'] = analysis.get('has_budget_responsibility', False)
                df_result.loc[idx, 'llm_has_client_facing'] = analysis.get('has_client_facing', False)
                df_result.loc[idx, 'llm_requires_technical_skills'] = analysis.get('requires_technical_skills', False)
                df_result.loc[idx, 'llm_education_level'] = analysis.get('education_level', 'Other')
                df_result.loc[idx, 'llm_key_skills'] = ', '.join(analysis.get('key_skills', []))
                df_result.loc[idx, 'llm_validation_notes'] = analysis.get('validation_notes', '')
                df_result.loc[idx, 'llm_confidence'] = analysis.get('confidence', 0.0)
        
        # Llenar valores faltantes
        llm_columns = [col for col in df_result.columns if col.startswith('llm_')]
        for col in llm_columns:
            if col not in df_result.columns:
                if 'has_' in col:
                    df_result[col] = False
                elif col == 'llm_confidence':
                    df_result[col] = 0.0
                else:
                    df_result[col] = 'Unknown'
        
        return df_result

def run_description_analysis(df, description_column='description', table_name='df_descriptions_analyzed', model_name="llama3:8b"):
    """
    Función para ejecutar el análisis de descripciones de trabajo con Ollama
    
    Args:
        df: DataFrame a procesar
        description_column: Nombre de la columna que contiene las descripciones
        table_name: Nombre de la tabla a crear en DuckDB
        model_name: Modelo de Ollama a utilizar
    
    Returns:
        pd.DataFrame: DataFrame con los análisis agregados
    """
    print(f"=== ANÁLISIS DE DESCRIPCIONES CON OLLAMA - Modelo: {model_name} ===")
    
    # Crear instancia del analizador
    ollama_analyzer = OllamaDescriptionAnalyzer(model_name=model_name)
    
    # Procesar DataFrame
    df_result = ollama_analyzer.process_dataframe(df, description_column)
    
    # Análisis de resultados
    print("\n=== RESULTADOS ANÁLISIS OLLAMA ===")
    print("Distribución Seniority:")
    print(df_result['llm_seniority'].value_counts())
    
    print("\nDistribución Área Funcional:")
    print(df_result['llm_functional_area'].value_counts())
    
    print("\nDistribución Nivel de Gestión:")
    print(df_result['llm_management_level'].value_counts())
    
    print("\nDistribución Nivel de Responsabilidad:")
    print(df_result['llm_responsibility_level'].value_counts())
    
    print("\nCaracterísticas de Liderazgo:")
    print(f"Con liderazgo: {df_result['llm_has_leadership'].sum()}")
    print(f"Rol estratégico: {df_result['llm_has_strategic_role'].sum()}")
    print(f"Responsabilidad presupuestaria: {df_result['llm_has_budget_responsibility'].sum()}")
    print(f"Cara al cliente: {df_result['llm_has_client_facing'].sum()}")
    print(f"Requiere habilidades técnicas: {df_result['llm_requires_technical_skills'].sum()}")
    
    print("\nSalario promedio por categoría LLM:")
    if 'salary' in df_result.columns:
        salary_by_llm_seniority = df_result.groupby('llm_seniority')['salary'].mean().sort_values(ascending=False)
        print(salary_by_llm_seniority.round(0))
    
    print(f"\nConfianza promedio: {df_result['llm_confidence'].mean():.2f}")
    
    # Guardar resultados
    csv_filename = f"{table_name}.csv"
    df_result.to_csv(os.path.join(DATA_PATH, csv_filename), index=False)
    
    # Guardar en DuckDB
    con = duckdb.connect(f"{DB_PATH}/{DB_NAME}")
    con.register(f'{table_name}_temp', df_result)
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {table_name}_temp")
    con.close()
    
    print(f"\n✅ Resultados guardados en {csv_filename} y tabla DuckDB '{table_name}'")
    
    return df_result

if __name__ == "__main__":
    # Este código solo se ejecuta si el archivo se ejecuta directamente
    con = duckdb.connect(f"{DB_PATH}/{DB_NAME}")
    df_completo = con.execute("SELECT * FROM df_completo").df()
    con.close()
    print("Ejecutando análisis de descripciones por defecto...")
    df_analyzed = run_description_analysis(df_completo, 'description', 'df_descriptions_analyzed')