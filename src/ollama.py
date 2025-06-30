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



class OllamaJobClassifier:
    def __init__(self, model_name="llama3:8b", base_url="http://localhost:11434"):
        """
        Clasificador usando Ollama con modelos locales
        
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
    
    def create_classification_prompt(self, job_titles: List[str]) -> str:
        """
        Crea prompt optimizado para clasificación
        """
        prompt = f"""Analiza estos títulos de trabajo y clasifícalos EXACTAMENTE en las categorías especificadas.

TÍTULOS A CLASIFICAR:
{json.dumps(job_titles, indent=2)}

CATEGORÍAS DE SENIORITY:
- Entry Level: 0-2 años experiencia, junior, assistant, intern, trainee, copywriter
- Mid Level: 3-7 años experiencia, specialist, analyst, coordinator  
- Senior Level: 8-15 años experiencia, senior, lead, principal
- Executive: 15+ años experiencia, director, VP, CEO, CTO, manager

CATEGORÍAS FUNCIONALES:
- Technology: software, developer, engineer, IT, data scientist, programmer,
- Sales_Marketing: sales, marketing, account, business development, social media, brand
- Finance: financial, accounting, audit, treasury, investment
- HR: human resources, recruiter, talent, training
- Operations: operations, project manager, supply chain, logistics
- Research: scientist, researcher, R&D, clinical
- Customer_Service: customer service, support, help desk
- Product: product manager, product owner
- Other: cualquier otro no clasificable

FORMATO DE RESPUESTA (JSON válido únicamente):
{{
  "results": [
    {{
      "job_title": "título exacto del input",
      "seniority_level": "una de las 4 categorías de seniority. Si dice junior, se condidera Entry Level, si contiene senior y no tenes dudas, consideralo Senior Level. Los directores son ejecutivos, los proyect managers son senior level",
      "functional_area": "una de las 9 categorías funcionales, si contiene HR es HR, si contiene product es product",
      "salary_expectation": "Low/Medium/High/Very_High. Limitate a estas 4 categorías.", 
      "confidence": 0.95
    }}
  ]
}}

Responde SOLO con el JSON, sin texto adicional."""

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
                "num_predict": 2000,  # Máximo tokens de respuesta
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, timeout=120)
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
    
    def parse_classification_response(self, response: str, original_titles: List[str]) -> List[Dict]:
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
            for title in original_titles:
                fallback_results.append({
                    "job_title": title,
                    "seniority_level": "Mid Level",
                    "functional_area": "Other", 
                    "salary_expectation": "Medium",
                    "confidence": 0.5
                })
            return fallback_results
    
    def classify_job_titles(self, job_titles: List[str], batch_size: int = 5) -> List[Dict]:
        """
        Clasifica títulos de trabajo en lotes
        """
        all_results = []
        
        # Procesar en lotes pequeños para evitar timeouts
        for i in range(0, len(job_titles), batch_size):
            batch = job_titles[i:i + batch_size]
            batch_clean = [str(title) if pd.notna(title) else 'Unknown' for title in batch]
            
            print(f"Procesando lote {i//batch_size + 1}/{(len(job_titles)-1)//batch_size + 1}: {len(batch)} títulos")
            
            # Crear prompt y consultar
            prompt = self.create_classification_prompt(batch_clean)
            response = self.query_ollama(prompt)
            
            if response:
                batch_results = self.parse_classification_response(response, batch_clean)
                all_results.extend(batch_results)
                print(f"  ✅ Clasificados {len(batch_results)} títulos")
            else:
                print(f"  ❌ Error en lote {i//batch_size + 1}")
                # Agregar fallbacks
                for title in batch_clean:
                    all_results.append({
                        "job_title": title,
                        "seniority_level": "Mid Level",
                        "functional_area": "Other",
                        "salary_expectation": "Medium", 
                        "confidence": 0.0
                    })
            
            # Pausa entre lotes
            time.sleep(1)
        
        return all_results
    
    def process_dataframe(self, df: pd.DataFrame, job_title_col: str = 'job_title') -> pd.DataFrame:
        """
        Procesa DataFrame completo
        """
        # Obtener títulos únicos para reducir consultas
        unique_titles = df[job_title_col].dropna().unique().tolist()
        print(f"Clasificando {len(unique_titles)} títulos únicos con {self.model_name}...")
        
        # Clasificar títulos únicos
        classifications = self.classify_job_titles(unique_titles)
        
        # Crear mapeo
        classification_map = {}
        for item in classifications:
            classification_map[item['job_title']] = item
        
        # Aplicar al DataFrame
        df_result = df.copy()
        
        df_result['llm_seniority'] = df_result[job_title_col].map(
            lambda x: classification_map.get(str(x) if pd.notna(x) else 'Unknown', {}).get('seniority_level', 'Unknown')
        )
        df_result['llm_functional_area'] = df_result[job_title_col].map(
            lambda x: classification_map.get(str(x) if pd.notna(x) else 'Unknown', {}).get('functional_area', 'Other')
        )
        df_result['llm_salary_expectation'] = df_result[job_title_col].map(
            lambda x: classification_map.get(str(x) if pd.notna(x) else 'Unknown', {}).get('salary_expectation', 'Medium')
        )
        df_result['llm_confidence'] = df_result[job_title_col].map(
            lambda x: classification_map.get(str(x) if pd.notna(x) else 'Unknown', {}).get('confidence', 0.0)
        )
        
        return df_result

def run_classification(df, job_title_column='job_title', table_name='df_ollama', model_name="llama3:8b"):
    """
    Función para ejecutar la clasificación de trabajos con Ollama
    
    Args:
        df: DataFrame a procesar
        job_title_column: Nombre de la columna que contiene los títulos de trabajo
        table_name: Nombre de la tabla a crear en DuckDB
        model_name: Modelo de Ollama a utilizar
    
    Returns:
        pd.DataFrame: DataFrame con las clasificaciones agregadas
    """
    print(f"=== CLASIFICACIÓN CON OLLAMA - Modelo: {model_name} ===")
    
    # Crear instancia del clasificador
    ollama_classifier = OllamaJobClassifier(model_name=model_name)
    
    # Procesar DataFrame
    df_result = ollama_classifier.process_dataframe(df, job_title_column)
    
    # Análisis de resultados
    print("\n=== RESULTADOS OLLAMA ===")
    print("Distribución Seniority:")
    print(df_result['llm_seniority'].value_counts())
    
    print("\nDistribución Área Funcional:")
    print(df_result['llm_functional_area'].value_counts())
    
    print("\nSalario promedio por categoría LLM:")
    if 'salary' in df_result.columns:
        salary_by_llm_seniority = df_result.groupby('llm_seniority')['salary'].mean().sort_values(ascending=False)
        print(salary_by_llm_seniority.round(0))
    
    print(f"\nConfianza promedio: {df_result['llm_confidence'].mean():.2f}")
    
    # Comparar con métodos anteriores si están disponibles
    if 'pattern_seniority' in df_result.columns:
        print("\nComparación de métodos:")
        comparison = pd.crosstab(df_result['llm_seniority'], df_result['pattern_seniority'], margins=True)
        print(comparison)
    
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
    print("Ejecutando clasificación por defecto...")
    df_ollama = run_classification(df_completo, 'job_title', 'df_ollama')