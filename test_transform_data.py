#!/usr/bin/env python3
"""
Script de transformation pour l'étude de brainstorming avec IA
Auteur: Étude de recherche brainstorming
Date: Septembre 2025

Ce script transforme le dataset donnee_test.csv en extrayant et normalisant 
les données JSON imbriquées pour faciliter l'analyse.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_output_directory():
    """Créer le répertoire de sortie si nécessaire"""
    output_dir = Path('data/transform')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def clean_and_parse_json(json_str):
    """
    Nettoyer et parser une chaîne JSON avec des guillemets échappés complexes
    
    Args:
        json_str (str): Chaîne JSON brute avec des guillemets problématiques
    
    Returns:
        dict or None: Dictionnaire parsé ou None si erreur
    """
    if pd.isna(json_str) or json_str.strip() == '':
        return None
    
    try:
        cleaned = str(json_str).strip()
        
        # Debug: afficher le JSON brut pour la première erreur seulement
        if not hasattr(clean_and_parse_json, 'debug_shown'):
            print(f"🔍 Debug JSON brut (100 premiers chars): {cleaned[:100]}")
            clean_and_parse_json.debug_shown = True
        
        # Stratégie 1: Nettoyage standard
        def strategy_standard(json_text):
            if json_text.startswith('"') and json_text.endswith('"'):
                json_text = json_text[1:-1]
            return json_text.replace('""', '"')
        
        # Stratégie 2: Extraire seulement la partie entre { et }
        def strategy_extract_braces(json_text):
            first_brace = json_text.find('{')
            last_brace = json_text.rfind('}')
            if first_brace >= 0 and last_brace > first_brace:
                extracted = json_text[first_brace:last_brace + 1]
                return extracted.replace('""', '"')
            return json_text
        
        # Stratégie 3: Nettoyage agressif des guillemets multiples
        def strategy_aggressive(json_text):
            # Enlever tous les guillemets au début jusqu'à trouver {
            while json_text.startswith('"') and len(json_text) > 1:
                json_text = json_text[1:]
            
            # Enlever tous les guillemets à la fin jusqu'à trouver }
            while json_text.endswith('"') and len(json_text) > 1:
                json_text = json_text[:-1]
            
            # Remplacer les guillemets doubles partout
            json_text = json_text.replace('""', '"')
            
            # Si ça ne commence toujours pas par {, chercher la première accolade
            if not json_text.startswith('{'):
                brace_index = json_text.find('{')
                if brace_index > 0:
                    json_text = json_text[brace_index:]
            
            return json_text
        
        # Essayer chaque stratégie dans l'ordre
        strategies = [strategy_standard, strategy_extract_braces, strategy_aggressive]
        
        for i, strategy in enumerate(strategies):
            try:
                processed = strategy(cleaned)
                result = json.loads(processed)
                
                # Debug: afficher quelle stratégie a marché pour la première réussite
                if not hasattr(clean_and_parse_json, 'success_shown'):
                    print(f"✅ Stratégie {i+1} réussie! JSON nettoyé (50 premiers chars): {processed[:50]}")
                    clean_and_parse_json.success_shown = True
                
                return result
                
            except (json.JSONDecodeError, ValueError):
                continue  # Essayer la stratégie suivante
        
        # Si aucune stratégie n'a marché, essayer une approche de regex
        import re
        
        # Dernière tentative: utiliser regex pour extraire un JSON valide
        json_pattern = r'\{.*\}'
        match = re.search(json_pattern, cleaned, re.DOTALL)
        
        if match:
            potential_json = match.group(0).replace('""', '"')
            try:
                return json.loads(potential_json)
            except:
                pass
        
        return None
        
    except Exception as e:
        if not hasattr(clean_and_parse_json, 'error_shown'):
            print(f"⚠️  Erreur parsing JSON: {str(e)[:50]}...")
            clean_and_parse_json.error_shown = True
        return None

def extract_json_features(df):
    """
    Extraire les features analytiquement utiles des colonnes JSON
    Focus : comprendre l'opinion des testeurs sur le brainstorming avec IA
    
    Args:
        df (pd.DataFrame): DataFrame original
    
    Returns:
        pd.DataFrame: DataFrame avec les colonnes d'analyse extraites du JSON
    """
    print("🔄 Extraction des features d'analyse d'opinion...")
    
    # UNIQUEMENT les colonnes utiles pour analyser l'opinion sur le brainstorming IA
    json_features = {
        # Profil utilisateur (contexte)
        'user_language': 'language',
        'education_level': 'educationLevel',
        
        # Expérience et confort avec l'IA (cœur de l'analyse)
        'ai_usage_frequency': 'aiToolsUsage',        # Fréquence d'usage IA
        'ai_comfort_score': 'aiComfortLevel',        # Niveau de confort avec IA (1-5)
        'ai_creative_perception': 'aiCreativeProcess', # Perception IA dans créativité
        'ai_tools_used': 'aiToolSpecification',      # Quels outils IA utilisés
        
        # Expérience brainstorming (comportement/préférences)
        'brainstorming_experience': 'brainstormingExperience', # Expérience brainstorming
        'brainstorming_preferences': 'brainstormingModalities', # Individuel vs groupe
        'self_creativity_rating': 'creativityLevel',  # Auto-évaluation créativité
    }
    
    print(f"📊 Focus sur {len(json_features)} variables d'analyse d'opinion")
    
    # Initialiser les nouvelles colonnes
    for col_name in json_features.keys():
        df[col_name] = None
    
    # Compteurs pour le rapport
    json_parsed_count = 0
    json_error_count = 0
    
    # Parser chaque ligne avec du JSON
    for idx, row in df.iterrows():
        json_str = row['Responses (JSON)']
        
        if pd.notna(json_str) and str(json_str).strip() != '':
            json_data = clean_and_parse_json(json_str)
            
            if json_data:
                json_parsed_count += 1
                # Extraire chaque feature
                for new_col, json_key in json_features.items():
                    if json_key in json_data:
                        df.at[idx, new_col] = json_data[json_key]
            else:
                json_error_count += 1
    
    print(f"✅ JSON parsing: {json_parsed_count} réussies, {json_error_count} erreurs")
    return df

def create_derived_features(df):
    """
    Créer des variables dérivées centrées sur l'analyse d'opinion
    Focus : patterns d'usage, satisfaction, efficacité perçue du brainstorming IA
    
    Args:
        df (pd.DataFrame): DataFrame avec features extraites
    
    Returns:
        pd.DataFrame: DataFrame avec features d'analyse
    """
    print("🔄 Création des variables d'analyse d'opinion...")
    
    # 1. Indicateur de participant valide (a rempli les questions importantes)
    df['valid_participant'] = (
        df['ai_usage_frequency'].notna() & 
        df['ai_comfort_score'].notna() &
        df['brainstorming_experience'].notna()
    ).astype(int)
    
    # 2. Profil d'expertise IA (pour segmentation)
    def categorize_ai_expertise(usage, comfort):
        if pd.isna(usage) or pd.isna(comfort):
            return 'unknown'
        
        usage = str(usage).lower()
        comfort = float(comfort) if str(comfort).replace('.','').isdigit() else 0
        
        if usage in ['daily', 'weekly'] and comfort >= 4:
            return 'expert'        # Usage fréquent + très à l'aise
        elif usage in ['daily', 'weekly'] or comfort >= 4:
            return 'advanced'      # Soit usage fréquent, soit très à l'aise
        elif usage in ['monthly', 'occasionally'] or comfort >= 2:
            return 'intermediate'  # Usage modéré ou moyennement à l'aise
        elif usage == 'never' or comfort < 2:
            return 'novice'        # Peu/pas d'usage + peu à l'aise
        else:
            return 'unknown'
    
    df['ai_expertise_profile'] = df.apply(
        lambda x: categorize_ai_expertise(x['ai_usage_frequency'], x['ai_comfort_score']), 
        axis=1
    )
    
    # 3. Attitude envers l'IA créative (variable clé d'opinion)
    def analyze_ai_creative_attitude(ai_perception):
        if pd.isna(ai_perception):
            return 'unknown'
        
        perception = str(ai_perception).lower()
        if 'always' in perception or 'daily' in perception:
            return 'enthusiast'     # Très positif envers IA créative
        elif 'sometimes' in perception or 'occasionally' in perception:
            return 'moderate'       # Ouvert mais modéré
        elif 'never' in perception or 'rarely' in perception:
            return 'skeptical'      # Réticent à l'IA créative
        else:
            return 'neutral'
    
    df['ai_creative_attitude'] = df['ai_creative_perception'].apply(analyze_ai_creative_attitude)
    
    # 4. Niveau d'éducation standardisé (contexte démographique)
    def standardize_education(level):
        if pd.isna(level):
            return 'unknown'
        
        level = str(level).lower()
        if any(x in level for x in ['phd', 'doctorate', 'doctorat', 'bac8']):
            return 'graduate+'      # PhD/Doctorat
        elif any(x in level for x in ['master', 'bac5', 'bac+5']):
            return 'graduate'       # Master
        elif any(x in level for x in ['bachelor', 'licence', 'bac3', 'bac+3']):
            return 'undergraduate'  # Licence/Bachelor
        elif any(x in level for x in ['bac', 'high_school']):
            return 'secondary'      # Baccalauréat
        else:
            return 'other'
    
    df['education_category'] = df['education_level'].apply(standardize_education)
    
    # 5. Score de créativité personnel (auto-perception)
    def normalize_creativity_score(score):
        if pd.isna(score):
            return None
        
        try:
            numeric_score = float(score)
            if numeric_score <= 2:
                return 'low'
            elif numeric_score <= 3:
                return 'moderate'
            elif numeric_score <= 4:
                return 'high'
            else:
                return 'very_high'
        except:
            return None
    
    df['creativity_self_assessment'] = df['self_creativity_rating'].apply(normalize_creativity_score)
    
    # 6. Préférence de modalité brainstorming (comportement collaboratif)
    def analyze_brainstorming_preference(modalities):
        if pd.isna(modalities):
            return 'unknown'
        
        modality = str(modalities).lower()
        if 'individual' in modality and 'group' not in modality:
            return 'individual_only'
        elif 'group' in modality and 'individual' not in modality:
            return 'group_only'
        elif 'individual' in modality and 'group' in modality:
            return 'flexible'
        else:
            return 'other'
    
    df['brainstorming_preference'] = df['brainstorming_preferences'].apply(analyze_brainstorming_preference)
    
    # 7. Expérience brainstorming (niveau de pratique)
    def categorize_brainstorming_experience(experience):
        if pd.isna(experience):
            return 'unknown'
        
        exp = str(experience).lower()
        if exp in ['daily', 'weekly']:
            return 'frequent'
        elif exp in ['monthly', 'occasionally']:
            return 'occasional'
        elif exp in ['rarely', 'never']:
            return 'rare'
        else:
            return 'unknown'
    
    df['brainstorming_experience_level'] = df['brainstorming_experience'].apply(categorize_brainstorming_experience)
    
    # 8. Diversité des outils IA utilisés (indicateur de sophistication)
    def count_ai_tools(tools_spec):
        if pd.isna(tools_spec):
            return 0
        
        tools = str(tools_spec).lower()
        tool_indicators = ['chatgpt', 'claude', 'copilot', 'gemini', 'midjourney', 'dall-e']
        return sum(1 for tool in tool_indicators if tool in tools)
    
    df['ai_tools_diversity'] = df['ai_tools_used'].apply(count_ai_tools)
    
    print("✅ Variables d'analyse d'opinion créées")
    return df

def generate_data_quality_report(df_original, df_transformed):
    """
    Générer un rapport centré sur la qualité des variables d'analyse d'opinion
    
    Args:
        df_original (pd.DataFrame): DataFrame original
        df_transformed (pd.DataFrame): DataFrame transformé
    
    Returns:
        dict: Rapport de transformation centré sur l'analyse d'opinion
    """
    report = {
        'original_shape': df_original.shape,
        'transformed_shape': df_transformed.shape,
        'analysis_focus': 'Opinion des testeurs sur brainstorming IA',
        'new_columns_added': [],
        'completion_stats': {},
        'data_quality': {}
    }
    
    # Identifier les nouvelles colonnes d'analyse
    original_cols = set(df_original.columns)
    transformed_cols = set(df_transformed.columns)
    report['new_columns_added'] = list(transformed_cols - original_cols)
    
    # Statistiques de participants
    report['completion_stats'] = {
        'total_responses': len(df_transformed),
        'valid_participants': df_transformed.get('valid_participant', pd.Series([0])).sum(),
        'participants_with_ai_data': df_transformed['ai_usage_frequency'].notna().sum(),
        'participants_with_brainstorming_data': df_transformed['brainstorming_experience'].notna().sum(),
    }
    
    # Variables clés d'analyse d'opinion à évaluer
    opinion_analysis_vars = [
        'ai_expertise_profile',      # Profil expertise IA
        'ai_creative_attitude',      # Attitude IA créative  
        'creativity_self_assessment', # Auto-évaluation créativité
        'brainstorming_preference',  # Préférence modalité
        'brainstorming_experience_level', # Expérience brainstorming
        'education_category',        # Contexte éducatif
        'ai_tools_diversity'         # Diversité outils utilisés
    ]
    
    for var in opinion_analysis_vars:
        if var in df_transformed.columns:
            report['data_quality'][var] = {
                'non_null': df_transformed[var].notna().sum(),
                'unique_values': df_transformed[var].nunique(),
                'top_values': df_transformed[var].value_counts().head(3).to_dict()
            }
    
    return report

def save_transformation_report(report, output_dir):
    """Sauvegarder le rapport de transformation centré sur l'analyse d'opinion"""
    report_path = output_dir / 'transformation_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RAPPORT DE TRANSFORMATION - ANALYSE D'OPINION BRAINSTORMING IA\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"🎯 OBJECTIF: Comprendre l'opinion des testeurs sur le brainstorming avec IA\n\n")
        
        f.write(f"📊 DIMENSIONS\n")
        f.write(f"Original: {report['original_shape'][0]} lignes × {report['original_shape'][1]} colonnes\n")
        f.write(f"Transformé: {report['transformed_shape'][0]} lignes × {report['transformed_shape'][1]} colonnes\n")
        f.write(f"Variables d'analyse créées: {len(report['new_columns_added'])}\n\n")
        
        f.write(f"👥 PARTICIPANTS\n")
        for key, value in report['completion_stats'].items():
            f.write(f"- {key}: {value}\n")
        f.write("\n")
        
        f.write(f"🧠 VARIABLES D'OPINION ANALYSÉES\n")
        opinion_vars = {
            'ai_expertise_profile': 'Profil d\'expertise IA',
            'ai_creative_attitude': 'Attitude envers l\'IA créative', 
            'creativity_self_assessment': 'Auto-évaluation créativité',
            'brainstorming_preference': 'Préférence modalité brainstorming',
            'brainstorming_experience_level': 'Niveau d\'expérience brainstorming'
        }
        
        for var, description in opinion_vars.items():
            if var in report['data_quality']:
                stats = report['data_quality'][var]
                f.write(f"\n{description} ({var}):\n")
                f.write(f"  - Réponses valides: {stats['non_null']}\n")
                f.write(f"  - Catégories: {stats['unique_values']}\n")
                f.write(f"  - Distribution: {stats['top_values']}\n")
        
        f.write(f"\n📈 RECOMMANDATIONS D'ANALYSE\n")
        f.write("1. SEGMENTATION: Analyser par 'ai_expertise_profile' (novice/intermediate/advanced/expert)\n")
        f.write("2. OPINION CORE: Focus sur 'ai_creative_attitude' (enthusiast/moderate/skeptical)\n") 
        f.write("3. COMPORTEMENT: Croiser 'brainstorming_preference' × 'ai_expertise_profile'\n")
        f.write("4. PERCEPTION: Analyser 'creativity_self_assessment' en fonction de l'expertise IA\n")
        f.write("5. PATTERNS: Identifier les profils d'adoption de l'IA créative\n\n")
        
        f.write(f"🔍 INSIGHTS ANALYTIQUES POSSIBLES\n")
        f.write("- Les experts IA sont-ils plus enthousiastes pour le brainstorming IA?\n")
        f.write("- Y a-t-il une corrélation créativité perçue × attitude IA?\n")
        f.write("- Les préférences individuelles vs groupe varient-elles par expertise IA?\n")
        f.write("- Quels outils IA sont préférés par segment d'utilisateur?\n")
        f.write("- Comment l'expérience brainstorming influence l'adoption IA?\n\n")
        
        f.write(f"⚠️  EXCLUSIONS VOLONTAIRES\n")
        f.write("- Métadonnées techniques (Session ID, Response ID, etc.)\n")
        f.write("- Variables de consentement et validation technique\n")
        f.write("- Identifiants Prolific et paramètres techniques\n")
        f.write("- Focus uniquement sur les variables d'opinion et comportement\n")

def main():
    """Fonction principale de transformation"""
    print("🚀 DÉMARRAGE DE LA TRANSFORMATION DES DONNÉES")
    print("=" * 50)
    
    # 1. Créer répertoire de sortie
    output_dir = create_output_directory()
    print(f"📁 Répertoire de sortie: {output_dir}")
    
    # 2. Charger les données
    print("\n🔄 Chargement des données...")
    input_file = 'data/donnee_test.csv'
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Fichier non trouvé: {input_file}")
    
    # Charger avec les bons paramètres (skiprows=1 car première ligne = titre)
    df_original = pd.read_csv(input_file, sep=';', skiprows=1)
    print(f"✅ Dataset chargé: {df_original.shape[0]} lignes × {df_original.shape[1]} colonnes")
    
    # Debug: vérifier la structure des données chargées
    print("\n🔍 DIAGNOSTIC DES DONNÉES CHARGÉES:")
    print("Colonnes du dataset:", list(df_original.columns))
    
    # Vérifier les colonnes JSON
    json_cols = ['Responses (JSON)', 'Metadata (JSON)']
    for col in json_cols:
        if col in df_original.columns:
            non_null = df_original[col].notna().sum()
            print(f"{col}: {non_null} valeurs non-nulles")
            
            if non_null > 0:
                # Afficher un exemple de contenu
                sample = df_original[df_original[col].notna()][col].iloc[0]
                print(f"  Exemple (100 premiers chars): {str(sample)[:100]}...")
        else:
            print(f"⚠️  Colonne manquante: {col}")
    
    # Vérifier si les données semblent cohérentes
    print("\nÉchantillon des premières lignes:")
    for i in range(min(3, len(df_original))):
        row = df_original.iloc[i]
        print(f"Ligne {i}: Session ID = {row.get('Session ID', 'N/A')}")
        print(f"         Response ID = {row.get('Response ID', 'N/A')}")
        print(f"         Response Type = {row.get('Response Type', 'N/A')}")
    
    # 3. Extraire les features JSON
    df_with_json = extract_json_features(df_original.copy())
    
    # 4. Créer les features dérivées
    df_transformed = create_derived_features(df_with_json)
    
    # 5. Diagnostic post-transformation
    print("\n🔍 DIAGNOSTIC POST-TRANSFORMATION:")
    
    # Vérifier si les nouvelles colonnes d'analyse ont des données
    analysis_cols = [
        'ai_expertise_profile', 'ai_creative_attitude', 
        'creativity_self_assessment', 'brainstorming_preference',
        'brainstorming_experience_level'
    ]
    
    for col in analysis_cols:
        if col in df_transformed.columns:
            non_null = df_transformed[col].notna().sum()
            if non_null > 0:
                unique_vals = df_transformed[col].dropna().unique()[:3]
                print(f"✅ {col}: {non_null} valeurs, exemples: {list(unique_vals)}")
            else:
                print(f"⚠️  {col}: 0 valeurs extraites")
        else:
            print(f"❌ {col}: colonne manquante")
    
    # 6. Générer le rapport
    print("\n📊 Génération du rapport de transformation...")
    report = generate_data_quality_report(df_original, df_transformed)
    
    # 7. Sauvegarder les résultats
    output_file = output_dir / 'donnee_test_processed.csv'
    df_transformed.to_csv(output_file, index=False, encoding='utf-8')
    print(f"💾 Dataset transformé sauvegardé: {output_file}")
    
    # 8. Sauvegarder le rapport
    save_transformation_report(report, output_dir)
    print(f"📝 Rapport sauvegardé: {output_dir / 'transformation_report.txt'}")
    
    # 9. Vérification finale du fichier de sortie
    print("\n🔍 VÉRIFICATION FICHIER DE SORTIE:")
    try:
        df_check = pd.read_csv(output_file, nrows=3)
        print(f"Fichier vérifié: {df_check.shape[0]} lignes × {df_check.shape[1]} colonnes")
        print("Colonnes de sortie:", list(df_check.columns)[:10], "..." if len(df_check.columns) > 10 else "")
    except Exception as e:
        print(f"⚠️  Erreur lors de la vérification: {e}")
    
    # 10. Afficher un résumé centré sur l'analyse d'opinion
    print(f"\n🎉 TRANSFORMATION TERMINÉE!")
    print("=" * 50)
    print(f"👥 Participants valides: {report['completion_stats']['valid_participants']}")
    print(f"🧠 Variables d'analyse créées: {len(report['new_columns_added'])}")
    print(f"🤖 Participants avec données IA: {report['completion_stats']['participants_with_ai_data']}")
    print(f"💡 Participants avec données brainstorming: {report['completion_stats']['participants_with_brainstorming_data']}")
    
    if report['completion_stats']['valid_participants'] == 0:
        print("\n⚠️  ATTENTION: Aucun participant valide détecté!")
        print("Vérifiez le fichier de sortie et le rapport pour plus de détails.")
    else:
        print(f"\n✅ Prêt pour l'analyse d'opinion sur le brainstorming IA!")
    
    print(f"\n🔄 Prochaine étape: Analyser le fichier data/etude_prolific.csv")

if __name__ == "__main__":
    main()