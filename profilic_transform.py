#!/usr/bin/env python3
"""
Script de transformation pour les données Prolific de l'étude brainstorming IA
Auteur: Étude de recherche brainstorming
Date: Septembre 2025

Ce script transforme le dataset etude_prolific.csv en se concentrant sur :
- Les données démographiques des participants
- Les groupes expérimentaux (A vs B)  
- Les réponses créatives de brainstorming
- Le lien avec donnee_test.csv via Session ID
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_output_directory():
    """Créer le répertoire de sortie si nécessaire"""
    output_dir = Path('data/transform')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def analyze_brainstorming_content(text):
    """
    Analyser le contenu des réponses de brainstorming
    Extraire métriques de créativité et thèmes
    
    Args:
        text (str): Texte de la réponse brainstorming
    
    Returns:
        dict: Métriques d'analyse de contenu
    """
    if pd.isna(text) or str(text).strip() == '':
        return {
            'response_length': 0,
            'idea_count': 0,
            'technical_terms': 0,
            'security_focus': 0,
            'innovation_words': 0,
            'practical_solutions': 0
        }
    
    text = str(text).lower()
    
    # Compter les idées (séparées par points, virgules, "and", etc.)
    idea_separators = ['.', ',', ';', ' and ', ' or ', '\n', ' - ', '•']
    idea_count = 1
    for sep in idea_separators:
        idea_count += text.count(sep)
    
    # Mots techniques/technologiques
    tech_terms = ['digital', 'encryption', 'authentication', 'biometric', 'algorithm', 
                  'password', 'security', 'token', 'blockchain', 'ai', 'smart']
    technical_terms = sum(1 for term in tech_terms if term in text)
    
    # Focus sécurité
    security_words = ['secure', 'safe', 'protect', 'lock', 'privacy', 'hack', 'threat']
    security_focus = sum(1 for word in security_words if word in text)
    
    # Mots d'innovation
    innovation_words = ['creative', 'innovative', 'new', 'novel', 'unique', 'original', 'different']
    innovation_count = sum(1 for word in innovation_words if word in text)
    
    # Solutions pratiques
    practical_words = ['use', 'implement', 'apply', 'install', 'setup', 'configure']
    practical_solutions = sum(1 for word in practical_words if word in text)
    
    return {
        'response_length': len(text),
        'idea_count': min(idea_count, 20),  # Cap à 20 pour éviter les outliers
        'technical_terms': technical_terms,
        'security_focus': security_focus,  
        'innovation_words': innovation_count,
        'practical_solutions': practical_solutions
    }

def extract_prolific_features(df):
    """
    Extraire et standardiser les features démographiques Prolific
    
    Args:
        df (pd.DataFrame): DataFrame original
    
    Returns:
        pd.DataFrame: DataFrame avec features démographiques standardisées
    """
    print("🔄 Extraction des features démographiques Prolific...")
    
    # 1. Standardiser le sexe
    df['gender'] = df['Sex'].map({
        'Male': 'male',
        'Female': 'female'
    }).fillna('other')
    
    # 2. Standardiser le statut étudiant (ignorer DATA_EXPIRED)
    def clean_student_status(status):
        if pd.isna(status) or status == 'DATA_EXPIRED':
            return 'unknown'
        return 'student' if status == 'Yes' else 'non_student'
    
    df['student_category'] = df['Student status'].apply(clean_student_status)
    
    # 3. Standardiser le statut d'emploi
    def categorize_employment(status):
        if pd.isna(status) or status == 'DATA_EXPIRED':
            return 'unknown'
        
        status = str(status).lower()
        if 'full-time' in status:
            return 'full_time'
        elif 'part-time' in status:
            return 'part_time'  
        elif 'unemployed' in status:
            return 'unemployed'
        elif 'not in paid work' in status or 'retired' in status:
            return 'not_working'
        else:
            return 'other'
    
    df['employment_category'] = df['Employment status'].apply(categorize_employment)
    
    # 4. Normaliser la langue (déjà homogène = français)
    df['primary_language'] = df['Language'].str.lower().fillna('unknown')
    
    # 5. Extraire le pays de naissance (standardiser les plus fréquents)
    def standardize_country(country):
        if pd.isna(country):
            return 'unknown'
        
        country = str(country).lower()
        if 'france' in country or 'french' in country:
            return 'france'
        elif 'united kingdom' in country or 'uk' in country:
            return 'uk'
        elif 'united states' in country or 'usa' in country:
            return 'usa'
        else:
            return 'other'
    
    df['birth_country_category'] = df['Country of birth'].apply(standardize_country)
    
    print("✅ Features démographiques extraites")
    return df

def analyze_experimental_groups(df):
    """
    Analyser les groupes expérimentaux et leurs caractéristiques
    
    Args:
        df (pd.DataFrame): DataFrame avec groupes
    
    Returns:
        pd.DataFrame: DataFrame avec analyse des groupes
    """
    print("🔄 Analyse des groupes expérimentaux...")
    
    # Standardiser les groupes
    df['experimental_group'] = df['Groupe'].fillna('unknown')
    
    # Analyser les questions posées par groupe
    def categorize_question_type(question):
        if pd.isna(question):
            return 'unknown'
        
        question = str(question).lower()
        if 'security' in question or 'secure' in question:
            return 'security_focused'
        elif 'creative' in question or 'creative' in question:
            return 'creativity_focused'  
        elif 'solution' in question:
            return 'solution_focused'
        else:
            return 'general'
    
    df['question_type'] = df['Question de brainstorming (onboarding)'].apply(categorize_question_type)
    
    # Marquer les participants valides (avec données complètes)
    df['valid_prolific_participant'] = (
        df['experimental_group'].isin(['A', 'B']) &
        df['Question de brainstorming (onboarding)'].notna() &
        df['Session de brainstorming (initial_survey)'].notna() &
        df['Session ID'].notna()
    ).astype(int)
    
    print("✅ Analyse des groupes expérimentaux terminée")
    return df

def analyze_creative_responses(df):
    """
    Analyser les réponses créatives de brainstorming
    
    Args:
        df (pd.DataFrame): DataFrame avec réponses
    
    Returns:
        pd.DataFrame: DataFrame avec métriques de créativité
    """
    print("🔄 Analyse des réponses créatives...")
    
    # Analyser chaque réponse de brainstorming
    response_metrics = df['Session de brainstorming (initial_survey)'].apply(analyze_brainstorming_content)
    
    # Convertir en colonnes séparées
    for metric in ['response_length', 'idea_count', 'technical_terms', 'security_focus', 'innovation_words', 'practical_solutions']:
        df[f'creativity_{metric}'] = response_metrics.apply(lambda x: x[metric])
    
    # Créer un score de créativité composite
    def calculate_creativity_score(row):
        # Normaliser chaque composant (0-1)
        length_score = min(row['creativity_response_length'] / 500, 1.0)  # Max 500 chars = 1.0
        idea_score = min(row['creativity_idea_count'] / 10, 1.0)  # Max 10 idées = 1.0  
        tech_score = min(row['creativity_technical_terms'] / 5, 1.0)  # Max 5 termes = 1.0
        innovation_score = min(row['creativity_innovation_words'] / 3, 1.0)  # Max 3 mots = 1.0
        
        # Score pondéré (idées comptent plus que longueur)
        composite_score = (
            length_score * 0.2 +
            idea_score * 0.4 +
            tech_score * 0.2 + 
            innovation_score * 0.2
        )
        
        return round(composite_score, 3)
    
    df['creativity_composite_score'] = df.apply(calculate_creativity_score, axis=1)
    
    # Catégoriser les niveaux de créativité
    def categorize_creativity_level(score):
        if score >= 0.7:
            return 'high'
        elif score >= 0.4:
            return 'moderate'
        elif score > 0:
            return 'low'
        else:
            return 'none'
    
    df['creativity_level_category'] = df['creativity_composite_score'].apply(categorize_creativity_level)
    
    print("✅ Analyse des réponses créatives terminée")
    return df

def create_demographic_profiles(df):
    """
    Créer des profils démographiques combinés pour l'analyse
    
    Args:
        df (pd.DataFrame): DataFrame avec features démographiques
    
    Returns:
        pd.DataFrame: DataFrame avec profils combinés
    """
    print("🔄 Création des profils démographiques...")
    
    # Profil socio-économique
    def create_socioeconomic_profile(row):
        if row['student_category'] == 'student':
            return 'student'
        elif row['employment_category'] == 'full_time':
            return 'working_professional'
        elif row['employment_category'] == 'part_time':
            return 'part_time_worker'
        elif row['employment_category'] == 'unemployed':
            return 'job_seeker'
        else:
            return 'other'
    
    df['socioeconomic_profile'] = df.apply(create_socioeconomic_profile, axis=1)
    
    # Profil démographique simple
    df['demographic_profile'] = df['gender'] + '_' + df['student_category']
    
    # Age approximatif basé sur le statut (estimation grossière)
    def estimate_age_group(row):
        if row['student_category'] == 'student':
            return 'young_adult'  # 18-25 typiquement
        elif row['employment_category'] == 'full_time':
            return 'adult'  # 25-55 typiquement  
        elif 'retired' in str(row['Employment status']).lower():
            return 'senior'  # 55+
        else:
            return 'adult'  # Par défaut
    
    df['estimated_age_group'] = df.apply(estimate_age_group, axis=1)
    
    print("✅ Profils démographiques créés")
    return df

def generate_prolific_report(df_original, df_transformed):
    """
    Générer un rapport spécifique aux données Prolific
    
    Args:
        df_original (pd.DataFrame): DataFrame original  
        df_transformed (pd.DataFrame): DataFrame transformé
    
    Returns:
        dict: Rapport d'analyse Prolific
    """
    report = {
        'original_shape': df_original.shape,
        'transformed_shape': df_transformed.shape,
        'analysis_focus': 'Données démographiques et créatives Prolific',
        'participant_stats': {},
        'experimental_groups': {},
        'demographic_analysis': {},
        'creativity_analysis': {}
    }
    
    # Statistiques participants
    report['participant_stats'] = {
        'total_submissions': len(df_transformed),
        'valid_participants': df_transformed['valid_prolific_participant'].sum(),
        'approved_participants': len(df_transformed[df_transformed['Status'] == 'APPROVED']),
        'session_ids_available': df_transformed['Session ID'].notna().sum()
    }
    
    # Analyse groupes expérimentaux
    if 'experimental_group' in df_transformed.columns:
        group_counts = df_transformed['experimental_group'].value_counts().to_dict()
        report['experimental_groups'] = {
            'group_distribution': group_counts,
            'valid_by_group': df_transformed.groupby('experimental_group')['valid_prolific_participant'].sum().to_dict()
        }
    
    # Analyse démographique
    demo_vars = ['gender', 'student_category', 'employment_category', 'socioeconomic_profile']
    for var in demo_vars:
        if var in df_transformed.columns:
            report['demographic_analysis'][var] = df_transformed[var].value_counts().head(5).to_dict()
    
    # Analyse créativité
    if 'creativity_level_category' in df_transformed.columns:
        report['creativity_analysis'] = {
            'creativity_distribution': df_transformed['creativity_level_category'].value_counts().to_dict(),
            'avg_composite_score': df_transformed['creativity_composite_score'].mean(),
            'creativity_by_group': df_transformed.groupby('experimental_group')['creativity_composite_score'].mean().to_dict()
        }
    
    return report

def save_prolific_report(report, output_dir):
    """Sauvegarder le rapport d'analyse Prolific"""
    report_path = output_dir / 'prolific_transformation_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RAPPORT DE TRANSFORMATION - DONNÉES PROLIFIC BRAINSTORMING IA\n")
        f.write("=" * 65 + "\n\n")
        
        f.write("🎯 OBJECTIF: Analyser les profils démographiques et créatifs des participants\n\n")
        
        f.write("📊 DIMENSIONS\n")
        f.write(f"Original: {report['original_shape'][0]} lignes × {report['original_shape'][1]} colonnes\n")
        f.write(f"Transformé: {report['transformed_shape'][0]} lignes × {report['transformed_shape'][1]} colonnes\n\n")
        
        f.write("👥 PARTICIPANTS\n")
        for key, value in report['participant_stats'].items():
            f.write(f"- {key}: {value}\n")
        f.write("\n")
        
        f.write("🧪 GROUPES EXPÉRIMENTAUX\n")
        if 'group_distribution' in report['experimental_groups']:
            for group, count in report['experimental_groups']['group_distribution'].items():
                f.write(f"- Groupe {group}: {count} participants\n")
        f.write("\n")
        
        f.write("👤 PROFILS DÉMOGRAPHIQUES\n")
        for var, distribution in report['demographic_analysis'].items():
            f.write(f"\n{var}:\n")
            for category, count in distribution.items():
                f.write(f"  - {category}: {count}\n")
        
        f.write("\n🎨 ANALYSE DE CRÉATIVITÉ\n")
        if report['creativity_analysis']:
            creativity = report['creativity_analysis']
            f.write(f"Score moyen de créativité: {creativity.get('avg_composite_score', 0):.3f}\n")
            
            if 'creativity_distribution' in creativity:
                f.write("Distribution des niveaux:\n")
                for level, count in creativity['creativity_distribution'].items():
                    f.write(f"  - {level}: {count}\n")
            
            if 'creativity_by_group' in creativity:
                f.write("Créativité par groupe expérimental:\n")
                for group, score in creativity['creativity_by_group'].items():
                    f.write(f"  - Groupe {group}: {score:.3f}\n")
        
        f.write(f"\n🔗 INTÉGRATION AVEC DONNEE_TEST.CSV\n")
        f.write("- Utiliser 'Session ID' comme clé de jointure\n")  
        f.write("- Combiner profils démographiques + préférences IA\n")
        f.write("- Analyser créativité × expertise IA × groupe expérimental\n\n")
        
        f.write(f"📈 ANALYSES RECOMMANDÉES\n")
        f.write("1. COMPARAISON GROUPES: Groupe A vs B sur métriques créativité\n")
        f.write("2. PROFILS DÉMOGRAPHIQUES: Impact genre/âge/emploi sur créativité\n") 
        f.write("3. CORRÉLATIONS: Créativité Prolific × confort IA (après merge)\n")
        f.write("4. SEGMENTATION: Profils socio-économiques × attitudes IA\n")
        f.write("5. EFFICACITÉ: Groupe expérimental × qualité solutions\n")

def main():
    """Fonction principale de transformation Prolific"""
    print("🚀 DÉMARRAGE TRANSFORMATION DONNÉES PROLIFIC")
    print("=" * 50)
    
    # 1. Créer répertoire de sortie
    output_dir = create_output_directory()
    print(f"📁 Répertoire de sortie: {output_dir}")
    
    # 2. Charger les données
    print("\n🔄 Chargement des données Prolific...")
    input_file = 'data/etude_prolific.csv'
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Fichier non trouvé: {input_file}")
    
    # Charger avec skiprows=1 (même structure que donnee_test.csv)
    df_original = pd.read_csv(input_file, sep=';', skiprows=1)
    print(f"✅ Dataset chargé: {df_original.shape[0]} lignes × {df_original.shape[1]} colonnes")
    
    # Diagnostic initial
    print(f"\n🔍 DIAGNOSTIC INITIAL:")
    approved = (df_original['Status'] == 'APPROVED').sum()
    with_session_id = df_original['Session ID'].notna().sum() 
    with_brainstorming = df_original['Question de brainstorming (onboarding)'].notna().sum()
    print(f"Participants approuvés: {approved}")
    print(f"Avec Session ID: {with_session_id}")  
    print(f"Avec données brainstorming: {with_brainstorming}")
    
    # 3. Extraire features démographiques
    df_with_demo = extract_prolific_features(df_original.copy())
    
    # 4. Analyser groupes expérimentaux  
    df_with_groups = analyze_experimental_groups(df_with_demo)
    
    # 5. Analyser réponses créatives
    df_with_creativity = analyze_creative_responses(df_with_groups)
    
    # 6. Créer profils démographiques
    df_transformed = create_demographic_profiles(df_with_creativity)
    
    # 7. Diagnostic post-transformation
    print("\n🔍 DIAGNOSTIC POST-TRANSFORMATION:")
    
    key_vars = ['experimental_group', 'creativity_level_category', 'socioeconomic_profile', 'gender']
    for var in key_vars:
        if var in df_transformed.columns:
            non_null = df_transformed[var].notna().sum()
            unique_vals = df_transformed[var].dropna().unique()
            print(f"✅ {var}: {non_null} valeurs, catégories: {list(unique_vals)}")
    
    # 8. Générer le rapport
    print("\n📊 Génération du rapport Prolific...")
    report = generate_prolific_report(df_original, df_transformed)
    
    # 9. Sauvegarder les résultats
    output_file = output_dir / 'etude_prolific_processed.csv'
    df_transformed.to_csv(output_file, index=False, encoding='utf-8')
    print(f"💾 Dataset Prolific transformé: {output_file}")
    
    # 10. Sauvegarder le rapport
    save_prolific_report(report, output_dir)
    print(f"📝 Rapport Prolific: {output_dir / 'prolific_transformation_report.txt'}")
    
    # 11. Résumé final
    print(f"\n🎉 TRANSFORMATION PROLIFIC TERMINÉE!")
    print("=" * 50)
    print(f"👥 Participants valides: {report['participant_stats']['valid_participants']}")
    print(f"🧪 Groupes expérimentaux: {len(report['experimental_groups'].get('group_distribution', {}))}")
    print(f"🎨 Score créativité moyen: {report['creativity_analysis'].get('avg_composite_score', 0):.3f}")
    print(f"🔗 Session IDs pour merge: {report['participant_stats']['session_ids_available']}")
    
    print(f"\n🔄 Prochaine étape: Merger avec donnee_test_processed.csv via Session ID")

if __name__ == "__main__":
    main()