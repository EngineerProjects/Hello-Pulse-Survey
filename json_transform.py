import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

def parse_array_field(value, separator='|'):
    """Convert array to pipe-separated string or return empty string"""
    if value is None:
        return ''
    if isinstance(value, list):
        return separator.join(str(v) for v in value)
    return str(value)

def safe_get(d, *keys, default=None):
    """Safely get nested dictionary values"""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d

def parse_research_study(input_file, output_dir='data/transformed'):
    """
    Parse research study JSON and create 4 CSV files
    
    Parameters:
    -----------
    input_file : str
        Path to input JSON file
    output_dir : str
        Directory for output CSV files
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Reading JSON file: {input_file}")
    
    # Read JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sessions = data.get('sessions', [])
    print(f"Found {len(sessions)} sessions to process")
    
    # Initialize lists to store data
    participants_data = []
    sessions_data = []
    brainstorming_data = []
    questionnaire_data = []
    
    # Process each session
    for session_idx, session in enumerate(sessions, 1):
        session_id = session.get('sessionId')
        
        # Initialize variables for this session
        participant_info = {}
        group_assigned = None
        
        # Process each response in the session
        responses = session.get('responses', [])
        
        for response in responses:
            response_type = response.get('responseType')
            questionnaire_type = response.get('questionnaire_type')
            response_data = response.get('responses', {})
            
            # 1. PARTICIPANT ONBOARDING DATA
            if questionnaire_type == 'research_study_onboarding':
                prolific_id = response_data.get('prolificId', '')
                
                participant_info = {
                    'participant_id': prolific_id,
                    'session_id': session_id,
                    'language': response_data.get('language', ''),
                    'education_level': response_data.get('educationLevel', ''),
                    'education_other': response_data.get('educationOther', ''),
                    'ai_tools_usage': response_data.get('aiToolsUsage', ''),
                    'ai_comfort_level': response_data.get('aiComfortLevel', ''),
                    'creativity_level': response_data.get('creativityLevel', ''),
                    'ai_creative_process': response_data.get('aiCreativeProcess', ''),
                    'ai_tool_specification': response_data.get('aiToolSpecification', ''),
                    'brainstorming_experience': response_data.get('brainstormingExperience', ''),
                    'brainstorming_modalities': parse_array_field(response_data.get('brainstormingModalities', [])),
                    'consent_confirmation': response_data.get('consentConfirmation', ''),
                    'instructions_understood': response_data.get('instructionsUnderstood', '')
                }
            
            # 2. GROUP SELECTION DATA
            elif questionnaire_type == 'survey' and 'selectedGroup' in response_data:
                group_assigned = response_data.get('selectedGroup', '')
            
            # 3. BRAINSTORMING SESSION DATA
            elif questionnaire_type == 'brainstorming_session':
                # Count ideas (split by double newline, filter empty)
                ideas_text = response_data.get('ideas', '')
                ideas_list = [idea.strip() for idea in ideas_text.split('\n\n') if idea.strip()]
                ideas_count = len(ideas_list)
                
                brainstorming_entry = {
                    'session_id': session_id,
                    'mode': response_data.get('mode', ''),
                    'group': response_data.get('group', ''),
                    'ideas_text': ideas_text,
                    'ideas_count': ideas_count,
                    'used_ideas': parse_array_field(response_data.get('usedIdeas', [])),
                    'predefined_ideas_used': response_data.get('predefinedIdeasUsed', ''),
                    'total_predefined_ideas': safe_get(response.get('metadata', {}), 'total_predefined_ideas', default=''),
                    'ai_usage_count': response_data.get('aiUsageCount', ''),
                    'session_duration_seconds': response_data.get('sessionDuration', ''),
                    'current_idea_index': response_data.get('currentIdeaIndex', ''),
                    'question_prompt': safe_get(response.get('metadata', {}), 'question', default=''),
                    'completed_at': response.get('completedAt', '')
                }
                brainstorming_data.append(brainstorming_entry)
            
            # 4. POST-QUESTIONNAIRE DATA
            elif questionnaire_type == 'brainstorming_final_survey':
                questionnaire_entry = {
                    'session_id': session_id,
                    'group': response_data.get('group', ''),
                    'ai_benefits': parse_array_field(response_data.get('aiBenefits', [])),
                    'ai_benefits_other': response_data.get('otherBenefitDetails', ''),
                    'difficulties': parse_array_field(response_data.get('difficulties', [])),
                    'difficulties_other': response_data.get('otherDifficultyDetails', ''),
                    'external_help': response_data.get('externalHelp', ''),
                    'external_help_details': response_data.get('otherHelpDetails', ''),
                    'ai_helpfulness': response_data.get('aiHelpfulness', ''),
                    'ai_limitations': response_data.get('aiLimitations', ''),
                    'general_impression': response_data.get('generalImpression', ''),
                    'idea_quality_utility': response_data.get('ideaQualityUtility', ''),
                    'idea_quality_originality': response_data.get('ideaQualityOriginality', ''),
                    'personal_implication': response_data.get('personalImplication', ''),
                    'additional_comments': response_data.get('additionalComments', ''),
                    'technical_difficulties': response_data.get('technicalDifficultiesDetails', ''),
                    'completed_at': response.get('completedAt', '')
                }
                questionnaire_data.append(questionnaire_entry)
        
        # Add participant data if we have it
        if participant_info:
            participants_data.append(participant_info)
        
        # Add session data
        session_metadata = session.get('metadata', {})
        session_entry = {
            'session_id': session_id,
            'participant_id': participant_info.get('participant_id', ''),
            'group_assigned': group_assigned or '',
            'session_created_at': session_metadata.get('createdAt', ''),
            'total_responses': session_metadata.get('totalResponses', ''),
            'completed_responses': session_metadata.get('completedResponses', '')
        }
        sessions_data.append(session_entry)
        
        # Progress indicator
        if session_idx % 100 == 0 or session_idx == len(sessions):
            print(f"Processed {session_idx}/{len(sessions)} sessions...")
    
    # Create DataFrames
    print("\nCreating DataFrames...")
    df_participants = pd.DataFrame(participants_data)
    df_sessions = pd.DataFrame(sessions_data)
    df_brainstorming = pd.DataFrame(brainstorming_data)
    df_questionnaires = pd.DataFrame(questionnaire_data)
    
    # Add auto-incrementing IDs
    if not df_brainstorming.empty:
        df_brainstorming.insert(0, 'activity_id', range(1, len(df_brainstorming) + 1))
    if not df_questionnaires.empty:
        df_questionnaires.insert(0, 'questionnaire_id', range(1, len(df_questionnaires) + 1))
    
    # Save to CSV
    print("\nSaving CSV files...")
    
    files_created = []
    
    if not df_participants.empty:
        participants_file = output_path / 'participants.csv'
        df_participants.to_csv(participants_file, index=False, encoding='utf-8')
        files_created.append(participants_file)
        print(f"✓ Created: {participants_file} ({len(df_participants)} rows)")
    
    if not df_sessions.empty:
        sessions_file = output_path / 'sessions.csv'
        df_sessions.to_csv(sessions_file, index=False, encoding='utf-8')
        files_created.append(sessions_file)
        print(f"✓ Created: {sessions_file} ({len(df_sessions)} rows)")
    
    if not df_brainstorming.empty:
        brainstorming_file = output_path / 'brainstorming_activities.csv'
        df_brainstorming.to_csv(brainstorming_file, index=False, encoding='utf-8')
        files_created.append(brainstorming_file)
        print(f"✓ Created: {brainstorming_file} ({len(df_brainstorming)} rows)")
    
    if not df_questionnaires.empty:
        questionnaires_file = output_path / 'post_questionnaires.csv'
        df_questionnaires.to_csv(questionnaires_file, index=False, encoding='utf-8')
        files_created.append(questionnaires_file)
        print(f"✓ Created: {questionnaires_file} ({len(df_questionnaires)} rows)")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total sessions processed: {len(sessions)}")
    print(f"Participants: {len(df_participants)}")
    print(f"Sessions: {len(df_sessions)}")
    print(f"Brainstorming activities: {len(df_brainstorming)}")
    print(f"Post-questionnaires: {len(df_questionnaires)}")
    
    if not df_sessions.empty:
        print(f"\nGroup Distribution:")
        print(df_sessions['group_assigned'].value_counts())
    
    if not df_brainstorming.empty:
        print(f"\nMode Distribution:")
        print(df_brainstorming['mode'].value_counts())
        print(f"\nAverage AI usage count: {df_brainstorming['ai_usage_count'].astype(float).mean():.2f}")
        print(f"Average ideas per session: {df_brainstorming['ideas_count'].astype(float).mean():.2f}")
    
    if not df_questionnaires.empty:
        print(f"\nAverage AI Helpfulness: {df_questionnaires['ai_helpfulness'].astype(float).mean():.2f}")
        print(f"Average General Impression: {df_questionnaires['general_impression'].astype(float).mean():.2f}")
    
    print("\n" + "="*60)
    print("✓ All files created successfully!")
    print("="*60)
    
    return files_created

# Main execution
if __name__ == "__main__":
    try:
        # Run the parser
        input_file = 'data/research_study_export_2025-09-27.json'
        output_dir = 'data/transformed'
        
        files = parse_research_study(input_file, output_dir)
        
        print(f"\nFiles saved in: {output_dir}/")
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {input_file}")
        print("Please ensure the file exists in the correct location.")
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON format: {e}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()