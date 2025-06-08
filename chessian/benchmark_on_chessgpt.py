"""
Benchmark on ChessGPT paper, the position evaluation task.
Download the dataset JSON from https://raw.githubusercontent.com/waterhorse1/ChessGPT/refs/heads/main/eval/eval_task/chess_state_value/chess_state_value_multi_choice_2_nob.json
"""
import os.path
import json
import chess
import chess.pgn
import io
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

from weighted_features_heuristic import WeightedFeaturesHeuristic
from feature_extraction.simple_feature_extractor import SimpleFeatures
from feature_extraction.gemini_eval_features import GeminiEvalFeatures
from feature_extraction.composite_features import CompositeFeatureExtractor
from gemini_evaluator import GeminiEvaluator

FILENAME = "chess_state_value_multi_choice_2_nob.json"
DF_NAME = "chess_state_value_multi_choice_2_nob.csv"


# Weights found by lasso regression
CLASSIC_FEATURES_WEIGHTS = {
    'material_simple': 0.765,
    'mobility_simple': 0.036
}

HYBRID_FEATURES_WEIGHTS = {
    'material_simple' : 0.781, 
    'mobility_simple' : 0.033, 
    'pawn_structure' : 0.597, 
    'mobility' : 0.423, 
    'material' : 0.092, 
    'king_safety' : 0.049, 
    'tempo' : 0.000,
    'total' : 0.000,
}

def score_to_classification(score):
    if score > 2.0:
        return 'white'
    elif score < -2.0:
        return 'black'
    else:
        return 'equal'

def main():
    if not os.path.exists(DF_NAME):
        with open(FILENAME, "r") as f:
            data = json.load(f)
        examples = data["examples"]
        print(f"Number of examples: {len(examples)}")
        records = []
        gemini_feature_extractor = GeminiEvalFeatures('gemma-3-27b-it')
        gemini_evaluator = GeminiEvaluator('gemma-3-27b-it')
        for example in tqdm(examples):
            moves = example['input']
            score = example['score']
            classicifation = {
                'Black has advantage.': 'black',
                'White has advantage.': 'white',
                'The game is equal.': 'equal',
            }[example['target']]
            game = chess.pgn.read_game(io.StringIO(moves))
            position = game.end().board()
            # Our evaluation
            heuristic1 = WeightedFeaturesHeuristic(
                feature_evaluator=SimpleFeatures(),
                weights=CLASSIC_FEATURES_WEIGHTS
            )
            score1 = heuristic1.evaluate_position(position)
            heuristic2 = WeightedFeaturesHeuristic(
                feature_evaluator=SimpleFeatures(),
                weights={
                    'material_simple': 1.0,
                    'mobility_simple': 1.0
                }
            )
            score2 = heuristic2.evaluate_position(position)

            gemini_features = gemini_feature_extractor.get_features(position)
            gemini_direct_score = gemini_evaluator.evaluate_position(position)
            records.append({
                'score': score,
                'gt_classification': classicifation,
                'fen': position.fen(),
                'classic_reg_score': score1,
                'classic_reg_classification': score_to_classification(score1),
                'classic_score': score2,
                'classic_classification': score_to_classification(score2),
                'gemini_direct_score': gemini_direct_score,
                'gemini_direct_classification': score_to_classification(gemini_direct_score),
                **{
                    f'gemini_feature_{k}': v for k, v in gemini_features.items()
                }
            })
        dataset_df = pd.DataFrame(records)
        dataset_df.to_csv(DF_NAME, index=False)
    else:
        dataset_df = pd.read_csv(DF_NAME)
    dataset_size = len(dataset_df)
    def accuracy_of(pred_col_name):
        num_correct = sum(
            dataset_df['gt_classification'] == dataset_df[pred_col_name])
        return num_correct / dataset_size
    accuracy_direct_gemini = accuracy_of('gemini_direct_classification')
    print(f"Accuracy of Gemini direct evaluation: {accuracy_direct_gemini:.4f}")
    eval_score = 0.613 * dataset_df['gemini_feature_mobility'] + 0.336 * dataset_df['gemini_feature_pawn_structure'] + 0.027 * dataset_df['gemini_feature_material'] + 0.000 * dataset_df['gemini_feature_king_safety'] + 0.000 * dataset_df['gemini_feature_tempo'] + 0.000 * dataset_df['gemini_feature_total']
    dataset_df['gemini_regression'] = eval_score.apply(score_to_classification)
    accuracy_gemini_regression = accuracy_of('gemini_regression')
    print(f"Accuracy of Gemini regression evaluation: {accuracy_gemini_regression:.4f}")


if __name__ == "__main__":
    main()
