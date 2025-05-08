import argparse
import pandas as pd
import chess
import chess.engine
from datasets import load_dataset
from chessian.feature_extraction.stockfish_eval_features import StockfishEvalFeatures
from chessian.feature_extraction.gemini_eval_features import GeminiEvalFeatures
from tqdm import tqdm


def extract_features_for_dataset(df, extractor):
    features_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Extracting features'):
        position_id = row['position_id']
        fen = row['fen']
        board = chess.Board(fen)
        features = extractor.get_features(board)
        features['position_id'] = position_id
        features_list.append(features)
    return features_list


def main():
    parser = argparse.ArgumentParser(description='Extract Stockfish classical features for positions in a Hugging Face dataset.')
    parser.add_argument('--dataset', required=True, help='Hugging Face dataset name (e.g., username/dataset)')
    parser.add_argument('--stockfish', required=True, help='Path to Stockfish executable')
    parser.add_argument('--output', default='features_with_position_id.csv', help='Output CSV file')
    parser.add_argument('--llm', action='store_true', help='Use GeminiEvalFeatures instead of StockfishEvalFeatures')
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if 'train' in dataset:
        df = pd.DataFrame(dataset['train'])
    else:
        default_split = list(dataset.keys())[0]
        df = pd.DataFrame(dataset[default_split])

    if args.llm:
        extractor = GeminiEvalFeatures("gemma-3-27b-it")
        features_list = extract_features_for_dataset(df, extractor)
    else:
        with chess.engine.SimpleEngine.popen_uci(args.stockfish) as stockfish:
            extractor = StockfishEvalFeatures(stockfish)
            features_list = extract_features_for_dataset(df, extractor)
                
    features_df = pd.DataFrame(features_list)
    features_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
