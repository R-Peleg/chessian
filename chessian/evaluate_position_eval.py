import chess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import importlib.util
import sys

def load_evaluator_from_file(file_path, class_name):
    """
    Dynamically load a class from a Python file.
    
    Args:
        file_path: Path to the Python file containing the evaluator class
        class_name: Name of the evaluator class to load
        
    Returns:
        An instance of the evaluator class
    """
    # Load the module from the file path
    spec = importlib.util.spec_from_file_location("evaluator_module", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluator_module"] = module
    spec.loader.exec_module(module)
    
    # Get the class from the module and instantiate it
    evaluator_class = getattr(module, class_name)
    return evaluator_class()

def load_chess_dataset(dataset_name):
    """
    Load the chess positions dataset from Hugging Face.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        
    Returns:
        A pandas DataFrame containing the chess positions and evaluations
    """
    dataset = load_dataset(dataset_name)
    
    # Convert to pandas DataFrame for easier manipulation
    if 'train' in dataset:
        df = pd.DataFrame(dataset['train'])
    else:
        # Try to get the default split if 'train' is not available
        default_split = list(dataset.keys())[0]
        df = pd.DataFrame(dataset[default_split])
    df = df[~df['bestmove_is_capture'] & ~df['bestmove_is_check'] & ~df['bestmove_is_promotion']]
    return df


def test_evaluator(evaluator, dataset, num_samples=None, filter_criteria=None):
    """
    Test a chess position evaluator against a dataset of positions with known evaluations.
    
    Args:
        evaluator: An object with an evaluate_position method
        dataset: DataFrame containing chess positions and evaluations
        num_samples: Number of samples to test (random subsample if provided)
        filter_criteria: Function to filter positions (takes a DataFrame row and returns boolean)
        
    Returns:
        Dictionary of performance metrics
    """
    # Filter dataset if criteria provided
    if filter_criteria:
        dataset = dataset[dataset.apply(filter_criteria, axis=1)]
    
    # Take a random sample if num_samples is specified
    if num_samples and num_samples < len(dataset):
        dataset = dataset.sample(n=num_samples, random_state=42)
    
    predictions = []
    ground_truth = []
    errors = []
    fens = []
    
    # Create progress bar
    pbar = tqdm(total=len(dataset), desc="Evaluating positions")
    
    # Process each position
    for _, row in dataset.iterrows():
        # Columns:
        # game_id,position_id,fen,evaluation,best_move,bestmove_is_capture,bestmove_is_check,bestmove_is_promotion,random_plies,moves_from_start
        # Create a chess board from FEN
        board = chess.Board(row['fen'])
        
        # Get evaluation from the evaluator
        try:
            pred_eval = evaluator.evaluate_position(board)
            true_eval = row['evaluation']
            if board.turn == chess.BLACK:
                true_eval = -true_eval  # Eval is inverted in GT
            # Clip to (-10, 10) range
            pred_eval = np.clip(pred_eval, -1000, 1000)
            true_eval = np.clip(true_eval, -1000, 1000)
            
            predictions.append(pred_eval)
            ground_truth.append(true_eval)
            errors.append(pred_eval - true_eval)
            fens.append(row['fen'])
        except Exception as e:
            print(f"Error evaluating position {row['position_id']}: {e}")
        
        pbar.update(1)
    
    pbar.close()
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(ground_truth, predictions),
        'rmse': np.sqrt(mean_squared_error(ground_truth, predictions)),
        'mae': mean_absolute_error(ground_truth, predictions),
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'max_error': np.max(np.abs(errors)),
        'max_error_position': fens[np.argmax(np.abs(errors))],
        'correlation': np.corrcoef(ground_truth, predictions)[0, 1],
        'num_positions': len(predictions)
    }
    
    return metrics, predictions, ground_truth

def analyze_by_feature(dataset, predictions, ground_truth, feature_name):
    """
    Analyze evaluator performance across different values of a dataset feature.
    
    Args:
        dataset: DataFrame containing the dataset
        predictions: List of predicted evaluations
        ground_truth: List of true evaluations
        feature_name: Name of the feature to analyze (column in dataset)
        
    Returns:
        Dictionary mapping feature values to performance metrics
    """
    # Create a DataFrame with predictions and ground truth
    results_df = pd.DataFrame({
        'prediction': predictions,
        'ground_truth': ground_truth,
        'error': np.array(predictions) - np.array(ground_truth),
        feature_name: dataset[feature_name].values[:len(predictions)]
    })
    
    # Group by the feature and calculate metrics
    grouped = results_df.groupby(feature_name)
    
    metrics_by_feature = {}
    for group_name, group_data in grouped:
        metrics_by_feature[group_name] = {
            'mae': mean_absolute_error(group_data['ground_truth'], group_data['prediction']),
            'mean_error': group_data['error'].mean(),
            'count': len(group_data)
        }
    
    return metrics_by_feature

def plot_results(predictions, ground_truth, metrics, output_path=None):
    """
    Create visualizations of evaluator performance.
    
    Args:
        predictions: List of predicted evaluations
        ground_truth: List of true evaluations
        metrics: Dictionary of performance metrics
        output_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(12, 10))
    
    # Scatter plot with perfect prediction line
    plt.subplot(2, 2, 1)
    plt.scatter(ground_truth, predictions, alpha=0.5, s=10)
    min_val = min(min(ground_truth), min(predictions))
    max_val = max(max(ground_truth), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Stockfish Evaluation')
    plt.ylabel('Model Evaluation')
    plt.title(f'Evaluation Comparison (r={metrics["correlation"]:.3f})')
    
    # Histogram of errors
    plt.subplot(2, 2, 2)
    errors = np.array(predictions) - np.array(ground_truth)
    plt.hist(errors, bins=50)
    plt.xlabel('Error (Model - Stockfish)')
    plt.ylabel('Count')
    plt.title(f'Error Distribution (MAE={metrics["mae"]:.2f})')
    
    # Plot errors vs. true eval
    plt.subplot(2, 2, 3)
    plt.scatter(ground_truth, errors, alpha=0.5, s=10)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Stockfish Evaluation')
    plt.ylabel('Error (Model - Stockfish)')
    plt.title('Error vs. Stockfish Evaluation')
    
    # Print metrics as text
    plt.subplot(2, 2, 4)
    plt.axis('off')
    metrics_text = '\n'.join([
        f"Number of positions: {metrics['num_positions']}",
        f"Mean Squared Error: {metrics['mse']:.2f}",
        f"Root Mean Squared Error: {metrics['rmse']:.2f}",
        f"Mean Absolute Error: {metrics['mae']:.2f}",
        f"Mean Error: {metrics['mean_error']:.2f}",
        f"Median Error: {metrics['median_error']:.2f}",
        f"Maximum Absolute Error: {metrics['max_error']:.2f}",
        f"Correlation: {metrics['correlation']:.3f}"
    ])
    plt.text(0.05, 0.95, metrics_text, va='top', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Test a chess position evaluator against a dataset')
    parser.add_argument('dataset_name', default='ReuvenP/chess_eval_for_search', help='Name of the dataset on Hugging Face (e.g., "username/chess-positions")')
    parser.add_argument('--samples', type=int, default=None, help='Number of random positions to test')
    parser.add_argument('--output', default=None, help='Path to save the results plot')
    parser.add_argument('--random-only', action='store_true', help='Only test positions with random moves applied')
    parser.add_argument('--actual-only', action='store_true', help='Only test positions from actual games (no random moves)')
    
    args = parser.parse_args()
    
    # Load the evaluator
    from chessian.evaluator import Evaluator
    from chessian.chatgpt_code_evaluator import CharGPTCodeEvaluator
    from chessian.random_evluator import RandomEvaluator
    from gemini_evaluator import GeminiEvaluator
    evaluator = Evaluator()
    
    # Load the dataset
    print(f"Loading dataset from {args.dataset_name}")
    dataset = load_chess_dataset(args.dataset_name)
    
    # Define filter criteria based on arguments
    def filter_criteria(row):
        if args.random_only and row['random_plies'] == 0:
            return False
        if args.actual_only and row['random_plies'] > 0:
            return False
        return True
    
    # Test the evaluator
    print("Testing evaluator...")
    metrics, predictions, ground_truth = test_evaluator(
        evaluator, 
        dataset, 
        num_samples=args.samples,
        filter_criteria=filter_criteria
    )
    
    # Print overall metrics
    print("\nOverall Performance:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    if 'random_plies' in dataset.columns:
        print("\nPerformance by Number of Random Plies:")
        random_metrics = analyze_by_feature(dataset, predictions, ground_truth, 'random_plies')
        for plies, m in sorted(random_metrics.items()):
            print(f"Random Plies {plies} ({m['count']} positions): MAE={m['mae']:.2f}, Mean Error={m['mean_error']:.2f}")
    
    # Plot results
    plot_results(predictions, ground_truth, metrics, args.output)

if __name__ == "__main__":
    main()
