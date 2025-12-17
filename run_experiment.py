from PoseNet_evaluation import PoseNetEvaluator

if __name__ == "__main__":
        
    config = {
        'dataset_root': "",
        'split_val': "",
        'save_dir': "",
        'model_dir': "checkpoints/",
        'batch_size': 32
    }

    evaluator = PoseNetEvaluator(config)
    evaluator.run()