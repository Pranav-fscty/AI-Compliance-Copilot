import mlflow

mlflow.set_experiment("compliance-copilot-retrieval")

with mlflow.start_run(run_name="MiniLM_heading_chunks_baseline"):
    mlflow.log_param("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    mlflow.log_param("chunking_strategy", "markdown_heading_based")
    mlflow.log_param("retrieval_k", 8)
    mlflow.log_param("llm", "llama3.1")
    mlflow.log_metric("retrieval_accuracy", 0.85)
    mlflow.log_metric("avg_judge_score", 3.35)
    mlflow.log_metric("pct_acceptable", 0.80)
    mlflow.log_metric("total_questions", 20)
    print("Logged baseline run successfully")