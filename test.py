from transformers import TrainingArguments

# Configuration for a full fine-tuning run
training_args = TrainingArguments(
    # Output directory to save model checkpoints
    output_dir="./results",

    # --- Core Training Hyperparameters ---
    # The number of complete passes through the training data
    num_train_epochs=3,
    
    # Batch size per GPU for training
    per_device_train_batch_size=2,
    
    # Accumulate gradients over 8 steps to simulate a larger batch size
    gradient_accumulation_steps=8, 

    # --- Optimizer and Scheduler Hyperparameters ---
    # The initial learning rate for the AdamW optimizer
    learning_rate=2e-5,
    
    # Regularization to prevent overfitting
    weight_decay=0.01,
    
    # Number of steps for the linear warmup phase
    warmup_steps=500,

    # --- Logging and Saving ---
    # How often to save the model checkpoint
    save_strategy="epoch",
    
    # How often to log training metrics
    logging_steps=50
)

# This `training_args` object would then be passed to the Trainer
# along with the model, dataset, and tokenizer.
